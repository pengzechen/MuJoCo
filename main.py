"""MuJoCo demo: a four-wheel robot with a 3-servo arm picking up a tennis ball."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import glfw
import mujoco
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
MODEL_PATH = ROOT / "scene.xml"
SEQUENCE_PATH = ROOT / "sequences.json"

# Sliding friction of the ball on the floor. Overrides the ball_floor contact
# pair in scene.xml, and only that pair - the wheels and the rest of the scene
# keep their own friction. 2.2 is what the scene effectively had before the pair
# existed (the floor's 2.2 beating the ball's 1.8 under the max rule), and it is
# also near the middle of the range that keeps pick_ball working: below ~1.8 the
# ball slides out from under the fingers, above ~2.5 the fingers press it clean
# through the 6 cm floor slab.
BALL_FLOOR_FRICTION = 2.2

# Keys nudging the ball-on-floor friction by this much per press.
FRICTION_KEYS = {
    glfw.KEY_LEFT_BRACKET: -0.1,
    glfw.KEY_RIGHT_BRACKET: 0.1,
}

# Arm servos driven by the on-screen sliders, in Controller.arm_ids order.
ARM_AXES = ("shoulder", "elbow", "gripper")

# Duration stored for each pose saved off the sliders, in seconds. Edit the
# saved steps in sequences.json by hand to pace them differently.
SAVED_STEP_DURATION = 0.6

# Slider panel geometry, in framebuffer pixels with a bottom-left origin. At
# mjFONTSCALE_150 the font is ~10 px per character ("shoulder" is 81 px), and
# mjr_label silently drops the text of a box shorter than ~28 px, so the label
# boxes are wider and taller than the glyphs alone would suggest.
SLIDER_PANEL_LEFT = 16
SLIDER_PANEL_BOTTOM = 16
SLIDER_LABEL_WIDTH = 100
SLIDER_VALUE_WIDTH = 90
SLIDER_BOX_HEIGHT = 30
SLIDER_ROW_HEIGHT = 34
SLIDER_TRACK_WIDTH = 200
SLIDER_TRACK_HEIGHT = 8
SLIDER_GAP = 12
SLIDER_KNOB_WIDTH = 10
SLIDER_KNOB_HEIGHT = 20
SLIDER_HIT_MARGIN = 8


def slider_row_bottom(index: int) -> int:
    return SLIDER_PANEL_BOTTOM + index * SLIDER_ROW_HEIGHT


def slider_track_rect(index: int) -> tuple[int, int, int, int]:
    """(left, bottom, width, height) of one slider track, framebuffer pixels."""
    bottom = slider_row_bottom(index) + (SLIDER_BOX_HEIGHT - SLIDER_TRACK_HEIGHT) // 2
    left = SLIDER_PANEL_LEFT + SLIDER_LABEL_WIDTH + SLIDER_GAP
    return left, bottom, SLIDER_TRACK_WIDTH, SLIDER_TRACK_HEIGHT


def format_sequences(sequences: dict[str, list[dict[str, float]]]) -> str:
    """Render sequences.json the way it is hand-written: one step per line."""
    lines = ["{"]
    for name_index, (name, steps) in enumerate(sequences.items()):
        lines.append(f'  "{name}": [')
        for step_index, step in enumerate(steps):
            body = ", ".join(f'"{key}": {json.dumps(value)}' for key, value in step.items())
            lines.append(f"    {{{body}}}{',' if step_index + 1 < len(steps) else ''}")
        lines.append("  ]" + ("," if name_index + 1 < len(sequences) else ""))
    lines.append("}")
    return "\n".join(lines) + "\n"


class Controller:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.keys: set[int] = set()
        self.paused = False
        self.sequence_name = ""
        self.sequence: list[dict[str, float]] = []
        self.step_index = 0
        self.step_started = 0.0
        self.grasped = False
        self.grasp_offset = np.zeros(3)
        self.robot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qpos = int(model.jnt_qposadr[ball_joint])
        self.ball_qvel = int(model.jnt_dofadr[ball_joint])
        caster_swivel_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "caster_swivel")
        self.caster_qpos = int(model.jnt_qposadr[caster_swivel_joint])
        self.caster_qvel = int(model.jnt_dofadr[caster_swivel_joint])
        self.drive_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_left_drive"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_right_drive"),
        ]
        self.arm_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder_servo"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_servo"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_servo"),
        ]
        self.ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tennis_ball")
        self.gripper_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_tip")
        self.finger_bodies = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
        ]
        self.weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_grasp")
        self.pair_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, "ball_floor")
        self.ball_floor_friction = 0.0
        self.set_ball_floor_friction(BALL_FLOOR_FRICTION)
        self.slider_values = np.array([data.ctrl[actuator] for actuator in self.arm_ids], dtype=float)
        self.slider_limits = [tuple(model.actuator_ctrlrange[actuator]) for actuator in self.arm_ids]
        self.dragging = -1
        # Framebuffer size plus the window->framebuffer scale, refreshed every
        # frame so the mouse callbacks can convert cursor positions without
        # calling GLFW (and stay testable without a window).
        self.viewport = (0.0, 0.0, 1.0, 1.0)
        try:
            self.manual_steps = len(self.read_sequences().get("manual", []))
        except (OSError, ValueError):
            # A missing or hand-broken sequences.json only matters when a
            # sequence is actually loaded; it must not stop the demo starting.
            self.manual_steps = 0

    def read_sequences(self) -> dict[str, list[dict[str, float]]]:
        return json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))

    def set_ball_floor_friction(self, sliding: float) -> None:
        """Set the sliding friction of the ball on the floor.

        mjModel.pair_friction holds the raw contact layout (tangent1, tangent2,
        spin, roll1, roll2) and is copied into the contact verbatim, so the
        single sliding value has to be written to both tangent slots.
        """
        self.ball_floor_friction = max(0.0, float(sliding))
        if self.pair_id >= 0:
            self.model.pair_friction[self.pair_id, :2] = self.ball_floor_friction

    def adjust_ball_floor_friction(self, key: int) -> None:
        """Apply a FRICTION_KEYS nudge; other keys are ignored."""
        if self.pair_id < 0 or key not in FRICTION_KEYS:
            return
        self.set_ball_floor_friction(self.ball_floor_friction + FRICTION_KEYS[key])

    def load_sequence(self, name: str) -> None:
        sequences = self.read_sequences()
        if name not in sequences:
            raise KeyError(f"sequence not found: {name}")
        self.sequence_name = name
        self.sequence = sequences[name]
        self.step_index = 0
        self.step_started = time.monotonic()

    def set_arm_targets(self, targets: dict[str, float]) -> None:
        self.data.ctrl[self.arm_ids[0]] = float(targets["shoulder"])
        self.data.ctrl[self.arm_ids[1]] = float(targets["elbow"])
        self.data.ctrl[self.arm_ids[2]] = float(targets["gripper"])

    def update_sequence(self) -> None:
        if not self.sequence:
            return
        step = self.sequence[self.step_index]
        self.set_arm_targets(step)
        if time.monotonic() - self.step_started >= float(step["duration"]):
            if self.step_index + 1 < len(self.sequence):
                self.step_index += 1
                self.step_started = time.monotonic()
            else:
                self.sequence = []
                self.sequence_name = ""

    def sync_sliders(self) -> None:
        """Let the knobs follow the servos, except the one being dragged."""
        for index, actuator in enumerate(self.arm_ids):
            if index != self.dragging:
                self.slider_values[index] = float(self.data.ctrl[actuator])

    def slider_fraction(self, index: int) -> float:
        """Where one knob sits along its track, 0..1."""
        lower, upper = self.slider_limits[index]
        if upper <= lower:
            return 0.0
        return float(np.clip((self.slider_values[index] - lower) / (upper - lower), 0.0, 1.0))

    def value_from_x(self, index: int, x: float) -> float:
        """Map a framebuffer x coordinate onto one slider's actuator range."""
        lower, upper = self.slider_limits[index]
        left, _, width, _ = slider_track_rect(index)
        fraction = (x - left) / width
        return float(np.clip(lower + fraction * (upper - lower), lower, upper))

    def set_slider(self, index: int, x: float) -> None:
        """Move one slider to a framebuffer x coordinate and command its servo."""
        value = self.value_from_x(index, x)
        self.slider_values[index] = value
        self.data.ctrl[self.arm_ids[index]] = value

    def to_framebuffer(self, x: float, y: float) -> tuple[float, float]:
        """GLFW cursor coordinates (top-left origin) -> framebuffer pixels.

        MjrRect counts from the bottom-left, so y has to be flipped, and scaled
        by the framebuffer/window ratio or HiDPI displays end up off by 2x.
        """
        _, height, scale_x, scale_y = self.viewport
        return x * scale_x, height - y * scale_y

    def begin_drag(self, x: float, y: float) -> bool:
        """Grab the slider under a framebuffer point, if there is one."""
        for index in range(len(self.arm_ids)):
            left, bottom, width, height = slider_track_rect(index)
            inside = (
                left - SLIDER_HIT_MARGIN <= x <= left + width + SLIDER_HIT_MARGIN
                and bottom - SLIDER_HIT_MARGIN <= y <= bottom + height + SLIDER_HIT_MARGIN
            )
            if inside:
                self.dragging = index
                # The sequence player and the sliders both write data.ctrl, so
                # taking a slider hands the arm over and stops the playback.
                self.sequence = []
                self.sequence_name = ""
                self.set_slider(index, x)
                return True
        return False

    def drag_to(self, x: float) -> None:
        """Follow the cursor while dragging; y is ignored so the drag can roam."""
        if self.dragging >= 0:
            self.set_slider(self.dragging, x)

    def end_drag(self) -> None:
        self.dragging = -1

    def save_pose(self) -> int:
        """Append the current slider pose to the "manual" sequence on disk."""
        sequences = self.read_sequences()
        step = {axis: round(float(value), 4) for axis, value in zip(ARM_AXES, self.slider_values)}
        step["duration"] = SAVED_STEP_DURATION
        steps = sequences.setdefault("manual", [])
        steps.append(step)
        SEQUENCE_PATH.write_text(format_sequences(sequences), encoding="utf-8")
        self.manual_steps = len(steps)
        print(f"saved pose {self.manual_steps} to {SEQUENCE_PATH.name} [manual]")
        return self.manual_steps

    def draw_sliders(self, viewport: mujoco.MjrRect, context: mujoco.MjrContext) -> None:
        """Draw the slider panel over the main viewport, after mjr_render.

        Every string goes through mjr_label, which fills its rectangle and
        centres the text in it - mjr_text takes viewport fractions rather than
        pixels, and a box shorter than ~28 px silently loses its text entirely.
        """
        if viewport.width <= 0 or viewport.height <= 0:
            return
        font = mujoco.mjtFont.mjFONT_NORMAL
        panel_width = SLIDER_LABEL_WIDTH + SLIDER_GAP + SLIDER_TRACK_WIDTH + SLIDER_GAP + SLIDER_VALUE_WIDTH
        panel_height = SLIDER_PANEL_BOTTOM + len(ARM_AXES) * SLIDER_ROW_HEIGHT
        mujoco.mjr_rectangle(mujoco.MjrRect(SLIDER_PANEL_LEFT, 0, panel_width, panel_height), 0.07, 0.08, 0.11, 1.0)

        for index, axis in enumerate(ARM_AXES):
            bottom = slider_row_bottom(index)
            left, track_bottom, width, height = slider_track_rect(index)
            mujoco.mjr_rectangle(mujoco.MjrRect(left, track_bottom, width, height), 0.16, 0.18, 0.22, 1.0)
            filled = int(round(width * self.slider_fraction(index)))
            if filled:
                mujoco.mjr_rectangle(
                    mujoco.MjrRect(left, track_bottom, filled, height), 0.20, 0.45, 0.85, 1.0
                )
            knob_left = int(
                np.clip(
                    left + filled - SLIDER_KNOB_WIDTH // 2,
                    left - SLIDER_KNOB_WIDTH // 2,
                    left + width - SLIDER_KNOB_WIDTH // 2,
                )
            )
            mujoco.mjr_rectangle(
                mujoco.MjrRect(
                    knob_left,
                    track_bottom - (SLIDER_KNOB_HEIGHT - height) // 2,
                    SLIDER_KNOB_WIDTH,
                    SLIDER_KNOB_HEIGHT,
                ),
                0.95,
                0.72,
                0.20,
                1.0,
            )
            mujoco.mjr_label(
                mujoco.MjrRect(SLIDER_PANEL_LEFT, bottom, SLIDER_LABEL_WIDTH, SLIDER_BOX_HEIGHT),
                font,
                axis,
                0.12,
                0.14,
                0.18,
                1.0,
                0.80,
                0.84,
                0.90,
                context,
            )
            mujoco.mjr_label(
                mujoco.MjrRect(left + width + SLIDER_GAP, bottom, SLIDER_VALUE_WIDTH, SLIDER_BOX_HEIGHT),
                font,
                f"{self.slider_values[index]:+.3f}",
                0.12,
                0.14,
                0.18,
                1.0,
                0.95,
                0.80,
                0.35,
                context,
            )

    def update_drive(self) -> None:
        forward = int(glfw.KEY_W in self.keys) - int(glfw.KEY_S in self.keys)
        turn = int(glfw.KEY_A in self.keys) - int(glfw.KEY_D in self.keys)
        speed = 5.0
        left = np.clip(speed * (forward + 0.55 * turn), -8, 8)
        right = np.clip(speed * (forward - 0.55 * turn), -8, 8)
        self.data.ctrl[self.drive_ids[0]] = left
        self.data.ctrl[self.drive_ids[1]] = right

        # Apply a direct planar chassis command. Wheel contact alone is too
        # sensitive to the simplified wheel geometry for a reliable demo.
        rotation = self.data.xmat[self.robot_body].reshape(3, 3)
        local_forward = -rotation[:, 0]
        local_velocity = rotation.T @ self.data.qvel[:3]
        forward_velocity = -local_velocity[0]
        target_forward_velocity = 1.2 * forward
        force_local = np.array(
            [28.0 * (target_forward_velocity - forward_velocity), -10.0 * local_velocity[1], 0.0]
        )
        self.data.xfrc_applied[self.robot_body, :3] = local_forward * force_local[0] + rotation[:, 1] * force_local[1]
        self.data.xfrc_applied[self.robot_body, 5] = 55.0 * turn - 10.0 * self.data.qvel[5]
        if not forward and not turn:
            self.data.xfrc_applied[self.robot_body, :3] -= 4.0 * self.data.qvel[:3]

        self.update_caster_visual(turn)

    def update_caster_visual(self, turn: int) -> None:
        rotation = self.data.xmat[self.robot_body].reshape(3, 3)
        local_velocity = rotation.T @ self.data.qvel[:3]
        caster_offset = np.array([0.48, 0.0, 0.0])
        angular_velocity = rotation.T @ self.data.qvel[3:6]
        caster_velocity = local_velocity + np.cross(angular_velocity, caster_offset)
        if np.linalg.norm(caster_velocity[:2]) < 0.02 and turn:
            caster_velocity[:2] = [0.0, 0.48 * turn]
        if np.linalg.norm(caster_velocity[:2]) < 0.02:
            return

        self.data.qpos[self.caster_qpos] = math.atan2(caster_velocity[1], caster_velocity[0])
        self.data.qvel[self.caster_qvel] = 0.0

    def try_grasp(self) -> None:
        # Activate the weld only after the gripper is close and closing.
        ball_pos = self.data.xpos[self.ball_body]
        gripper_distance = np.linalg.norm(ball_pos - self.data.xpos[self.gripper_body])
        finger_distance = min(np.linalg.norm(ball_pos - self.data.xpos[body]) for body in self.finger_bodies)
        gripper_target = self.data.ctrl[self.arm_ids[2]]
        if not self.grasped and (gripper_distance < 0.22 or finger_distance < 0.19) and gripper_target < 0.05:
            gripper_rotation = self.data.xmat[self.gripper_body].reshape(3, 3)
            self.grasp_offset = gripper_rotation.T @ (ball_pos - self.data.xpos[self.gripper_body])
            self.grasped = True

    def update_grasped_ball(self) -> None:
        if not self.grasped:
            return
        gripper_rotation = self.data.xmat[self.gripper_body].reshape(3, 3)
        self.data.qpos[self.ball_qpos : self.ball_qpos + 3] = self.data.xpos[self.gripper_body] + (
            gripper_rotation @ self.grasp_offset
        )
        self.data.qvel[self.ball_qvel : self.ball_qvel + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset_grasp(self) -> None:
        """Let go of the ball: open the fingers and stop tracking it.

        Opening them is not just cosmetic. try_grasp only fires while the
        gripper is commanded shut, so leaving it closed would re-grasp the ball
        on the very next step and the release would look like a no-op.
        """
        self.data.eq_active[self.weld_id] = 0
        self.grasped = False
        _, open_position = self.model.actuator_ctrlrange[self.arm_ids[2]]
        self.data.ctrl[self.arm_ids[2]] = open_position

    def reset_ball(self) -> None:
        """Put the ball back on its starting spot, at rest."""
        # Let go first, otherwise update_grasped_ball would drag it straight
        # back to the gripper on the next step.
        self.reset_grasp()
        span = slice(self.ball_qpos, self.ball_qpos + 7)
        self.data.qpos[span] = self.model.qpos0[span]
        self.data.qvel[self.ball_qvel : self.ball_qvel + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)


def key_callback(window: object, key: int, scancode: int, action: int, mods: int, controller: Controller) -> None:
    del window, scancode, mods
    if action == glfw.PRESS:
        controller.keys.add(key)
        if key == glfw.KEY_SPACE:
            controller.paused = not controller.paused
        elif key == glfw.KEY_1:
            controller.load_sequence("pick_ball")
        elif key == glfw.KEY_2:
            controller.load_sequence("home")
        elif key == glfw.KEY_R:
            controller.reset_grasp()
        elif key == glfw.KEY_B:
            controller.reset_ball()
        elif key == glfw.KEY_P:
            controller.save_pose()
        elif key == glfw.KEY_3:
            # Nothing saved yet is not an error, the sequence just does not exist.
            if "manual" in controller.read_sequences():
                controller.load_sequence("manual")
        else:
            controller.adjust_ball_floor_friction(key)
    elif action == glfw.REPEAT:
        # A held friction key keeps ticking, so one press can sweep the range.
        controller.adjust_ball_floor_friction(key)
    elif action == glfw.RELEASE:
        controller.keys.discard(key)


def mouse_button_callback(window: object, button: int, action: int, mods: int, controller: Controller) -> None:
    del mods
    if button != glfw.MOUSE_BUTTON_LEFT:
        return
    if action == glfw.RELEASE:
        controller.end_drag()
    elif action == glfw.PRESS:
        controller.begin_drag(*controller.to_framebuffer(*glfw.get_cursor_pos(window)))


def cursor_position_callback(window: object, xpos: float, ypos: float, controller: Controller) -> None:
    del window
    controller.drag_to(controller.to_framebuffer(xpos, ypos)[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ball-floor-friction",
        type=float,
        metavar="MU",
        help="sliding friction of the ball on the floor, overriding BALL_FLOOR_FRICTION",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    controller = Controller(model, data)
    if args.ball_floor_friction is not None:
        controller.set_ball_floor_friction(args.ball_floor_friction)

    if not glfw.init():
        raise RuntimeError("GLFW initialization failed; a graphical session is required")
    window = glfw.create_window(1280, 720, "MuJoCo Tennis Robot", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("could not create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_key_callback(window, lambda w, k, s, a, m: key_callback(w, k, s, a, m, controller))
    glfw.set_mouse_button_callback(window, lambda w, b, a, m: mouse_button_callback(w, b, a, m, controller))
    glfw.set_cursor_pos_callback(window, lambda w, x, y: cursor_position_callback(w, x, y, controller))

    cam = mujoco.MjvCamera()
    first_person_cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam.azimuth = 135
    cam.elevation = -25
    cam.distance = 4.4
    cam.lookat[:] = [0.2, 0, 0.35]
    first_person_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    first_person_cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")
    viewport = mujoco.MjrRect(0, 0, 0, 0)
    first_person_viewport = mujoco.MjrRect(0, 0, 0, 0)
    last = time.monotonic()

    while not glfw.window_should_close(window):
        now = time.monotonic()
        elapsed = min(now - last, 0.05)
        last = now
        # Releasing the button outside the window never reaches our callback, so
        # poll for it instead of leaving a slider stuck to the cursor.
        if controller.dragging >= 0 and glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.RELEASE:
            controller.end_drag()
        controller.update_drive()
        controller.update_sequence()
        controller.sync_sliders()
        if not controller.paused:
            for _ in range(max(1, math.ceil(elapsed / model.opt.timestep))):
                mujoco.mj_step(model, data)
            controller.try_grasp()
            controller.update_grasped_ball()

        width, height = glfw.get_framebuffer_size(window)
        viewport.width, viewport.height = width, height
        window_width, window_height = glfw.get_window_size(window)
        controller.viewport = (
            width,
            height,
            width / window_width if window_width else 1.0,
            height / window_height if window_height else 1.0,
        )
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        first_person_viewport.width = max(260, width // 4)
        first_person_viewport.height = max(146, height // 4)
        first_person_viewport.left = width - first_person_viewport.width - 12
        first_person_viewport.bottom = height - first_person_viewport.height - 12
        mujoco.mjv_updateScene(model, data, opt, None, first_person_cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(first_person_viewport, scene, context)
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            first_person_viewport,
            "front camera",
            "",
            context,
        )
        status = "paused" if controller.paused else "running"
        sequence = controller.sequence_name or "idle"
        grasp = "grasped" if controller.grasped else "open"
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            f"W/S: forward/back   A/D: turn\n"
            f"1: pick_ball   2: home   3: manual   Space: pause   R: release   B: ball home\n"
            f"[ / ] ball friction: {controller.ball_floor_friction:.2f}"
            f"    P: save pose (manual: {controller.manual_steps})\n"
            f"state: {status} | sequence: {sequence} | gripper: {grasp}",
            "",
            context,
        )
        controller.draw_sliders(viewport, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    run()
