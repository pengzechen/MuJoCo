"""MuJoCo demo: a four-wheel robot with a 3-servo arm picking up a tennis ball."""

from __future__ import annotations

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
        self.robot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
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
        self.weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_grasp")

    def load_sequence(self, name: str) -> None:
        sequences = json.loads(SEQUENCE_PATH.read_text(encoding="utf-8"))
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
        distance = np.linalg.norm(self.data.xpos[self.ball_body] - self.data.xpos[self.gripper_body])
        gripper_target = self.data.ctrl[self.arm_ids[2]]
        if not self.grasped and distance < 0.24 and gripper_target < 0.04:
            self.data.eq_active[self.weld_id] = 1
            self.grasped = True

    def reset_grasp(self) -> None:
        self.data.eq_active[self.weld_id] = 0
        self.grasped = False


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
    elif action == glfw.RELEASE:
        controller.keys.discard(key)


def run() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    controller = Controller(model, data)

    if not glfw.init():
        raise RuntimeError("GLFW initialization failed; a graphical session is required")
    window = glfw.create_window(1280, 720, "MuJoCo Tennis Robot", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("could not create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_key_callback(window, lambda w, k, s, a, m: key_callback(w, k, s, a, m, controller))

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam.azimuth = 135
    cam.elevation = -25
    cam.distance = 4.4
    cam.lookat[:] = [0.2, 0, 0.35]
    viewport = mujoco.MjrRect(0, 0, 0, 0)
    last = time.monotonic()

    while not glfw.window_should_close(window):
        now = time.monotonic()
        elapsed = min(now - last, 0.05)
        last = now
        controller.update_drive()
        controller.update_sequence()
        if not controller.paused:
            for _ in range(max(1, math.ceil(elapsed / model.opt.timestep))):
                mujoco.mj_step(model, data)
            controller.try_grasp()

        width, height = glfw.get_framebuffer_size(window)
        viewport.width, viewport.height = width, height
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        status = "paused" if controller.paused else "running"
        sequence = controller.sequence_name or "idle"
        grasp = "grasped" if controller.grasped else "open"
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            f"W/S: forward/back   A/D: turn\n1: pick_ball   2: home   Space: pause   R: release\nstate: {status} | sequence: {sequence} | gripper: {grasp}",
            "",
            context,
        )
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    run()
