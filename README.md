# MuJoCo 网球抓取小车

这是一个自包含的 MuJoCo Python 示例，包含：

- 一个带自由运动底盘的三轮小车
- 前左、前右两个独立主动轮
- 车尾中间一个被动万向轮
- 大臂、小臂、夹爪三个舵机
- 一个放在地面上的可碰撞网球
- 一圈接近小车高度的围挡，防止网球滚出场地
- 可编辑的舵机动作序列，以及自动夹球 weld

## 安装和运行

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

如果系统禁止全局 pip 安装，可以先创建虚拟环境：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## 操作

- `W` / `S`：小车前进 / 后退
- `A` / `D`：差速左转 / 右转
- `1`：加载并执行 `pick_ball` 序列
- `2`：加载并执行 `home` 序列
- `Space`：暂停 / 继续
- `R`：释放已经抓住的网球

## 编辑舵机序列

直接编辑 `sequences.json`，每一步包含三个舵机目标角度和停留时间：

```json
{
  "shoulder": 0.55,
  "elbow": -1.25,
  "gripper": 0.015,
  "duration": 0.7
}
```

角度单位是弧度；夹爪是滑动舵机，范围是 `0` 到 `0.10` 米，数值越小表示夹紧。程序在按下 `1` 或 `2` 时重新读取 JSON，因此修改文件后无需修改代码。

## 模型说明

场景在 `scene.xml` 中定义。当前底盘使用前双轮差速驱动、后中置万向轮支撑，适合先验证小车和机械臂控制流程。若要提高底盘运动精度，可以把主动轮替换为带轮地切向摩擦方向的轮胎 mesh，并加入更完整的轮地动力学。
