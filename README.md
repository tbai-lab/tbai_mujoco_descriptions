# tbai_mujoco_descriptions

Robot MuJoCo (MJCF) descriptions and interactive visualizer.

<div align="center">
<img src="assets/screen2.png" width="800">
</div>

## Setup

```bash
git clone --recurse-submodules git@github.com:tbai-lab/tbai_mujoco_descriptions.git
```

## Usage

```bash
uv run python visualize_mjcf.py              # list available robots
uv run python visualize_mjcf.py --robot go2   # visualize specified robot
uv run python view_robot.py --robot go2       # simulate specified robot
uv run python sit_stand.py --robot go2        # PD-controlled sit/stand demo
```

## Robots

<div align="center">

| Robot | Description | Reference | License |
|-------|-------------|-----------|---------|
| [__ANYmal B__](robots/anymal_b/) | 12-DoF quadruped from ANYbotics | [Link](https://www.anybotics.com/) | - |
| [__ANYmal C__](robots/anymal_c/) | 12-DoF quadruped from ANYbotics | [Link](https://www.anybotics.com/) | - |
| [__ANYmal D__](robots/anymal_d/) | 12-DoF quadruped from ANYbotics | [Link](https://www.anybotics.com/) | - |
| [__Franka Panda__](robots/franka_panda/) | 7-DoF manipulator with gripper from Franka Robotics | [Link](https://franka.de/) | - |
| [__G1__](robots/g1/) | 29-DoF humanoid from Unitree Robotics | [Link](https://www.unitree.com/) | - |
| [__Go2__](robots/go2/) | 12-DoF quadruped from Unitree Robotics | [Link](https://www.unitree.com/) | - |
| [__Go2W__](robots/go2w/) | 16-DoF wheeled version of Go2 | [Link](https://www.unitree.com/) | - |
| [__Spot__](robots/spot/) | 12-DoF quadruped from Boston Dynamics | [Link](https://bostondynamics.com/) | - |
| [__Spot Arm__](robots/spot_arm/) | 12-DoF quadruped + 7-DoF arm from Boston Dynamics | [Link](https://bostondynamics.com/) | - |

</div>
