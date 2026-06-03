# Seeed reBot Arm B601 Follower Integration with LeRobot

This repository provides the **Follower Arm (Robot)** integration for the **reBot Arm B601** with the [LeRobot](https://github.com/huggingface/lerobot) framework. It enables the B601 arm to be used as a follower robot in teleoperation and data collection workflows.

## Supported Hardware

*   **Robot**: Seeed reBot Arm B601 Series (6-DOF + Gripper)
*   **Motors**: Damiao (DM4340 + DM4310), RobStride
*   **Communication**: CAN bus via USB-CAN adapter, including SocketCAN-compatible adapters and Damiao's USB2CAN adapter

## Installation

1.  **Install LeRobot**:
    Follow the instructions in the [LeRobot repository](https://github.com/huggingface/lerobot) to install the base library. A very quick summary is shown below.

    ```shell
    conda create -y -n lerobot python=3.12
    conda activate lerobot
    conda install ffmpeg -c conda-forge
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    pip install -e .
    ```

2.  **Install the `motorbridge` Python package**:

    ```shell
    # Go to https://github.com/tianrking/motorbridge/releases
    # Download the wheel that matches your platform and Python version
    # Example for Ubuntu x86_64 with Python 3.12:
    pip install motorbridge-0.1.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

    # TODO: switch to pip install from PyPI once published
    ```

    You can also install the optional `motor_cli` utility:

    ```shell
    wget https://github.com/tianrking/motorbridge/releases/download/v0.1.5/motor-cli-v0.1.5-linux-x86_64.tar.gz
    tar zxvf motor-cli-v0.1.5-linux-x86_64.tar.gz
    sudo cp motor-cli-v0.1.5-linux-x86_64/bin/motor_cli /usr/local/bin/
    sudo chmod a+x /usr/local/bin/motor_cli
    ```

3.  **Install this package**:
    Clone this repository and install it in editable mode:

    ```bash
    git clone https://github.com/Seeed-Projects/lerobot-robot-seeed-b601.git
    cd lerobot-robot-seeed-b601
    pip install -e .
    ```

    Or install from PyPI:

    ```bash
    pip install lerobot-robot-seeed-b601
    ```

    Upon installation, two robot variants are registered:
    *   `seeed_b601_dm_follower`: B601 follower using Damiao motors - **Primary supported path**
    *   `seeed_b601_rs_follower`: B601 follower using RobStride motors - **Registered, still being refined**

    ```shell
    # Verify that the follower configs are visible to LeRobot
    lerobot-teleoperate --help | grep SeeedB601

    # Expected output includes
    SeeedB601DMFollowerConfig ['robot']:
    SeeedB601RSFollowerConfig ['robot']:
    ```

## Configuration

Default motor mapping for the B601 follower:

*   `shoulder_pan`: master ID `0x01`, feedback ID `0x11`, motor model `dm4340p`
*   `shoulder_lift`: master ID `0x02`, feedback ID `0x12`, motor model `dm4340p`
*   `elbow_flex`: master ID `0x03`, feedback ID `0x13`, motor model `dm4340p`
*   `wrist_flex`: master ID `0x04`, feedback ID `0x14`, motor model `dm4310`
*   `wrist_yaw`: master ID `0x05`, feedback ID `0x15`, motor model `dm4310`
*   `wrist_roll`: master ID `0x06`, feedback ID `0x16`, motor model `dm4310`
*   `gripper`: master ID `0x07`, feedback ID `0x17`, motor model `dm4310`

*   **CAN adapter types**:
    *   `socketcan`: for SocketCAN-compatible adapters such as `can0`
    *   `damiao`: for Damiao serial bridge adapters such as `/dev/ttyACM0`
    *   `robstride`: registered in config, but the dedicated adapter path is not yet supported by the current `motorbridge` Python SDK integration

Make sure your actual motor names, IDs, wiring, and motor models match these defaults before running calibration or teleoperation.

Direction and gripper scale conversion are now owned by the follower integration. For the DM follower, these defaults live in `lerobot_robot_seeed_b601/config_seeed_b601_dm_follower.py`; for the RS follower, they live in `lerobot_robot_seeed_b601/config_seeed_b601_rs_follower.py`.

## Usage

### Pair a B601 follower with a reBot Arm 102 leader

```shell
lerobot-teleoperate \
    --robot.type=seeed_b601_dm_follower \
    --robot.id=follower1 \
    --robot.port=/dev/ttyACM4 \
    --robot.can_adapter=damiao \
    --teleop.type=rebot_arm_102_leader \
    --teleop.id=leader1 \
    --teleop.port=/dev/ttyUSB0
```

The leader now outputs unclamped joint positions within its own configured ranges, and the follower applies per-joint direction or scale conversion through `joint_directions` before clipping to follower-side `joint_limits`.

### Teleoperate with follower cameras

```shell
lerobot-teleoperate \
    --robot.type=seeed_b601_dm_follower \
    --robot.id=my_b601_follower \
    --robot.port=/dev/ttyACM4 \
    --robot.can_adapter=damiao \
    --robot.cameras="{ up: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}, side: {type: intelrealsense, serial_number_or_name: 233522074606, width: 640, height: 480, fps: 30}}" \
    --teleop.type=rebot_arm_102_leader \
    --teleop.id=my_b601_leader \
    --teleop.port=/dev/ttyUSB0 \
    --display_data=true
```

Fore more lerobot operations, please refer to the lerobot official documentation:

https://huggingface.co/docs/lerobot/il_robots

## Notes

*   About calibration: the reBot Arm doesn't need an explicit calibration, it only reset every motor's zero position on the launch time of the lerobot scripts.
