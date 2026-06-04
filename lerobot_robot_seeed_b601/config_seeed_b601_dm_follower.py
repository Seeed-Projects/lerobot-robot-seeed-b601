from dataclasses import dataclass, field

from lerobot.robots.robot import RobotConfig
from .seeed_b601_follower import SeeedB601FollowerConfigBase


@RobotConfig.register_subclass("seeed_b601_dm_follower")
@dataclass
class SeeedB601DMFollowerConfig(RobotConfig, SeeedB601FollowerConfigBase):
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan":  (0x01, 0x11),
            "shoulder_lift": (0x02, 0x12),
            "elbow_flex":    (0x03, 0x13),
            "wrist_flex":    (0x04, 0x14),
            "wrist_yaw":     (0x05, 0x15),
            "wrist_roll":    (0x06, 0x16),
            "gripper":       (0x07, 0x17),
        }
    )

    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan":  (-145.0, 145.0),
            "shoulder_lift": (-170.0, 0.0),
            "elbow_flex":    (-200.0, 0.0),
            "wrist_flex":    (-80.0, 90.0),
            "wrist_yaw":     (-90.0, 90.0),
            "wrist_roll":    (-90.0, 90.0),
            "gripper":       (-270.0, 0.0),
        }
    )

    joint_directions: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": -1.0,
            "shoulder_lift": -1.0,
            "elbow_flex": 1.0,
            "wrist_flex": 1.0,
            "wrist_yaw": 1.0,
            "wrist_roll": -1.0,
            "gripper": -6.0,
        }
    )


    mit_kp: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 50.0,
            "shoulder_lift": 500.0,
            "elbow_flex": 500.0,
            "wrist_flex": 40.0,
            "wrist_yaw": 40.0,
            "wrist_roll": 40.0,
        }
    )

    mit_kd: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 3.0,
            "shoulder_lift": 5.0,
            "elbow_flex": 5.0,
            "wrist_flex": 1.5,
            "wrist_yaw": 1.5,
            "wrist_roll": 1.5,
        }
    )

    # The v_des parameter for the position-velocity control mode of the joints.
    pos_vel_velocity: float | list[float] = 3600

    # Default torque/current ration for gripper's FORCE_POS mode, in range [0,1].
    force_pos_torque_ration: float = 0.1


