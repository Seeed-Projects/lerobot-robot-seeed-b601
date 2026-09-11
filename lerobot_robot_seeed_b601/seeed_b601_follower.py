import logging
import math
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from motorbridge import Controller as MotorBridgeController, Mode as MotorBridgeMode
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position


@dataclass
class SeeedB601FollowerConfigBase:
    """Base configuration for the Seeed B601 Follower arm."""

    # Communication port for CAN adapter (e.g., "can0" for SocketCAN, or "/dev/ttyACM0" for Damiao serial bridge)
    port: str
    
    # CAN adapter type:
    #   "socketcan"  - SocketCAN based adapters (PCAN, slcan, embedded can controller, etc.)
    #   "damiao"     - Damiao dedicated serial bridge
    #   "robstride"  - RobStride dedicated adapter (placeholder, not yet supported)
    can_adapter: str = "socketcan"

    # Baud rate for Damiao serial bridge (only used when can_adapter="damiao")
    dm_serial_baud: int = 921600

    disable_torque_on_disconnect: bool = True

    # Max relative target for joint movements, in degrees
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Motor configuration must be provided by concrete subclasses.
    # Maps motor names to (send_can_id, recv_can_id)
    motor_can_ids: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Control parameters are defined by concrete subclasses so different motor families
    # can keep their own defaults.
    ## Default target velocity for joints running in POS_VEL mode, in degrees/s.
    pos_vel_velocity: float | list[float] = field(default_factory=list)

    ## Default torque/current ration for gripper's FORCE_POS mode, in range [0,1].
    force_pos_torque_ration: float = 0.1

    # Soft joint limits in degrees. Concrete subclasses should define defaults.
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Per-joint action direction/scale applied before joint-limit clipping.
    # Use -1 for sign flip, 1 for no flip, and other values when scaling is required.
    joint_directions: dict[str, float] = field(default_factory=dict)

    # Gravity-compensation feedforward added to MIT tau for the 6 arm joints.
    # Disabled by default; enable in the concrete follower config (e.g. RS).
    gravity_compensation: bool = False
    # URDF for gravity calc. Set by concrete subclasses (e.g. RS follower).
    gravity_urdf_path: str | None = None

    # Temperature protection thresholds (degrees Celsius, read from each motor's t_mos).
    # Concrete subclasses may override per motor family.
    temp_alarm_threshold_c: float = 80.0              # print HIGH TEMP warning above this
    temp_overheat_threshold_c: float = 100.0           # raise KeyboardInterrupt above this
    temp_emergency_disable_threshold_c: float = 135.0  # safe_zero emergency disable-torque limit


logger = logging.getLogger(__name__)


FOLLOWER_GRIPPER_MOTOR = "gripper"
LONG_TIMEOUT_SEC = 0.1
MEDIUM_TIMEOUT_SEC = 0.01

class SeeedB601FollowerBase(Robot):
    """
    Base class for Seeed B601 Follower Arms (DM and RS variants).
    Uses CAN bus communication via motorbridge.
    """

    motor_type: str = ""

    def __init__(self, config: SeeedB601FollowerConfigBase):
        super().__init__(config)
        self.config = config
        self.bus = None
        self.motors = {}
        self.motor_names = list(config.motor_can_ids.keys())
        self._in_safe_zero = False
        self._emergency_disable_requested = False
        # motor_name -> URDF q_idx, built once in connect(); None disables ff.
        self._gravity_pairing = None
        # When True, disconnect() skips safe_zero(). Set by calibrate() so the
        # lerobot-calibrate entrypoint (which calls calibrate() then immediately
        # disconnect()) does not move the arm back to zero. connect() resets this
        # after its internal calibrate() call so normal use still safe-zeros.
        self._skip_safe_zero_on_disconnect = False

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.motor_names:
            features[f"{motor}.pos"] = float
            # features[f"{motor}.vel"] = float
            # features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus is not None and all(cam.is_connected for cam in self.cameras.values())

    def _add_motors_to_bus(self):
        """Must be implemented by subclasses to add specific motor types to self.bus."""
        raise NotImplementedError

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the follower arm and optionally calibrate."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting arm on {self.config.port} (adapter={self.config.can_adapter})...")
        if self.config.can_adapter == "damiao":
            self.bus = MotorBridgeController.from_dm_serial(
                serial_port=self.config.port,
                baud=self.config.dm_serial_baud,
            )
        elif self.config.can_adapter == "robstride":
            raise NotImplementedError(
                "RobStride dedicated USB-to-CAN adapter is not yet supported in motorbridge Python SDK."
            )
        else:
            # Default: socketcan (PCAN, slcan, etc.)
            self.bus = MotorBridgeController(channel=self.config.port)
        
        self._add_motors_to_bus()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()
            # calibrate() set _skip_safe_zero_on_disconnect for the calibrate
            # script's benefit; clear it here so a later disconnect (after
            # normal use of this connected arm) still runs safe_zero().
            self._skip_safe_zero_on_disconnect = False

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        # Skip during the calibrate flow (calibrate=False); the gripper
        # may not be homed yet and the probe would block connect().
        if calibrate:
            self.detect_gripper_zero()

        self._build_gravity_pairing()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return bool(self.calibration)

    def calibrate(self) -> None:
        """Calibration procedure for B601."""
        # Mark so that the disconnect() following this (as in the
        # lerobot-calibrate entrypoint) skips safe_zero(). connect() clears
        # this flag after its internal calibrate() call.
        self._skip_safe_zero_on_disconnect = True
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                return

        logger.info(f"\nRunning calibration for {self}")
        
        self.bus.disable_all()

        print(
            "\nCalibration: Set Zero Position\n"
            "Please MANUALLY move the robot to its ZERO POSITION, and close its gripper.\n"
            "Reference the B601 manual for Zero Pose (generally the default sit-down position).\n"
        )
        input("Press ENTER when ready...")

        for motor in self.motors.values():
            motor.set_zero_position()
            time.sleep(LONG_TIMEOUT_SEC)
        
        logger.info("Arm zero position set.")

        logger.info("Setting range: -90° to +90° by default for all joints")
        self.calibration = {}
        for motor_name, (send_id, recv_id) in self.config.motor_can_ids.items():
            self.calibration[motor_name] = MotorCalibration(
                id=send_id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        # Keep torque off while switching modes, then enable after all motors are configured.
        self.bus.disable_all()
        num_retry = 9
        for motor_name, motor in self.motors.items():
            target_mode = MotorBridgeMode.MIT if self.motor_type == "rs" else (
                MotorBridgeMode.FORCE_POS
                if motor_name == FOLLOWER_GRIPPER_MOTOR
                else MotorBridgeMode.POS_VEL
            )
            for _ in range(num_retry + 1):
                try:
                    motor.ensure_mode(target_mode)
                    break
                except Exception as e:
                    if _ == num_retry:
                        raise e
                    time.sleep(MEDIUM_TIMEOUT_SEC)
            logger.info(f"{motor_name} ensure mode {target_mode}")
        self.bus.enable_all()

    def disable_torque(self) -> None:
        """Disable follower motor torque so the arm can be moved by hand during read-only debugging."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disable_all()
        logger.info(f"{self} torque disabled.")

    def detect_gripper_zero(
        self,
        tolerance_deg: float = 5.0,
        force_ratio: float = 0.05,
        mit_tau: float = 0.3,
    ) -> dict:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        motor = self.motors.get(FOLLOWER_GRIPPER_MOTOR)
        if motor is None:
            return {"error": "gripper motor not found"}
        lo, hi = self.config.joint_limits[FOLLOWER_GRIPPER_MOTOR]
        closing_sign = -math.copysign(1.0, hi if abs(hi) > abs(lo) else lo)

        prev, stable = None, 0
        for _ in range(200):
            if self.motor_type == "rs":
                motor.send_mit(0.0, 0.0, 0.0, 1.5, closing_sign * mit_tau)
            else:
                motor.send_force_pos(
                    math.radians(closing_sign * 10.0),
                    math.radians(10.0),
                    force_ratio,
                )
            time.sleep(0.05)
            motor.request_feedback()
            try:
                self.bus.poll_feedback_once()
            except Exception:
                pass
            state = motor.get_state()
            if state is None:
                continue
            cur = math.degrees(state.pos)
            stable = stable + 1 if prev is not None and abs(cur - prev) < 0.1 else 0
            prev = cur
            if stable >= 5:
                break

        if self.motor_type == "rs":
            motor.send_mit(0.0, 0.0, 0.0, 1.5, 0.0)
        else:
            motor.send_force_pos(0.0, math.radians(10.0), force_ratio)
        if prev is not None and abs(prev) <= tolerance_deg:
            logger.info("gripper zero-detect: zero-point check passed")
        else:
            offset = prev if prev is not None else float("nan")
            raise RuntimeError(
                "gripper zero-detect: gripper not homed, offset "
                f"{offset:.2f} deg, please confirm gripper is at home or recalibrate"
            )

    def _read_motor_temperatures(self) -> dict[str, float]:
        """Read per-motor MOS temperatures once and return available values."""
        for motor in self.motors.values():
            motor.request_feedback()
        try:
            self.bus.poll_feedback_once()
        except Exception:
            logger.warning("Temperature check poll feedback failed.")

        temps: dict[str, float] = {}
        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            if state is not None:
                temps[motor_name] = state.t_mos

        return temps

    def _check_motor_temperatures(
        self,
        alarm_threshold_c: float,
        overheat_threshold_c: float,
        context: str = "",
    ) -> dict[str, float]:
        """Read motor MOS temperatures once, print a HIGH TEMP warning for any
        motor above ``alarm_threshold_c``, and raise ``KeyboardInterrupt`` if any
        motor exceeds ``overheat_threshold_c`` (aborts the control loop).

        Returns the dict of ``{motor_name: t_mos_c}`` that was read.
        """
        temperatures = self._read_motor_temperatures()
        logger.debug(
            "motor temps (°C): %s",
            " | ".join(
                f"{n}={temperatures[n]:.1f}" if n in temperatures else f"{n}=--"
                for n in self.motors
            ),
        )
        label = f" in {context}" if context else ""
        for motor_name, temp_c in temperatures.items():
            if temp_c > alarm_threshold_c:
                print(
                    f"[HIGH TEMP] {motor_name} t_mos={temp_c:.2f}C > {alarm_threshold_c:.2f}C"
                )
            if temp_c > overheat_threshold_c:
                logger.error(
                    "Overheat detected%s: %s t_mos=%.2fC > %.2fC.",
                    label,
                    motor_name,
                    temp_c,
                    overheat_threshold_c,
                )
                raise KeyboardInterrupt("Overheat detected")
        return temperatures

    def mit_output_torque_limit(
        self,
        motor: Any,
        pos_target_rad: float,
    ) -> float | None:
        """Compute MIT torque command from target position and motor state."""
        return 0.0

    def _build_gravity_pairing(self) -> None:
        """Build motor_name -> URDF q_idx pairing once at connect().

        Loads and warms the gravity model, then pairs follower arm motors
        (config order, excluding gripper) with URDF revolute joints (DFS
        order). On any failure sets self._gravity_pairing = None so
        _gravity_feedforward no-ops.
        """
        if not (self.config.gravity_compensation and self.motor_type == "rs"):
            self._gravity_pairing = None
            return
        urdf = self.config.gravity_urdf_path
        if not urdf:
            logger.warning("gravity_urdf_path not set, feedforward disabled")
            self._gravity_pairing = None
            return
        try:
            from .gravity import load_dynamics_model
        except Exception as exc:  # pragma: no cover
            logger.warning("gravity import failed, feedforward disabled: %s", exc)
            self._gravity_pairing = None
            return
        model = load_dynamics_model(urdf_path=urdf)
        arm_motors = [n for n in self.motor_names if n != FOLLOWER_GRIPPER_MOTOR]
        arm_qidx = [j.q_idx for j in model.joints if j.jtype in ("revolute", "continuous")]
        if len(arm_motors) != len(arm_qidx):
            logger.warning(
                "gravity feedforward disabled: arm motor count %d != URDF revolute %d",
                len(arm_motors), len(arm_qidx),
            )
            self._gravity_pairing = None
            return
        self._gravity_pairing = dict(zip(arm_motors, arm_qidx))

    def _gravity_feedforward(self) -> dict[str, float]:
        """Per-arm-motor gravity feedforward torque (N·m); {} if disabled."""
        pairing = self._gravity_pairing
        if not pairing:
            return {}
        import numpy as np
        from .gravity import load_dynamics_model, compute_generalized_gravity
        model = load_dynamics_model(urdf_path=self.config.gravity_urdf_path)
        q = np.zeros(model.nq)
        for motor_name, qi in pairing.items():
            motor = self.motors.get(motor_name)
            if motor is None:
                continue
            state = motor.get_state()
            if state is not None:
                q[qi] = float(state.pos)
        tau = compute_generalized_gravity(model=model, q=q)
        return {name: float(tau[qi]) for name, qi in pairing.items()}

    def safe_zero(self, step_interval_s: float = 0.02, exit_on_complete: bool = True) -> None:
        """Move arm joints back to zero in a safer two-stage interpolation.

        Stage 1: CAN ID 1/4/5/6 -> 0
        Stage 2: CAN ID 2/3 -> 0
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._in_safe_zero:
            logger.warning("safe_zero skipped: already running.")
            return

        if step_interval_s < 0.0:
            raise ValueError("step_interval_s must be >= 0")

        self._in_safe_zero = True
        try:
            id_to_joint: dict[int, str] = {
                send_id: motor_name
                for motor_name, (send_id, _) in self.config.motor_can_ids.items()
                if motor_name != FOLLOWER_GRIPPER_MOTOR
            }

            stage_1 = [id_to_joint[i] for i in (1, 4, 5, 6) if i in id_to_joint]
            stage_2 = [id_to_joint[i] for i in (2, 3) if i in id_to_joint]
            controlled_joints = stage_1 + [name for name in stage_2 if name not in stage_1]

            if not controlled_joints:
                logger.warning("safe_zero skipped: no arm joints mapped to CAN IDs 1-6.")
                return

            def _read_motor_deg(joint_name: str) -> float:
                """Read a joint's current angle in real motor degrees (no direction scaling)."""
                motor = self.motors.get(joint_name)
                if motor is None:
                    raise RuntimeError(f"safe_zero failed: motor '{joint_name}' not found")

                max_retry = 10
                for attempt in range(1, max_retry + 1):
                    try:
                        motor.request_feedback()
                        self.bus.poll_feedback_once()
                    except Exception:
                        logger.debug(
                            "safe_zero feedback poll failed for %s (attempt %d/%d)",
                            joint_name,
                            attempt,
                            max_retry,
                        )

                    state = motor.get_state()
                    if state is not None:
                        return math.degrees(state.pos)

                    if attempt < max_retry:
                        time.sleep(MEDIUM_TIMEOUT_SEC)

                raise RuntimeError(
                    f"safe_zero failed: unable to read state for '{joint_name}' after {max_retry} attempts"
                )

            def _frame_count(
                starts: dict[str, float],
                targets: dict[str, float] | None = None,
            ) -> int:
                targets = targets or {}
                max_delta_deg = max(
                    (abs(targets.get(j, 0.0) - v) for j, v in starts.items()),
                    default=0.0,
                )
                return max(1, math.ceil(max_delta_deg * 2.0))

            def _interp_to_zero(
                active_starts: dict[str, float],
                hold_joints: dict[str, float],
                targets: dict[str, float] | None = None,
            ) -> bool:
                if not active_starts:
                    return False
                targets = targets or {}
                frames = _frame_count(active_starts, targets)
                emergency_disable_threshold_c = self.config.temp_emergency_disable_threshold_c
                for frame in range(1, frames + 1):
                    temperatures = self._read_motor_temperatures()
                    for motor_name, temp_c in temperatures.items():
                        if temp_c > emergency_disable_threshold_c:
                            logger.error(
                                "Auto-disable on overtemperature during safe_zero: %s t_mos=%.2fC > %.2fC.",
                                motor_name,
                                temp_c,
                                emergency_disable_threshold_c,
                            )
                            self._emergency_disable_requested = True
                            self.disable_torque()
                            logger.error("safe_zero aborted: emergency overtemperature.")
                            return True

                    ratio = frame / frames
                    action: RobotAction = {}
                    # Convert motor degrees to action-space here to cancel
                    # send_action's *joint_directions transform.
                    for joint, start in hold_joints.items():
                        d = self.config.joint_directions.get(joint, 1.0) or 1.0
                        action[f"{joint}.pos"] = start / d
                    for joint, start in active_starts.items():
                        target = targets.get(joint, 0.0)
                        d = self.config.joint_directions.get(joint, 1.0) or 1.0
                        action[f"{joint}.pos"] = (start + (target - start) * ratio) / d
                    self.send_action(action)
                    if step_interval_s > 0.0:
                        time.sleep(step_interval_s)

                return False

            stage_1_start = {joint: _read_motor_deg(joint) for joint in stage_1}
            stage_2_start = {joint: _read_motor_deg(joint) for joint in stage_2}

            logger.info("safe_zero stage1 start: joints=%s", stage_1)
            if _interp_to_zero(stage_1_start, stage_2_start):
                return

            # Stage 2: move CAN ID 2/3 back to zero, and bring the gripper back to
            # 170° if it is currently past 180° (avoids leaving it wide open).
            stage_2_active = dict(stage_2_start)
            stage_2_targets: dict[str, float] = {}
            if FOLLOWER_GRIPPER_MOTOR in self.motors:
                try:
                    gripper_motor_deg = _read_motor_deg(FOLLOWER_GRIPPER_MOTOR)
                except RuntimeError as e:
                    logger.warning("safe_zero: could not read gripper position: %s", e)
                    gripper_motor_deg = None

                if gripper_motor_deg is not None and abs(gripper_motor_deg) > 180.0:
                    # Wide open: bring it back toward 170°, preserving sign.
                    target_motor_deg = math.copysign(170.0, gripper_motor_deg)
                    stage_2_active[FOLLOWER_GRIPPER_MOTOR] = gripper_motor_deg
                    stage_2_targets[FOLLOWER_GRIPPER_MOTOR] = target_motor_deg
                    logger.info(
                        "safe_zero gripper: %.2f° (motor, abs>180°), returning to %.2f°",
                        gripper_motor_deg,
                        target_motor_deg,
                    )

            logger.info("safe_zero stage2 start: joints=%s", stage_2)
            if _interp_to_zero(
                stage_2_active,
                {joint: 0.0 for joint in stage_1},
                stage_2_targets,
            ):
                return
            logger.info("safe_zero done.")
            time.sleep(2.0)
            if exit_on_complete:
                # Raise KeyboardInterrupt so upper-level control loops handle this
                # the same way as Ctrl+C.
                raise KeyboardInterrupt("safe_zero completed")
        finally:
            self._in_safe_zero = False

    def get_observation(self) -> RobotObservation:
        """Get current observation from robot."""
        start = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}

        # Request and poll feedback from motorbridge
        for motor in self.motors.values():
            motor.request_feedback()
        try:
            self.bus.poll_feedback_once()
        except:
            logger.warning(f"can bus poll feedback failed.")

        for motor_name, motor in self.motors.items():
            state = motor.get_state()
            if state is not None:
                # motorbridge works natively in radians. Convert to degrees.
                obs_dict[f"{motor_name}.pos"] = math.degrees(state.pos)
                obs_dict[f"{motor_name}.vel"] = math.degrees(state.vel)
                obs_dict[f"{motor_name}.torque"] = state.torq
            else:
                obs_dict[f"{motor_name}.pos"] = 0.0
                obs_dict[f"{motor_name}.vel"] = 0.0
                obs_dict[f"{motor_name}.torque"] = 0.0

        # Capture images
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")
        # logger.debug(f"Observation: {obs_dict}")

        return obs_dict

    def send_action(
        self,
        action: RobotAction
    ) -> RobotAction:
        """Send action command to robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._in_safe_zero:
            self._check_motor_temperatures(
                self.config.temp_alarm_threshold_c,
                self.config.temp_overheat_threshold_c,
                context="send_action",
            )

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Apply per-joint direction/scale mapping before clipping.
        for motor_name, position in goal_pos.items():
            direction = self.config.joint_directions.get(motor_name, 0.0)
            position = position * direction
            # print(f"motor_name: {motor_name}, position: {position}")
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f} to {clipped_position:.2f}")
                position = clipped_position

            goal_pos[motor_name] = position

        # To tolerate 6-DOF leader arms that don't have a wrist_yaw joint, we can allow the follower to ignore missing wrist_yaw commands by treating them as 0.
        if 'wrist_yaw' not in goal_pos:
            goal_pos['wrist_yaw'] = 0.0

        # Safety: Cap relative target
        if self.config.max_relative_target is not None:
            # We need current position in degrees to compare against relative limit safely
            present_pos = {}
            for motor_name, motor in self.motors.items():
                state = motor.get_state()
                if state is not None:
                    present_pos[motor_name] = math.degrees(state.pos)
                else:
                    present_pos[motor_name] = 0.0
            
            goal_present_pos = {key: (g_pos, present_pos.get(key, g_pos)) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Prepare and send commands
        gravity_tau = (
            self._gravity_feedforward()
            if (self.config.gravity_compensation and self.motor_type == "rs")
            else {}
        )

        for motor_name, position_degrees in goal_pos.items():
            try:
                idx = self.motor_names.index(motor_name)
            except ValueError:
                idx = 0 # Fallback

            # Convert target position from degrees to radians for motorbridge
            pos_rad = math.radians(position_degrees)
            vel_deg_s = (
                self.config.pos_vel_velocity[idx]
                if isinstance(self.config.pos_vel_velocity, list)
                else self.config.pos_vel_velocity
            )
            vel_rad = math.radians(vel_deg_s)

            motor = self.motors.get(motor_name)
            if motor is not None:
                if motor_name == FOLLOWER_GRIPPER_MOTOR:
                    if self.motor_type == "rs":
                        tau_ff = self.mit_output_torque_limit(motor, pos_rad)
                        if tau_ff is None:
                            tau_ff = 0.0
                        motor.send_mit(0, 0, 0, 1.5, tau_ff)
                        logger.debug(
                            f"Sent MIT command to {motor_name}: pos={position_degrees:.2f}°, "
                            f"tau_ff={tau_ff:.2f}"
                        )
                    else:
                        motor.send_force_pos(pos_rad, vel_rad, self.config.force_pos_torque_ration)
                        logger.debug(f"Sent FORCE_POS command to {motor_name}: pos={position_degrees:.2f}°, vel={vel_deg_s:.2f}°/s, ratio={0.1}")
                else:
                    if self.motor_type == "rs":
                        if self.config.gravity_compensation:
                            kp = getattr(self.config, "gravity_mit_kp", {}).get(motor_name, 0.0)
                            kd = getattr(self.config, "gravity_mit_kd", {}).get(motor_name, 0.0)
                        else:
                            kp = getattr(self.config, "mit_kp", {}).get(motor_name, 0.0)
                            kd = getattr(self.config, "mit_kd", {}).get(motor_name, 0.0)
                        tau = gravity_tau.get(motor_name, 0.0)
                        motor.send_mit(pos_rad, 0, kp, kd, tau)
                        logger.debug(
                            f"Sent MIT command to {motor_name}: "
                            f"pos={position_degrees:.2f}°, kp={kp}, kd={kd}, tau_ff={tau:.3f}"
                        )
                    else:
                        motor.send_pos_vel(pos_rad, 32)
                        logger.debug(f"Sent POS_VEL command to {motor_name}: target={pos_rad:.2f},pos={position_degrees:.2f}°, vel={vel_deg_s:.2f}°/s")

        # motorbridge sends packets mostly synchronously here over loop, 
        # so we don't need a bulk send command through ctypes.

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        """Disconnect from robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if (
            not self._in_safe_zero
            and not self._emergency_disable_requested
            and not self._skip_safe_zero_on_disconnect
        ):
            try:
                self.safe_zero(exit_on_complete=False)
            except Exception:
                logger.exception("safe_zero during disconnect failed.")
        elif self._skip_safe_zero_on_disconnect:
            logger.info(
                "safe_zero skipped on disconnect: calibrate context "
                "(_skip_safe_zero_on_disconnect=True)."
            )

        for motor in self.motors.values():
            if self.config.disable_torque_on_disconnect:
                motor.disable()
            motor.clear_error()
            motor.close()
        
        self.bus.close()
        self.bus = None

        for cam in self.cameras.values():
            cam.disconnect()

        self._emergency_disable_requested = False
        logger.info(f"{self} disconnected.")
