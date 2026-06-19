"""Robot abstraction layer with safe simulator adapter."""

from .adapters.base import RobotAdapter
from .adapters.dry_run_serial import DryRunSerialAdapter
from .adapters.real_gantry_serial import RealGantrySerialAdapter
from .adapters.simulator import SimulatorRobotAdapter
from .dry_run_runner import run_dry_run_plan
from .dry_run_test import DryRunControlPoint, load_control_points, run_dry_run_control_points
from .hil_runner import load_action_plan, load_scene_state, run_hil_pointer_plan
from .hil_safety import HardwareInLoopReview, validate_hil_review
from .motion_replay import load_motion_commands, replay_motion_commands, run_motion_replay
from .protocol import MotionCommand, MotionResult, RobotExecutionResult, SpeedProfile, ToolProfile, ToolResult

__all__ = [
    "DryRunSerialAdapter",
    "DryRunControlPoint",
    "HardwareInLoopReview",
    "RealGantrySerialAdapter",
    "load_control_points",
    "load_action_plan",
    "load_scene_state",
    "load_motion_commands",
    "MotionCommand",
    "MotionResult",
    "RobotAdapter",
    "RobotExecutionResult",
    "SimulatorRobotAdapter",
    "SpeedProfile",
    "ToolProfile",
    "ToolResult",
    "replay_motion_commands",
    "run_dry_run_control_points",
    "run_dry_run_plan",
    "run_hil_pointer_plan",
    "run_motion_replay",
    "validate_hil_review",
]
