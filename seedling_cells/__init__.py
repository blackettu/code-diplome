"""Grid, cell-state, and target-generation primitives."""

from .cell_state import CellStateBuilder
from .evaluation import (
    cost_sensitive_metrics,
    cost_sensitive_metrics_from_legacy,
    evaluate_action_targets,
    evaluate_cell_states,
    evaluate_scene_state_files,
    evaluate_scene_states,
)
from .grid import (
    Detection,
    GridCell,
    assign_to_cells,
    bbox_iou,
    cell_index,
    cell_index_for_grid_cells,
    center_distance,
    count_matrix,
    generate_grid,
    generate_grid_cells,
    generate_grid_polygons,
)
from .target_generation import LargestBBoxTargetStrategy, generate_largest_bbox_targets

__all__ = [
    "CellStateBuilder",
    "Detection",
    "GridCell",
    "LargestBBoxTargetStrategy",
    "assign_to_cells",
    "bbox_iou",
    "cell_index",
    "cell_index_for_grid_cells",
    "center_distance",
    "cost_sensitive_metrics",
    "cost_sensitive_metrics_from_legacy",
    "count_matrix",
    "evaluate_action_targets",
    "evaluate_cell_states",
    "evaluate_scene_state_files",
    "evaluate_scene_states",
    "generate_grid",
    "generate_grid_cells",
    "generate_grid_polygons",
    "generate_largest_bbox_targets",
]
