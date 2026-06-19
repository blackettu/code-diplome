from __future__ import annotations

import unittest

from seedling_cells.grid import (
    Detection,
    assign_to_cells,
    cell_index,
    cell_index_for_grid_cells,
    choose_removal_targets,
    count_matrix,
    generate_grid,
    generate_grid_polygons,
    point_in_polygon,
)


class GridTests(unittest.TestCase):
    def test_cell_index_uses_last_cell_for_max_boundary(self) -> None:
        self.assertEqual(cell_index([0, 0, 110, 110], (110, 110), 11, 11), (10, 10))
        self.assertEqual(cell_index([0, 0, 110, 110], (0, 0), 11, 11), (0, 0))
        self.assertIsNone(cell_index([0, 0, 110, 110], (111, 50), 11, 11))

    def test_assign_to_cells_and_count_matrix(self) -> None:
        detections = [
            Detection([0, 0, 10, 10]),
            Detection([12, 0, 20, 10]),
            Detection([90, 90, 100, 100]),
        ]

        cells = assign_to_cells([0, 0, 100, 100], detections, 10, 10)
        matrix = count_matrix(cells)

        self.assertEqual(matrix[0][0], 1)
        self.assertEqual(matrix[0][1], 1)
        self.assertEqual(matrix[9][9], 1)

    def test_choose_removal_targets_keeps_largest_bbox(self) -> None:
        cells = [[[] for _ in range(1)] for _ in range(1)]
        cells[0][0] = [
            Detection([0, 0, 20, 20]),
            Detection([0, 0, 10, 10]),
            Detection([0, 0, 5, 5]),
        ]

        targets = choose_removal_targets(cells)

        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0]["keep_box"], [0, 0, 20, 20])

    def test_generate_grid_polygons_supports_tray_corners(self) -> None:
        cells = generate_grid_polygons([[0, 0], [100, 10], [90, 110], [-10, 100]], 2, 2)

        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0].row, 0)
        self.assertEqual(cells[0].col, 0)
        self.assertEqual(len(cells[0].polygon_px), 4)

    def test_corner_grid_assignment_uses_cell_polygons_not_outer_bbox(self) -> None:
        cells = generate_grid_polygons([[20, 0], [100, 0], [80, 100], [0, 100]], 1, 1)

        self.assertEqual(cell_index_for_grid_cells(cells, (50, 50)), (0, 0))
        self.assertIsNone(cell_index_for_grid_cells(cells, (5, 10)))
        self.assertTrue(point_in_polygon((20, 0), cells[0].polygon_px))

    def test_generate_grid_rejects_invalid_bbox(self) -> None:
        with self.assertRaises(ValueError):
            generate_grid([0, 0, 0, 10], 11, 11)


if __name__ == "__main__":
    unittest.main()
