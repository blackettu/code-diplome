import cv2
import os
import numpy as np

GRID_COLORS = [
    (255, 255, 0),  # желтый
    (0, 255, 255),  # голубой
    (255, 0, 0),  # красный
    (0, 255, 0),  # зеленый
    (255, 0, 255)  # пурпурный
]

def generate_grid(bbox, rows, cols):
    x_min, y_min, x_max, y_max = bbox
    cell_w = (x_max - x_min) / cols
    cell_h = (y_max - y_min) / rows
    return [
        [
            (x_min + c * cell_w, y_min + r * cell_h,
             x_min + (c + 1) * cell_w, y_min + (r + 1) * cell_h)
            for c in range(cols)
        ]
        for r in range(rows)
    ]


def process_multiple_seedlings(img, containers, seedlings, grid_rows, grid_cols, output_folder, basename):
    img_all = img.copy()
    img_grid = img.copy()
    img_doubles = img.copy()

    # Рисуем все объекты
    for cont in containers:
        x1, y1, x2, y2 = map(int, cont['box'])
        cv2.rectangle(img_all, (x1, y1), (x2, y2), (255, 0, 255), 4)

    for s in seedlings:
        x1, y1, x2, y2 = map(int, s['box'])
        cv2.rectangle(img_all, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Обработка контейнеров
    for idx, cont in enumerate(containers, 1):
        bbox = cont['box']
        color = GRID_COLORS[(idx - 1) % len(GRID_COLORS)]

        # Добавляем подпись контейнера
        label = f"C{idx}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # Рисуем подпись на изображениях
        for img_out in [img_all, img_grid]:
            cv2.rectangle(img_out,
                          (int(bbox[0]), int(bbox[1]) - text_size[1] - 5),
                          (int(bbox[0]) + text_size[0] + 5, int(bbox[1]) - 5),
                          color, -1)
            cv2.putText(img_out, label,
                        (int(bbox[0]) + 3, int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Генерация и отрисовка сетки
        grid = generate_grid(bbox, grid_rows, grid_cols)
        for row in grid:
            for cell in row:
                x1, y1, x2, y2 = map(int, cell)
                cv2.rectangle(img_grid, (x1, y1), (x2, y2), color, 1)

        # Распределение саженцев
        cont_seedlings = [s for s in seedlings if
                          (bbox[0] <= (s['box'][0] + s['box'][2]) / 2 <= bbox[2] and
                           bbox[1] <= (s['box'][1] + s['box'][3]) / 2 <= bbox[3])]

        cell_seedlings = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]
        for s in cont_seedlings:
            cx = (s['box'][0] + s['box'][2]) / 2
            cy = (s['box'][1] + s['box'][3]) / 2
            row_idx = int((cy - bbox[1]) / (bbox[3] - bbox[1]) * grid_rows)
            col_idx = int((cx - bbox[0]) / (bbox[2] - bbox[0]) * grid_cols)
            if 0 <= row_idx < grid_rows and 0 <= col_idx < grid_cols:
                cell_seedlings[row_idx][col_idx].append(s)

        # Обработка множественных саженцев
        for r in range(grid_rows):
            for c in range(grid_cols):
                if len(cell_seedlings[r][c]) >= 2:
                    sorted_seeds = sorted(
                        cell_seedlings[r][c],
                        key=lambda s: (s['box'][2] - s['box'][0]) * (s['box'][3] - s['box'][1]),
                        reverse=True
                    )
                    main_box = sorted_seeds[0]['box']
                    secondary_box = sorted_seeds[1]['box']

                    for img_out in [img_grid, img_doubles]:
                        cv2.rectangle(img_out,
                                      tuple(map(int, main_box[:2])),
                                      tuple(map(int, main_box[2:])),
                                      (0, 255, 0), 2)
                        cv2.rectangle(img_out,
                                      tuple(map(int, secondary_box[:2])),
                                      tuple(map(int, secondary_box[2:])),
                                      (0, 0, 255), 2)

    # Сохраняем результаты
    cv2.imwrite(os.path.join(output_folder, f"{basename}_all.jpg"), img_all)
    cv2.imwrite(os.path.join(output_folder, f"{basename}_grid.jpg"), img_grid)
    cv2.imwrite(os.path.join(output_folder, f"{basename}_doubles.jpg"), img_doubles)
