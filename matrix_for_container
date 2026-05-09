import cv2
import os
import sys
import glob
import numpy as np
from ultralytics import YOLO
from pillow_heif import open_heif
from PIL import Image
from many_seedlings import process_multiple_seedlings

######################################

MODEL_PATH = r"D:\runs\train10\weights\best.pt"
IMAGES_FOLDER = r"D:\тест"
OUTPUT_FOLDER = r"D:\output_images"
CONF_THRESHOLD = 0
GRID_ROWS = 11
GRID_COLS = 11
MIN_CONTAINER_AREA = 10000
MERGE_DISTANCE_THRESHOLD = 50

# Цвета сеток для разных контейнеров
GRID_COLORS = [
    (255, 255, 0),  # жёлтый
    (0, 255, 255),  # голубой
    (255, 0, 0),    # красный
    (0, 255, 0),    # зелёный
    (255, 0, 255)   # пурпурный
]
# Цвета для боксов: контейнеры и саженцы
colors = {
    0: (255, 0, 255),  # контейнер
    1: (0, 255, 0)     # саженец
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#########################################
def convert_heic_to_jpg(heic_path):
    heif_file = open_heif(heic_path)
    image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
    jpg_path = heic_path.rsplit('.', 1)[0] + '.jpg'
    image.save(jpg_path, "JPEG")
    return jpg_path

# рисует сетку клеток на изображении
def draw_grid(img, cells, color=(255,255,0)):
    for row in cells:
        for (x1, y1, x2, y2) in row:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

# генерация вложенного списка клеток
def generate_grid(bbox, rows=GRID_ROWS, cols=GRID_COLS):
    x_min, y_min, x_max, y_max = bbox
    cw = (x_max - x_min) / cols
    ch = (y_max - y_min) / rows
    grid = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            x1 = x_min + c * cw
            y1 = y_min + r * ch
            x2 = x_min + (c + 1) * cw
            y2 = y_min + (r + 1) * ch
            row_cells.append((x1, y1, x2, y2))
        grid.append(row_cells)
    return grid

# вычисление индекса клетки по координатам центра
def get_cell_index(bbox, center, rows=GRID_ROWS, cols=GRID_COLS):
    x_min, y_min, x_max, y_max = bbox
    cx, cy = center
    if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
        return None
    col = min(int((cx - x_min) / (x_max - x_min) * cols), cols - 1)
    row = min(int((cy - y_min) / (y_max - y_min) * rows), rows - 1)
    return row, col

# создание матрицы заполнения сетки
def assign_seedlings_to_cells(bbox, seedlings):
    matrix = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    for s in seedlings:
        x1, y1, x2, y2 = s['box']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        idx = get_cell_index(bbox, (cx, cy))
        if idx:
            matrix[idx[0]][idx[1]] += 1
    return matrix

# фильтрация и объединение небольших контейнеров
def filter_and_merge_containers(containers):
    filtered, small = [], []
    for c in containers:
        x1, y1, x2, y2 = c['box']
        if (x2 - x1) * (y2 - y1) < MIN_CONTAINER_AREA:
            small.append(c)
        else:
            filtered.append(c)
    for s in small:
        scx, scy = ((s['box'][0] + s['box'][2]) / 2, (s['box'][1] + s['box'][3]) / 2)
        nearest, md = None, float('inf')
        for f in filtered:
            fcx, fcy = ((f['box'][0] + f['box'][2]) / 2, (f['box'][1] + f['box'][3]) / 2)
            dist = np.hypot(scx - fcx, scy - fcy)
            if dist < md and dist < MERGE_DISTANCE_THRESHOLD:
                nearest, md = f, dist
        if nearest:
            merged = s['box'] + nearest['box']
            xs = merged[0::2]
            ys = merged[1::2]
            nearest['box'] = [min(xs), min(ys), max(xs), max(ys)]
        else:
            filtered.append(s)
    return filtered

# обработка одного изображения
def process_image(img_path, model):
    if img_path.lower().endswith(('.heic', '.HEIC')):
        img_path = convert_heic_to_jpg(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Не удалось открыть {img_path}")
        return

    basename = os.path.splitext(os.path.basename(img_path))[0]
    results = model(img)
    containers, seedlings = [], []
    annotated = img.copy()

    # Рисуем все обнаруженные объекты
    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
            if conf < CONF_THRESHOLD:
                continue
            cls = int(box.cls[0])
            entry = {'box': [x1, y1, x2, y2], 'conf': conf}
            (containers if cls == 0 else seedlings).append(entry)
            col = colors.get(cls)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            cv2.putText(annotated, res.names[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    containers = filter_and_merge_containers(containers)

    if not containers:
        if seedlings:  # Сохраняем если есть саженцы
            error_img = annotated.copy()
            h, w = error_img.shape[:2]
            cv2.putText(error_img, "NO CONTAINERS FOUND",
                        (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            output_path = os.path.join(OUTPUT_FOLDER, f"{basename}_ERROR.jpg")
            cv2.imwrite(output_path, error_img)
            print(f"{basename}: контейнеры не найдены, но сохранено изображение с {len(seedlings)} саженцами")
        else:
            print(f"{basename}: нет контейнеров и саженцев")
        return

    matrix_file = os.path.join(OUTPUT_FOLDER, f"{basename}_matrices.txt")
    with open(matrix_file, 'w', encoding='utf-8') as mf:
        for idx, cont in enumerate(containers, start=1):
            bbox = cont['box']
            mat = assign_seedlings_to_cells(bbox, seedlings)
            mf.write(f"# Контейнер {idx} | bbox: {', '.join(map(str, map(int, bbox)))}\n")
            for row in mat:
                mf.write(' '.join(map(str, row)) + '\n')
            mf.write('\n')

    process_multiple_seedlings(
        img,
        containers,
        seedlings,
        GRID_ROWS,
        GRID_COLS,
        OUTPUT_FOLDER,
        basename
    )
    print(f"Готово для {basename}: matrix={matrix_file}")
    
# главный запуск
def main():
    model = YOLO(MODEL_PATH)
    patterns = [os.path.join(IMAGES_FOLDER, ext) for ext in ('*.jpg', '*.jpeg', '*.png', '*.heic', '*.HEIC')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        print("Нет изображений для обработки")
        return
    for fp in files:
        process_image(fp, model)

if __name__ == '__main__':
    main()
