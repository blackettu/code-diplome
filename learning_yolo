from ultralytics import YOLO

model = YOLO(r"C:\Users\user\Desktop\diplom_code\best.pt") 
# model = YOLO("yolo11n.pt")
model.train(
    data=r'E:\dataset\data.yaml',  # Путь к файлу конфигурации данных
    epochs=200,        # Количество эпох
    batch=16,         # Размер батча
    imgsz=640,        # Размер изображений
    save=True,        # Сохранение результатов
    project=r"E:\runs"  # Директория для сохранения
)

# Оценка модели
metrics = model.val()

# Запись метрик в файл
with open(r'C:\Users\user\Desktop\metrics.txt', 'w') as f:
    f.write(str(metrics))

print("Метрики сохранены!")
