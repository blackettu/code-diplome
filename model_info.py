from ultralytics import YOLO
import numpy as np

model_path = r"D:\runs\train\weights\best.pt"
data_yaml  = r"E:\dataset\data.yaml"

model   = YOLO(model_path)
metrics = model.val(data=data_yaml)

# Общая информация
print("\nИнформация о модели:")
model.model.info(verbose=True)

# Считаем среднюю точность и полноту по всем классам
precision_mean = float(np.mean(metrics.box.p))
recall_mean    = float(np.mean(metrics.box.r))

# Печатаем ключевые метрики
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50–95: {metrics.box.map:.4f}")
print(f"Precision: {precision_mean:.4f}")
print(f"Recall: {recall_mean:.4f}")
