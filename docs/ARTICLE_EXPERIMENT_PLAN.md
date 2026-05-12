# План экспериментов для подготовки статьи

Этот файл задаёт практическую последовательность действий перед подачей
статьи. Цель — получить воспроизводимые результаты, которые закрывают замечания
рецензента: честный split, независимый test, baseline, метрики конечной задачи,
доверительные интервалы и аккуратную таблицу артефактов.

## 0. Подготовка окружения

Выполнить один раз:

```powershell
py -m pip install -r requirements.txt
py -m seedling_experiments --help
```

Ожидаемый результат: CLI показывает команды `prepare`, `train`, `val`,
`predict`, `evaluate-cells`, `baseline-green`.

## 1. Подготовить исходный датасет

Собрать исходные изображения без аугментаций в структуру YOLO:

```text
E:/dataset/raw_seedlings/
  images/
  labels/
```

Проверить, что:

- в `images/` лежат только исходные фотографии;
- в `labels/` лежит YOLO-разметка с теми же именами файлов;
- классы согласованы: `0 = container`, `1 = seedlings`;
- аугментированные копии не лежат в исходном наборе.

Для статьи отдельно подготовить таблицу исходных сцен:

```text
image, date, seedling_size, species, tray_id, container_id, camera_height_m, lighting, notes
```

Эта таблица нужна, чтобы обосновать дату съёмки, размер сеянцев, число
контейнеров, число ячеек и условия освещения.

## 2. Настроить конфиг эксперимента

Скопировать пример:

```powershell
Copy-Item configs/example_experiment.yaml configs/article_yolo11n.yaml
```

В `configs/article_yolo11n.yaml` обязательно изменить:

- `dataset.raw_root`;
- `dataset.prepared_root`;
- `training.data`;
- `validation.data`;
- `prediction.images`;
- `evaluation.dataset`;
- `baseline.images`;
- `baseline.labels`.

Если несколько изображений относятся к одному контейнеру, лотку или дате,
задать `dataset.split.group_regex`, чтобы вся группа попадала только в один split.

## 3. Сделать честный split и аугментацию только train

Запустить:

```powershell
py -m seedling_experiments prepare --config configs/article_yolo11n.yaml
```

Проверить созданные файлы:

```text
E:/dataset/prepared_seedlings/data.yaml
E:/dataset/prepared_seedlings/split_manifest.csv
E:/dataset/prepared_seedlings/train_augmentation_manifest.csv
E:/dataset/prepared_seedlings/dataset_audit.json
E:/dataset/prepared_seedlings/prepare_summary.json
```

Для статьи выписать из `dataset_audit.json`:

- число изображений в `train/val/test`;
- число объектов `container` и `seedlings`;
- средний/минимальный/максимальный размер bbox;
- число файлов без labels;
- число orphan labels.

Критерий готовности: в `val` и `test` нет аугментированных копий изображений из
`train`.

## 4. Обучить основную модель YOLO11n

Запустить:

```powershell
py -m seedling_experiments train --config configs/article_yolo11n.yaml
```

После обучения проверить:

```text
runs/yolo11n_seedlings/run_snapshot.json
runs/yolo11n_seedlings/metrics_summary.json
runs/yolo11n_seedlings/weights/best.pt
```

Для статьи сохранить:

- точную версию Ultralytics;
- seed;
- epochs;
- batch;
- imgsz;
- optimizer/lr, если задавались;
- устройство обучения;
- путь к весам `best.pt`.

## 5. Провести независимую проверку на test

Запустить:

```powershell
py -m seedling_experiments val --config configs/article_yolo11n.yaml --split test
py -m seedling_experiments predict --config configs/article_yolo11n.yaml
py -m seedling_experiments evaluate-cells --config configs/article_yolo11n.yaml
```

Проверить артефакты:

```text
runs/yolo11n_seedlings_eval/test_metrics.json
runs/yolo11n_seedlings_test_predictions/predictions.json
runs/yolo11n_seedlings_cell_eval/cell_metrics.json
runs/yolo11n_seedlings_cell_eval/cell_confusion_matrix.csv
```

В статью занести две группы метрик.

Метрики детектора:

```text
mAP@50
mAP@50-95
precision
recall
```

Метрики конечной задачи:

```text
cell accuracy
macro precision / recall / F1 для классов 0 / 1 / >1
precision / recall множественных ячеек
precision / recall целей удаления
mean coordinate error, px
bootstrap 95% CI для cell accuracy
```

## 6. Запустить классический baseline

Запустить:

```powershell
py -m seedling_experiments baseline-green --config configs/article_yolo11n.yaml
```

В конфиге временно заменить:

```yaml
evaluation:
  predictions: runs/baseline_green_test_predictions/predictions.json
  output_dir: runs/baseline_green_cell_eval
  use_ground_truth_containers: true
```

Затем выполнить:

```powershell
py -m seedling_experiments evaluate-cells --config configs/article_yolo11n.yaml
```

В статью добавить таблицу сравнения:

```text
method, mAP@50, mAP@50-95, cell accuracy, multi-cell recall, target recall, coord error px
HSV connected components + known containers
YOLO11n
```

## 7. Провести ablation по разрешению и порогам

Создать копии конфига:

```text
configs/article_yolo11n_img640.yaml
configs/article_yolo11n_img960.yaml
configs/article_yolo11n_img1280.yaml
```

В каждой копии изменить `training.imgsz`, `validation.imgsz` и `prediction.imgsz`.
Для каждого конфига повторить:

```powershell
py -m seedling_experiments train --config <config>
py -m seedling_experiments val --config <config> --split test
py -m seedling_experiments predict --config <config>
py -m seedling_experiments evaluate-cells --config <config>
```

После выбора лучшего разрешения подобрать `prediction.conf` и `prediction.iou`
только на `val`. Зафиксировать выбранные значения и один раз пересчитать `test`.

В статью добавить таблицу:

```text
imgsz, conf, iou, mAP@50, mAP@50-95, cell accuracy, multi-cell recall, target recall
```

## 8. Отдельно оценить крупные и мелкие сеянцы

Если крупные и мелкие сеянцы находятся в разных наборах, сделать два независимых
конфига:

```text
configs/article_large_seedlings.yaml
configs/article_small_seedlings.yaml
```

Для каждого набора повторить шаги 3-7.

В статье не смешивать сезон и размер растения. Лучше использовать формулировки:

```text
выборка крупных сеянцев, съёмка 7 ноября 2024
выборка мелких сеянцев, съёмка 22 апреля 2025
```

## 9. Проверить эвристику выбора сеянца для сохранения

Текущая эвристика сохраняет сеянец с максимальной площадью bbox. Для статьи
нужно проверить её хотя бы на подвыборке.

Подготовить экспертную таблицу:

```text
image, container_id, row, col, keep_seedling_id, remove_seedling_ids, comment
```

Сравнить экспертное решение с автоматическим `removal_targets` из
`cell_metrics.json` / `predictions.json`.

В статье явно написать ограничение: площадь bbox не всегда равна биологической
развитости, особенно при заваливании сеянца.

## 10. Сформировать итоговую таблицу для статьи

Минимальная итоговая таблица:

```text
dataset
original images
train / val / test images
container boxes
seedling boxes
multiple cells
model
imgsz
mAP@50
mAP@50-95
precision
recall
cell accuracy
macro-F1 cells
multi-cell precision
multi-cell recall
target precision
target recall
coordinate error px
95% CI cell accuracy
```

Отдельно сохранить список артефактов:

```text
config path
run_snapshot.json
dataset_audit.json
test_metrics.json
predictions.json
cell_metrics.json
cell_confusion_matrix.csv
weights path
```

## 11. Критерии готовности к переписыванию статьи

Можно переписывать раздел экспериментов, если выполнены все условия:

- split сделан до аугментации;
- test содержит только исходные независимые изображения;
- есть `dataset_audit.json`;
- есть минимум один baseline;
- есть YOLO11n на том же split;
- есть test-метрики детектора;
- есть cell-level метрики;
- есть target-removal метрики;
- есть bootstrap CI;
- для мелких сеянцев выводы сформулированы осторожно;
- все конфиги и артефакты сохранены.

## 12. Что написать в статье после выполнения

В разделе методики:

- описать исходный датасет до аугментации;
- указать, что split выполнялся до аугментации;
- указать, что аугментация применялась только к train;
- перечислить гиперпараметры и версии;
- описать baseline;
- описать cell-level evaluation.

В разделе результатов:

- отдельно дать результаты крупных и мелких сеянцев;
- сравнить YOLO11n с baseline;
- показать confusion matrix ячеек;
- дать target-removal precision/recall;
- дать доверительные интервалы;
- отдельно обсудить ошибки мелких сеянцев и фона.

В заключении:

- не писать, что система готова к автономному лазерному прореживанию;
- писать, что прототип показал работоспособность на ограниченной выборке;
- для мелких сеянцев указать необходимость расширения данных и проверки
  альтернативных методов малых объектов.
