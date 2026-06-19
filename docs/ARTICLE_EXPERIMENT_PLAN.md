# План экспериментов для подготовки статьи

Этот файл задаёт практическую последовательность действий перед подачей
статьи. Цель — получить воспроизводимые результаты, которые закрывают замечания
рецензента: честное разбиение, независимый `test`, базовый метод сравнения, метрики конечной задачи,
доверительные интервалы и аккуратную таблицу артефактов.

## 0. Подготовка окружения

Выполнить один раз:

```powershell
py -m pip install -r requirements.txt
py -m seedling_experiments --help
```

Ожидаемый результат: CLI показывает команды `prepare`, `train`, `val`,
`predict`, `evaluate-cells`, `baseline-green`, `split`, `audit`,
`check-split`.

Если проект запускает внешний разработчик, сначала сверить контракт в
`docs/DEVELOPER_GUIDE.md`: формат YOLO-разметки, структура `predictions.json`,
смысл `detections`/`containers` и диагностические поля `cell_metrics.json`.

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
- аугментированные копии не лежат в исходном наборе;
- в именах файлов нет суффиксов `_filter`, `_flip`, `_scale`, `_rotate`,
  `_crop`, `_brightness`, `_contrast`, `_color`, `_sharpness`, `_hflip`,
  `_vflip`, если эти файлы не являются настоящими исходными снимками.

До `prepare` сохранить аудит исходного набора:

```powershell
py -m seedling_experiments audit --dataset E:/dataset/raw_seedlings --output E:/dataset/raw_seedlings/raw_dataset_audit.json
```

Этот аудит исходного набора нужен для фиксации отсутствующих и лишних файлов
разметки до разбиения. После `prepare` отсутствующая разметка превращается в
пустые `.txt` в подготовленном наборе, поэтому аудит подготовленного набора уже
не всегда показывает исходную проблему как отсутствие разметки.

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
- `dataset.augmented_name_markers`, если в проекте используются другие имена
  аугментаций;
- `training.data`;
- `validation.data`;
- `prediction.images`;
- `evaluation.dataset`;
- `baseline.images`;
- `baseline.labels`.

Если несколько изображений относятся к одному контейнеру, лотку или дате,
задать `dataset.split.group_regex`, чтобы вся группа попадала только в одну часть разбиения.

Для каждого независимого эксперимента задать уникальные:

- `dataset.prepared_root`;
- `training.name`;
- `validation.output_dir`;
- `prediction.output_dir`;
- `evaluation.output_dir`;
- `baseline.output_dir`.

Это снижает риск смешать артефакты разных запусков. Если используются
относительные пути, команды запускать из корня `code-diplome/`.

Оставить защитные параметры включёнными:

```yaml
dataset:
  allow_augmented_source: false
  allow_split_integrity_issues: false

evaluation:
  strict_predictions: true
  reject_augmented_eval_images: true
  container_prediction_source: detections
```

`container_prediction_source: containers` использовать только как отдельный
диагностический пересчёт, а не как тихую замену финальной end-to-end оценки.

## 3. Сделать честное разбиение и аугментацию только `train`

Запустить:

```powershell
py -m seedling_experiments prepare --config configs/article_yolo11n.yaml
```

Проверить созданные файлы:

```text
E:/dataset/prepared_seedlings/data.yaml
E:/dataset/prepared_seedlings/config.yaml
E:/dataset/prepared_seedlings/run_snapshot.json
E:/dataset/prepared_seedlings/split_manifest.csv
E:/dataset/prepared_seedlings/train_augmentation_manifest.csv
E:/dataset/prepared_seedlings/split_integrity_report.json
E:/dataset/prepared_seedlings/dataset_audit.json
E:/dataset/prepared_seedlings/prepare_summary.json
```

Отдельно можно повторить проверку разбиения:

```powershell
py -m seedling_experiments check-split --dataset E:/dataset/prepared_seedlings --output E:/dataset/prepared_seedlings/split_integrity_report.json
```

Для статьи выписать из prepared `dataset_audit.json`:

- число изображений в `train/val/test`;
- число объектов `container` и `seedlings`;
- средний/минимальный/максимальный размер bbox;

Из raw `raw_dataset_audit.json` выписать:

- число файлов без labels по raw-аудиту;
- число orphan labels по raw-аудиту.

Критерии готовности:

- в `split_integrity_report.json` стоит `ok: true`;
- в `val` и `test` нет аугментированных копий изображений из `train`;
- `dataset_audit.json` не показывает подозрительные аугментированные файлы в
  `val/test`.

Дополнительно проверить, что разбиение не только формально создано, но и пригодно для
оценки: в `val` и `test` есть объекты `container`, `seedlings` и достаточное
число ячеек со сложными случаями. Текущее разбиение выполняется по группам, но не
стратифицирует классы и состояния ячеек.

## 4. Обучить основную модель YOLO11n

Запустить:

```powershell
py -m seedling_experiments train --config configs/article_yolo11n.yaml
```

После обучения проверить:

```text
runs/yolo11n_seedlings/run_snapshot.json
runs/yolo11n_seedlings/config.yaml
runs/yolo11n_seedlings/metrics_summary.json
runs/yolo11n_seedlings/weights/best.pt
```

Для статьи сохранить:

- точную версию Ultralytics;
- случайное зерно;
- epochs;
- batch;
- imgsz;
- оптимизатор/скорость обучения, если задавались;
- устройство обучения;
- путь к весам `best.pt`.

## 5. Провести независимую проверку на `test`

Запустить:

```powershell
py -m seedling_experiments val --config configs/article_yolo11n.yaml --split test
py -m seedling_experiments predict --config configs/article_yolo11n.yaml
py -m seedling_experiments evaluate-cells --config configs/article_yolo11n.yaml
```

Проверить артефакты:

```text
runs/yolo11n_seedlings_eval/config.yaml
runs/yolo11n_seedlings_eval/test_metrics.json
runs/yolo11n_seedlings_test_predictions/config.yaml
runs/yolo11n_seedlings_test_predictions/predictions.json
runs/yolo11n_seedlings_cell_eval/config.yaml
runs/yolo11n_seedlings_cell_eval/cell_metrics.json
runs/yolo11n_seedlings_cell_eval/cell_confusion_matrix.csv
```

В `test_metrics.json` проверить:

```text
validation.split = test
validation.model
validation.data
validation.imgsz
validation.conf / validation.iou
dataset.images
dataset.class_counts
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

При описании `evaluate-cells` указать алгоритм построения матрицы: bbox
контейнера равномерно делится на `grid_rows x grid_cols`, сеянец назначается в
ячейку по центру bbox, а target-removal эвристика оставляет в ячейке `multiple`
самый большой bbox и удаляет остальные.

После `evaluate-cells` отдельно проверить диагностику контейнеров в
`cell_metrics.json`:

```text
container_recall
container_matching.prediction_source
container_matching.best_iou_summary
container_matching.best_iou_counts
prediction_coverage.missing_predictions
suspected_augmented_eval_images
images[].gt_containers
images[].pred_containers
images[].matched_containers
images[].best_container_iou
```

Критерий готовности cell-eval: `prediction_coverage.missing_predictions` пустой,
`suspected_augmented_eval_images` пустой. Если `container_recall` низкий или
равен нулю, в статью нельзя переносить только `cell_accuracy`; нужно отдельно
описать диагностику `container_matching`.

`matched_containers = 0` не всегда означает, что контейнер визуально не найден.
Сопоставление выполняется по IoU между bbox из разметки и bbox из предсказаний,
а порог задаётся `evaluation.container_iou`. В текущем контуре `evaluate-cells`
читает сырые `detections` из `predictions.json`; поле `containers` уже содержит
postprocessing после фильтрации и объединения. Если визуальный отчёт построен по
`containers`, а оценка идёт по `detections`, возможна ситуация, когда на картинке
контейнер выглядит правильным, но в метрике остаётся unmatched.

Для диагностики допускается отдельный пересчёт с:

```yaml
evaluation:
  container_prediction_source: containers
```

Такой пересчёт нужен только для объяснения причин несопоставления. Финальную таблицу
нужно подписывать по тому источнику контейнеров, который реально использован.

Даже при несопоставленном контейнере `evaluate-cells` продолжает считать метрики
уровня ячеек: для сетки используется bbox контейнера из разметки, а в эту сетку
раскладываются предсказанные сеянцы. Поэтому диагностику контейнеров и метрики
ячеек нужно обсуждать отдельно.

Если цель эксперимента — проверить именно матрицу ячеек и поиск сеянцев при
известной геометрии кассеты, допустимо поставить:

```yaml
evaluation:
  use_ground_truth_containers: true
```

Тогда в статье нужно явно написать, что метрики уровня ячеек не являются полной
сквозной оценкой детектора контейнеров.

## 6. Запустить классический базовый метод сравнения

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
method, object metrics source, mAP@50, mAP@50-95, cell accuracy, multi-cell recall, target recall, coord error px
HSV connected components + known containers, N/A, N/A, N/A, ...
YOLO11n, Ultralytics val, ...
```

Для `baseline-green` в текущем контуре нет `mAP`: команда создаёт
`predictions.json`, который оценивается через `evaluate-cells`. Поэтому базовый
метод сравнивается с YOLO по метрикам уровня ячеек и целей, а объектные колонки
mAP оставляются `N/A`, если не реализована отдельная оценка предсказаний
базового метода как детектора.

## 7. Провести абляцию по разрешению и порогам

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
После каждого изменения `prediction.conf`, `prediction.iou` или
`prediction.imgsz` нужно заново выполнить `predict`, а затем `evaluate-cells`;
иначе `cell_metrics.json` будет посчитан по старому `predictions.json`.

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

Для каждого набора повторить шаги 3-7. Для каждого датасета сохранить полный
набор трассировки:

```text
prepared_root/split_integrity_report.json
runs/<dataset>_<model>/config.yaml
runs/<dataset>_<model>/run_snapshot.json
runs/<dataset>_<model>_eval/test_metrics.json
runs/<dataset>_<model>_predictions/predictions.json
runs/<dataset>_<model>_cell_eval/cell_metrics.json
runs/<dataset>_<model>_cell_eval/cell_confusion_matrix.csv
```

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
raw_dataset_audit.json
split_manifest.csv
train_augmentation_manifest.csv
split_integrity_report.json
dataset_audit.json
prepare_summary.json
test_metrics.json
predictions.json
cell_metrics.json
cell_confusion_matrix.csv
weights path
```

## 11. Критерии готовности к переписыванию статьи

Можно переписывать раздел экспериментов, если выполнены все условия:

- разбиение сделано до аугментации;
- `test` содержит только исходные независимые изображения;
- `split_integrity_report.json` имеет `ok: true`;
- есть `raw_dataset_audit.json` и `dataset_audit.json`;
- есть минимум один базовый метод сравнения;
- есть YOLO11n на том же разбиении;
- есть тестовые метрики детектора;
- есть метрики уровня ячеек;
- есть target-removal метрики;
- `prediction_coverage.missing_predictions` пустой;
- `suspected_augmented_eval_images` пустой;
- `container_matching` проверен и интерпретирован;
- есть bootstrap CI;
- для мелких сеянцев выводы сформулированы осторожно;
- все конфиги и артефакты сохранены.

## 12. Что написать в статье после выполнения

В разделе методики:

- описать исходный датасет до аугментации;
- указать, что разбиение выполнялось до аугментации;
- указать, что аугментация применялась только к `train`;
- перечислить гиперпараметры и версии;
- описать базовый метод сравнения;
- описать оценку уровня ячеек.

В разделе результатов:

- отдельно дать результаты крупных и мелких сеянцев;
- сравнить YOLO11n с базовым методом;
- показать матрицу ошибок ячеек;
- дать precision/recall целей удаления;
- дать доверительные интервалы;
- отдельно обсудить ошибки мелких сеянцев и фона.

В заключении:

- не писать, что система готова к автономному лазерному прореживанию;
- писать, что прототип показал работоспособность на ограниченной выборке;
- для мелких сеянцев указать необходимость расширения данных и проверки
  альтернативных методов малых объектов.
