# Экспериментальный протокол

Этот протокол закрывает основные замечания рецензента по воспроизводимости.

Пошаговый план действий именно для подготовки статьи вынесен в
`docs/ARTICLE_EXPERIMENT_PLAN.md`.
Практический контракт для разработчика, который впервые запускает проект,
вынесен в `docs/DEVELOPER_GUIDE.md`.

## 1. Разбиение до аугментации

Команда `prepare` сначала делит исходные изображения на `train/val/test`, а затем
применяет аугментацию только к `train`. Это предотвращает попадание близких копий
одного изображения в `test`.

Если несколько файлов относятся к одному контейнеру, дате или лотку, задайте
`dataset.split.group_regex` в конфиге. Все изображения с одним group id попадут
в одну часть разбиения.
Для `seedling_data image-manifest` тот же regex может использовать named capture
groups `group_id`, `session_id` и `tray_id`; если `tray_id` не задан, он
заполняется из `group_id`, а `session_id` берётся из структуры каталогов или
CLI `--session-id`.

Перед разбиением команда `prepare` проверяет имена файлов исходного набора по
`dataset.augmented_name_markers`. По умолчанию подозрительными считаются
суффиксы вроде `_filter`, `_flip`, `_scale`, `_brightness`, `_contrast`,
`_color`, `_sharpness`, `_hflip`, `_vflip`. Если такие файлы найдены в
`raw_root/images`, `prepare` останавливается, потому что исходный набор должен
содержать только реальные фотографии. Обход этой защиты допускается только
явным `dataset.allow_augmented_source: true` и должен быть отдельно обоснован.

После подготовки датасета команда `prepare` дополнительно создаёт
`split_integrity_report.json`. В нём фиксируются:

- число изображений по частям разбиения;
- подозрительные аугментированные имена в `train/val/test`;
- подозрительные аугментированные изображения в `val/test`;
- базовые кадры, попавшие более чем в одну часть разбиения после удаления суффиксов
  аугментации.

Для независимой проверки уже подготовленного набора можно использовать:

```powershell
py -m seedling_experiments check-split --dataset E:/dataset/prepared_seedlings --output E:/dataset/prepared_seedlings/split_integrity_report.json
```

Критерий готовности: `split_integrity_report.json` должен иметь `ok: true`.

## 2. Аудит датасета

`dataset_audit.json` фиксирует:

- число изображений по частям разбиения;
- число bbox по классам;
- отсутствующие файлы разметки;
- лишние файлы разметки;
- подозрительные аугментированные имена файлов;
- нормированные размеры bbox.

`prepare` автоматически сохраняет оба аудита рядом с подготовленным датасетом:

- `raw_dataset_audit.json` до разбиения фиксирует исходные отсутствующие и лишние файлы разметки;
- `dataset_audit.json` после `prepare` фиксирует фактический набор `train/val/test`.

Это важно, потому что `prepare` создаёт пустой файл разметки для изображения без
исходной разметки. После этого аудит подготовленного набора видит пустой `.txt`,
а не отсутствующую разметку. Лишние файлы разметки из исходного `labels/` в
разбиение не копируются.

Перед статьёй стоит дополнить эти аудиты внешней таблицей: дата съёмки, порода,
размер сеянца, контейнер/лоток, число ячеек и число множественных ячеек.

Новый слой данных добавляет отдельные проверки для расширенной разметки:

```powershell
py -m seedling_data audit-cell-annotations --path data/zks_v0_1/manifests/cell_annotations.jsonl --ontology configs/ontology/ontology_v0_1.yaml --grid-rows 11 --grid-cols 11
py -m seedling_data audit-action-points --path data/zks_v0_1/manifests/action_points.jsonl --ontology configs/ontology/ontology_v0_1.yaml --cell-annotations data/zks_v0_1/manifests/cell_annotations.jsonl --min-safe-distance-px 12 --max-uncertainty-px 4
py -m seedling_data near-duplicates --manifest data/zks_v0_1/manifests/image_manifest.csv --out reports/near_duplicates.json
```

`near-duplicates` считает неполный манифест ошибкой проверки: каждая строка
должна содержать `split`, `sha256` и `file_path`, иначе нельзя утверждать, что
между выборками нет утечки по хэшу или pHash.

Минимальный набор артефактов подготовки датасета:

```text
raw_dataset_audit.json
config.yaml
run_snapshot.json
data.yaml
split_manifest.csv
train_augmentation_manifest.csv
split_integrity_report.json
dataset_audit.json
prepare_summary.json
artifact_registry.json
artifact_registry.csv
```

Если `dataset.augment_train` выключен, `train_augmentation_manifest.csv`
создаётся с одним заголовком, а `prepare_summary.json` фиксирует
`augmentation.created_images: 0`. `artifact_registry.json` содержит входной
config, исходный набор данных и все основные выходные файлы `prepare`.

После создания манифеста и разбиения набор данных должен быть зарегистрирован:

```powershell
py -m seedling_data registry add --dataset-root data/zks_v0_1 --name zks_v0_1
py -m seedling_data registry validate --dataset zks_v0_1
py -m seedling_data registry summarize --dataset zks_v0_1 --out reports/dataset_summary.md
py -m seedling_data registry summarize --dataset zks_v0_1 --out reports/dataset_summary.html
```

`dataset_summary.md`/`.html` должен показывать распределения по частям разбиения, идентификаторам классов YOLO,
состояния ячеек и типы целевых точек, если соответствующие манифесты,
`cell_annotations.jsonl` и `action_points.jsonl` присутствуют в версии данных.

Для `cell_metrics.json` нужно строить отдельную диагностику bootstrap CI и
таксономии ошибок:

```powershell
py -m seedling_reports diagnose-cell-metrics --metrics runs/evaluation/cell_metrics.json --out reports/cell_diagnostics
```

Единица bootstrap задаётся `--group-key`; по умолчанию используется уровень изображений
resampling, а не отдельные ячейки.
Если в `evaluate-cells` задан `evaluation.image_manifest` или рядом с датасетом
есть `manifests/image_manifest.csv`, строки `cell_metrics.json.images[]`
получают `group_id`, `tray_id`, `session_id` и другие поля manifest. Тогда
bootstrap можно запускать по кассете/группе:

```powershell
py -m seedling_reports diagnose-cell-metrics --metrics runs/evaluation/cell_metrics.json --out reports/cell_diagnostics --group-key group_id
```

Для нового schema-first контура экспертные и предсказанные сцены можно сравнить
на уровне `SceneState`:

```powershell
py -m seedling_experiments evaluate-scenes --gt data/annotations/expert_scene.json --pred runs/predict/scenes.json --out reports/scene_metrics.json
```

Команда `evaluate-cells` с конфигурацией тоже может работать с теми же схемами,
если заданы `evaluation.gt_scene` и `evaluation.pred_scene`. В этом режиме она
пишет обычные `cell_metrics.json` / `cell_confusion_matrix.csv` и добавляет
полный schema-first блок в `scene_state_metrics`.

```yaml
evaluation:
  gt_scene: data/annotations/expert_scene.json
  pred_scene: runs/predict/scenes.json
  output_dir: runs/schema_scene_cell_eval
  target_match_distance_px: 25
```

Этот оценщик читает `CellState` и `ActionTarget`, поддерживает расширенные
состояния (`weed_only`, `crop_and_weed`, `unknown`, `ambiguous`), считает ошибку
целей в px/mm при наличии координат, сравнивает экспертные решения keep/remove
и пишет метрики с ценой ошибок для критических случаев.
Старый режим YOLO `evaluate-cells` пишет такую же верхнеуровневую сводку
`cost_sensitive` из матрицы ошибок `empty/single/multiple` и чисел целей,
поэтому `task_level_results.csv` может сравнивать `cost_total`,
`critical_error_total` и `critical_error_rate_per_cell` между режимами оценки.
Schema-first блок `cell_metrics.cell_metrics` также включает `edge_cell_metrics`
и `corner_cell_metrics`: accuracy, числа ячеек, несовпадения и распределения
состояний для периметра и углов кассеты. Поэтому отказы на краях и углах можно
показывать отдельно от сильно несбалансированной accuracy всей кассеты.

## 3. Метрики конечной задачи

`evaluate-cells` сравнивает предсказания с YOLO-разметкой и считает:

- матрицу ошибок для ячеек `empty / single / multiple`;
- macro-точность, полнота и F1;
- precision/recall множественных ячеек;
- precision/recall целей удаления;
- среднюю ошибку координат удаления.
- сводку с ценой ошибок для пропущенных ячеек с несколькими сеянцами, ложных
  целей удаления и пропущенных целей удаления.

Это важнее для задачи лазерного прореживания, чем только `mAP`.

Матрица ячеек строится равномерным делением bbox контейнера на
`grid_rows x grid_cols`. Сеянец назначается в ячейку по центру bbox. Цели
удаления формируются только в ячейках `multiple`: текущая эвристика оставляет
сеянец с максимальной площадью bbox, а остальные считает кандидатами на удаление.
Это базовое правило сравнения, которое нужно отдельно проверять экспертной разметкой
`keep/remove`.

В schema-first контуре `CellStateBuilder` может использовать
`TrayState.corners_px`: сетка строится как набор четырёхугольных polygon cells,
а объекты назначаются через алгоритм point-in-polygon. Это отличается от старого
YOLO-оценки выше, которая сохраняет bbox-based контракт для сопоставимости с
ранними результатами.

Перед расчётом метрик `evaluate-cells` проверяет согласованность
`predictions.json` с выбранной частью разбиения. По умолчанию включены две защиты:

- `evaluation.strict_predictions: true` — каждое изображение из `evaluation.dataset`
  / `evaluation.split` должно присутствовать в `predictions.json`;
- `evaluation.reject_augmented_eval_images: true` — для `val` и `test` оценка
  останавливается, если имена изображений похожи на аугментированные.

В `cell_metrics.json` сохраняются диагностические поля `prediction_coverage` и
`suspected_augmented_eval_images`. Для финальных чисел статьи список
`missing_predictions` должен быть пустым, а `suspected_augmented_eval_images`
должен быть пустым.

### 3.1. Диагностика сопоставления контейнеров

Поля `container_recall` и `matched_containers` в `cell_metrics.json` являются
диагностикой сопоставления контейнеров. Они не означают напрямую, что модель
«видит» или «не видит» контейнер на визуализации.

При `use_ground_truth_containers: false` команда `evaluate-cells` сопоставляет
каждый контейнер из YOLO-разметки с предсказанным контейнером по IoU. Порог
задаётся параметром `evaluation.container_iou`, по умолчанию `0.5`. Если лучший
IoU ниже этого порога, контейнер остаётся unmatched, даже если bbox выглядит
похожим при ручном просмотре.

Типичные причины `matched_containers = 0`:

- сдвиг или другой масштаб bbox даёт IoU ниже порога;
- модель выдаёт несколько фрагментов контейнера вместо одного общего bbox;
- перепутаны `container_class` и `seedling_class`;
- визуализация показывает postprocessed контейнеры, а `evaluate-cells` читает
  сырые `detections` из `predictions.json`;
- в разметке и предсказаниях используются разные правила обведения контейнера.

Важно: `predictions.json` содержит как сырые `detections`, так и обработанные
`containers` после фильтрации/объединения. Базовая end-to-end оценка должна
использовать `evaluation.container_prediction_source: detections`. Для
диагностики можно временно поставить `containers`, чтобы проверить, помогает ли
postprocessing контейнеров сопоставлению с GT-разметкой. В статье нужно явно
указать, какой источник использован для финальных метрик.

`cell_metrics.json` содержит блок `container_matching`:

```text
prediction_source
iou_threshold
total_gt_containers
matched_containers
best_iou_summary
best_iou_counts
unmatched_samples
```

Если `container_recall = 0`, этот блок является обязательным источником анализа:
он показывает, есть ли вообще ненулевые IoU и сколько контейнеров проходит
пороги `0.25`, `0.5` и текущий `container_iou`.

При несопоставленном контейнере метрики уровня ячеек не обнуляются автоматически:
`evaluate-cells` использует bbox GT-контейнера как опорную геометрию и
раскладывает по этой сетке предсказанные сеянцы. Поэтому `matched_containers = 0`
надо читать как отдельную диагностику детекции контейнера, а не как прямой
показатель качества матрицы ячеек.

Если нужно оценить качество матрицы ячеек без смешивания с ошибками детекции
контейнера, включите
`evaluation.use_ground_truth_containers: true` и явно опишите это в статье. Если
оценивается сквозной конвейер, оставьте `false`, но отдельно анализируйте IoU
контейнеров и причины unmatched.

## 4. Базовый метод сравнения

`baseline-green` реализует классический метод:

1. HSV-порог зелёного цвета.
2. Морфологические операции open/close.
3. Связные компоненты / контуры.
4. bbox каждого компонента как кандидат-сеянец.

При `use_known_containers: true` контейнеры берутся из разметки. Такой базовый метод
сравнивает именно качество поиска сеянцев и матрицы ячеек, не смешивая его с
ошибками детекции контейнера.

`baseline-green` создаёт `predictions.json`, но не создаёт объектный
`test_metrics.json` и не считает `mAP`. В таблицах результатов его корректно
сравнивать с YOLO по метрикам конечной задачи из `cell_metrics.json`; колонки
mAP для HSV-метода в текущем контуре должны быть `N/A`, если не добавлена
отдельная объектная оценка его предсказаний.

## 5. Трассировка артефактов

Каждая команда, запускаемая через конфиг (`train`, `val`, `predict`,
`evaluate-cells`, `baseline-green`, `prepare`), должна сохранять рядом с
результатами:

```text
config.yaml
run_snapshot.json
```

Каждый `run_snapshot.json` включает имя команды, сериализованные аргументы,
канонический SHA-256 `config_hash`, сохранённую конфигурацию, версию Python,
платформу и ключевые версии пакетов.

`run_snapshot.json` фиксирует команду, конфиг и окружение; `config.yaml`
сохраняет человекочитаемую копию конфига именно для этого этапа. Для любого
числа в статье должно быть понятно:

- из какого файла оно взято;
- каким конфигом получено;
- какими весами получено;
- на каком разбиении посчитано;
- какое случайное зерно и какая версия Ultralytics использованы.

Команда `val --split test` создаёт `test_metrics.json`. Помимо
mAP, точности и полноты в нём должны быть сохранены `validation` и `dataset`: путь
к модели, `data.yaml`, разбиение, `imgsz`/`conf`/`iou`, число `test`-изображений
и число объектов по классам.

## 6. Что добавить при следующем расширении

- SAHI/tiling для мелких сеянцев.
- Сравнение YOLO11n / YOLO11s / YOLOv8n на одном разбиении.
- Несколько случайных зёрен и доверительные интервалы по контейнерам.
- Разметка экспертного решения: какой сеянец оставить.
- Перевод ошибки координат из пикселей в миллиметры через калибровку камеры.
