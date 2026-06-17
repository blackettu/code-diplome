# Экспериментальный протокол

Этот протокол закрывает основные замечания рецензента по воспроизводимости.

Пошаговый план действий именно для подготовки статьи вынесен в
`docs/ARTICLE_EXPERIMENT_PLAN.md`.
Практический контракт для разработчика, который впервые запускает проект,
вынесен в `docs/DEVELOPER_GUIDE.md`.

## 1. Split до аугментации

Команда `prepare` сначала делит исходные изображения на `train/val/test`, а затем
применяет аугментацию только к `train`. Это предотвращает попадание близких копий
одного изображения в test.

Если несколько файлов относятся к одному контейнеру, дате или лотку, задайте
`dataset.split.group_regex` в конфиге. Все изображения с одним group id попадут
в один split.

Перед split команда `prepare` проверяет имена файлов исходного набора по
`dataset.augmented_name_markers`. По умолчанию подозрительными считаются
суффиксы вроде `_filter`, `_flip`, `_scale`, `_brightness`, `_contrast`,
`_color`, `_sharpness`, `_hflip`, `_vflip`. Если такие файлы найдены в
`raw_root/images`, `prepare` останавливается, потому что исходный набор должен
содержать только реальные фотографии. Обход этой защиты допускается только
явным `dataset.allow_augmented_source: true` и должен быть отдельно обоснован.

После подготовки датасета команда `prepare` дополнительно создаёт
`split_integrity_report.json`. В нём фиксируются:

- число изображений по split;
- подозрительные аугментированные имена в `train/val/test`;
- подозрительные аугментированные изображения в `val/test`;
- базовые кадры, попавшие более чем в один split после удаления суффиксов
  аугментации.

Для независимой проверки уже подготовленного набора можно использовать:

```powershell
py -m seedling_experiments check-split --dataset E:/dataset/prepared_seedlings --output E:/dataset/prepared_seedlings/split_integrity_report.json
```

Критерий готовности: `split_integrity_report.json` должен иметь `ok: true`.

## 2. Аудит датасета

`dataset_audit.json` фиксирует:

- число изображений по split;
- число bbox по классам;
- отсутствующие labels;
- orphan labels;
- подозрительные аугментированные имена файлов;
- нормированные размеры bbox.

Нужно различать два аудита:

- raw-аудит до `prepare` фиксирует исходные missing labels и orphan labels;
- prepared-аудит после `prepare` фиксирует фактический train/val/test набор.

Это важно, потому что `prepare` создаёт пустой label-файл для изображения без
исходной разметки. После этого prepared-аудит видит пустой `.txt`, а не missing
label. Orphan labels из исходного `labels/` в split не копируются.

Перед статьёй стоит дополнить эти аудиты внешней таблицей: дата съёмки, порода,
размер сеянца, контейнер/лоток, число ячеек и число множественных ячеек.

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
```

## 3. Метрики конечной задачи

`evaluate-cells` сравнивает предсказания с YOLO-разметкой и считает:

- матрицу ошибок для ячеек `empty / single / multiple`;
- macro precision/recall/F1;
- precision/recall множественных ячеек;
- precision/recall целей удаления;
- среднюю ошибку координат удаления.

Это важнее для задачи лазерного прореживания, чем только `mAP`.

Матрица ячеек строится равномерным делением bbox контейнера на
`grid_rows x grid_cols`. Сеянец назначается в ячейку по центру bbox. Цели
удаления формируются только в ячейках `multiple`: текущая эвристика оставляет
сеянец с максимальной площадью bbox, а остальные считает кандидатами на удаление.
Это baseline-правило, которое нужно отдельно проверять экспертной разметкой
`keep/remove`.

Перед расчётом метрик `evaluate-cells` проверяет согласованность
`predictions.json` с выбранным split. По умолчанию включены две защиты:

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

При unmatched контейнере cell-level метрики не обнуляются автоматически:
`evaluate-cells` использует bbox GT-контейнера как опорную геометрию и
раскладывает по этой сетке предсказанные сеянцы. Поэтому `matched_containers = 0`
надо читать как отдельную диагностику детекции контейнера, а не как прямой
показатель качества матрицы ячеек.

Если нужно оценить качество матрицы ячеек без смешивания с ошибками детекции
контейнера, включите
`evaluation.use_ground_truth_containers: true` и явно опишите это в статье. Если
оценивается end-to-end pipeline, оставьте `false`, но отдельно анализируйте IoU
контейнеров и причины unmatched.

## 4. Baseline

`baseline-green` реализует классический метод:

1. HSV-порог зелёного цвета.
2. Morphology open/close.
3. Connected components / contours.
4. bbox каждого компонента как кандидат-сеянец.

При `use_known_containers: true` контейнеры берутся из разметки. Такой baseline
сравнивает именно качество поиска сеянцев и матрицы ячеек, не смешивая его с
ошибками детекции контейнера.

`baseline-green` создаёт `predictions.json`, но не создаёт object-level
`test_metrics.json` и не считает `mAP`. В таблицах результатов его корректно
сравнивать с YOLO по task-level метрикам из `cell_metrics.json`; колонки mAP для
HSV baseline в текущем контуре должны быть `N/A`, если не добавлена отдельная
оценка object detection для baseline-предсказаний.

## 5. Трассировка артефактов

Каждая команда, запускаемая через конфиг (`train`, `val`, `predict`,
`evaluate-cells`, `baseline-green`, `prepare`), должна сохранять рядом с
результатами:

```text
config.yaml
run_snapshot.json
```

`run_snapshot.json` фиксирует команду, конфиг и окружение; `config.yaml`
сохраняет человекочитаемую копию конфига именно для этого этапа. Для любого
числа в статье должно быть понятно:

- из какого файла оно взято;
- каким конфигом получено;
- какими весами получено;
- на каком split посчитано;
- какой seed и какая версия Ultralytics использованы.

Команда `val --split test` создаёт `test_metrics.json`. Помимо mAP/precision/recall
в нём должны быть сохранены `validation` и `dataset`: путь к модели, data.yaml,
split, imgsz/conf/iou, число test-изображений и число объектов по классам.

## 6. Что добавить при следующем расширении

- SAHI/tiling для мелких сеянцев.
- Сравнение YOLO11n / YOLO11s / YOLOv8n на одном split.
- Несколько seed и доверительные интервалы по контейнерам.
- Разметка экспертного решения: какой сеянец оставить.
- Перевод ошибки координат из пикселей в миллиметры через калибровку камеры.
