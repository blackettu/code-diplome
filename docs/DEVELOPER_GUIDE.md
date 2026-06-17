# Developer guide для воспроизводимого контура

Этот файл фиксирует практический контракт для разработчика, который впервые
запускает проект без устных пояснений. Научную интерпретацию результатов см.
в `docs/EXPERIMENT_PROTOCOL.md` и `docs/ARTICLE_EXPERIMENT_PLAN.md`.

## 1. Откуда запускать команды

Команды ниже предполагают, что текущая рабочая директория — корень репозитория
`code-diplome/`, где лежат `requirements.txt`, `configs/` и пакет
`seedling_experiments/`.

```powershell
cd code-diplome
py -m pip install -r requirements.txt
py -m seedling_experiments --help
```

CLI должен показать команды:

```text
prepare
train
val
predict
evaluate-cells
baseline-green
split
audit
check-split
```

## 2. Входной датасет

Исходный датасет должен быть в YOLO-формате:

```text
raw_seedlings/
  images/
    image_001.jpg
  labels/
    image_001.txt
```

Поддерживаемые расширения изображений: `.jpg`, `.jpeg`, `.png`, `.bmp`,
`.heic`, `.HEIC`.

Каждая строка label-файла:

```text
class_id x_center y_center width height
```

Координаты должны быть нормированы в диапазоне `0..1`, как в стандартной
YOLO-разметке. Для текущих конфигов ожидается порядок классов:

```text
0 = container
1 = seedlings
```

Если label-файл отсутствует, `prepare` создаст пустой label в подготовленном
датасете. Это удобно для запуска, но для статьи такие случаи надо отдельно
проверять до `prepare`.

Перед первым `prepare` полезно сохранить аудит исходного набора:

```powershell
py -m seedling_experiments audit --dataset E:/dataset/raw_seedlings --output E:/dataset/raw_seedlings/raw_dataset_audit.json
```

Это важно, потому что `prepare` копирует изображения в `prepared_root`, а при
отсутствующем исходном label-файле создаёт пустой `.txt`. После этого аудит
подготовленного датасета уже видит существующий пустой label, а не missing label.
Orphan labels из исходного `labels/` также не копируются в split. Поэтому для
статьи лучше сохранять оба аудита: raw до `prepare` и prepared после `prepare`.

## 3. Основной pipeline

Минимальная последовательность:

```powershell
py -m seedling_experiments prepare --config configs/example_experiment.yaml
py -m seedling_experiments train --config configs/example_experiment.yaml
py -m seedling_experiments val --config configs/example_experiment.yaml --split test
py -m seedling_experiments predict --config configs/example_experiment.yaml
py -m seedling_experiments evaluate-cells --config configs/example_experiment.yaml
```

`prepare` выполняет три действия:

- делит исходные изображения на `train/val/test`;
- применяет аугментации только к `train`;
- сохраняет аудит подготовленного датасета.

Аудит, который создаёт `prepare`, относится к `dataset.prepared_root`. Он нужен
для проверки финального train/val/test набора. Для поиска исходных missing labels
и orphan labels используйте отдельный `audit` на `dataset.raw_root` до запуска
`prepare`.

`group_regex` в конфиге нужен, если несколько файлов относятся к одной кассете,
дате или серии съёмки. Все изображения с одним group id попадут в один split.
Если `group_regex` не задан, каждое изображение считается отдельной группой.

Split строится по перемешанным группам и последовательно заполняет `train`,
затем `val`, затем `test`. Поэтому при крупных группах фактические доли могут
немного отличаться от `0.7/0.2/0.1`. Split не стратифицирует классы и состояния
ячеек, поэтому после `prepare` обязательно проверьте `dataset_audit.json` и
убедитесь, что в `val` и `test` есть нужные классы и сложные случаи.

Команды `split` и `audit` без полного конфига являются низкоуровневыми
утилитами. Они не обучают модель, не запускают аугментации и не создают
`prepare_summary.json`. Для полного воспроизводимого сценария используйте
`prepare --config`.

## 4. Конфиг, пути и повторные запуски

Пути в YAML-конфиге передаются в код как обычные `Path(...)`. Относительные пути
разрешаются относительно текущей рабочей директории, а не относительно файла
конфига. Поэтому безопасный вариант — запускать команды из корня `code-diplome/`
или указывать абсолютные пути.

Примерный набор путей должен быть согласован между секциями:

```yaml
dataset:
  prepared_root: E:/dataset/prepared_seedlings

training:
  data: E:/dataset/prepared_seedlings/data.yaml

validation:
  data: E:/dataset/prepared_seedlings/data.yaml

prediction:
  images: E:/dataset/prepared_seedlings/test/images

evaluation:
  dataset: E:/dataset/prepared_seedlings
  split: test

baseline:
  images: E:/dataset/prepared_seedlings/test/images
  labels: E:/dataset/prepared_seedlings/test/labels
```

`prediction.images`, `evaluation.dataset + evaluation.split` и
`baseline.images/labels` должны указывать на один и тот же split, если
сравниваются YOLO, baseline и cell-level метрики.

Для воспроизводимого перезапуска используйте новый пустой `dataset.prepared_root`
или вручную очистите старый подготовленный датасет перед `prepare`. Текущая
команда `prepare` создаёт и перезаписывает файлы, но не является полноценной
операцией очистки каталога. Если повторно запускать её поверх старого
`prepared_root`, можно случайно смешать старые файлы и новые аугментации.

`training.model: yolo11n.pt` означает, что Ultralytics будет использовать
предобученные веса с таким именем. Если файл отсутствует локально, библиотека
может попытаться скачать его при первом запуске. В среде без интернета укажите
путь к заранее сохранённому `.pt`-файлу.

Каталоги результатов тоже лучше делать уникальными для каждого независимого
запуска:

```yaml
training:
  name: yolo11n_seedlings_2026_06_01

validation:
  output_dir: runs/yolo11n_seedlings_2026_06_01_eval

prediction:
  output_dir: runs/yolo11n_seedlings_2026_06_01_test_predictions

evaluation:
  output_dir: runs/yolo11n_seedlings_2026_06_01_cell_eval
```

По умолчанию `training.exist_ok` считается `true`, поэтому Ultralytics может
использовать существующий каталог `runs/<training.name>`. Для чистого сравнения
между экспериментами задавайте новое `training.name` и новые output-директории
или явно проверяйте, какие артефакты были перезаписаны.

## 5. Пороги и параметры по этапам

В конфиге есть несколько похожих параметров, но они относятся к разным этапам.

`training.imgsz` задаёт размер изображения при обучении.

`validation.imgsz`, `validation.conf` и `validation.iou` используются командой:

```powershell
py -m seedling_experiments val --config <config> --split test
```

Эти параметры влияют на `test_metrics.json`, но не создают и не меняют
`predictions.json`.

`prediction.imgsz`, `prediction.conf` и `prediction.iou` используются командой:

```powershell
py -m seedling_experiments predict --config <config>
```

Именно они определяют, какие bbox попадут в `predictions.json`. Если поменять
эти параметры, нужно заново запустить `predict`, а затем `evaluate-cells`.

`evaluation.container_iou` — это не NMS-порог YOLO. Он используется только при
сопоставлении GT-контейнера с предсказанным контейнером в `evaluate-cells`.

`evaluation.target_match_distance_px` — максимальное расстояние в пикселях между
GT и predicted `remove_center`, при котором цель удаления считается matched.

`prediction.min_container_area` и `prediction.merge_distance` применяются только
к postprocessed полю `containers` и `container_analysis` в `predictions.json`.
Текущая реализация `evaluate-cells` читает сырые `detections`, поэтому изменение
этих двух параметров само по себе не исправит `matched_containers`.

Отдельный нюанс: если `validation.run_after_train: true`, команда `train`
запускает внутреннюю валидацию Ultralytics после обучения и сохраняет её в
`metrics_summary.json`. Эта внутренняя валидация в текущем коде не использует
все параметры секции `validation`, например `validation.conf` и
`validation.iou`. Для контролируемой проверки с параметрами из конфига запускайте
отдельную команду `val`.

## 6. Основные артефакты

После `prepare`:

```text
prepared_root/config.yaml
prepared_root/run_snapshot.json
prepared_root/data.yaml
prepared_root/split_summary.json
prepared_root/split_manifest.csv
prepared_root/split_integrity_report.json
prepared_root/dataset_audit.json
prepared_root/prepare_summary.json
```

`prepared_root/dataset_audit.json` описывает уже подготовленный набор. Если
нужно зафиксировать проблемы исходного датасета, дополнительно сохраните
`raw_dataset_audit.json` до `prepare`.

Если в конфиге включён блок `dataset.augment_train`, дополнительно создаётся:

```text
prepared_root/train_augmentation_manifest.csv
```

После `train`:

```text
runs/<training.name>/run_snapshot.json
runs/<training.name>/metrics_summary.json
runs/<training.name>/weights/best.pt
```

После `val --split test`:

```text
<validation.output_dir>/run_snapshot.json
<validation.output_dir>/test_metrics.json
```

После `predict` или `baseline-green`:

```text
<output_dir>/run_snapshot.json
<output_dir>/predictions.json
```

После `evaluate-cells`:

```text
<evaluation.output_dir>/run_snapshot.json
<evaluation.output_dir>/cell_metrics.json
<evaluation.output_dir>/cell_confusion_matrix.csv
```

## 7. Контракт `predictions.json`

`predictions.json` содержит список изображений. Для каждого изображения:

```text
image
path
width
height
detections
containers
seedlings
container_analysis
```

Смысл полей:

- `detections` — сырые bbox, полученные от YOLO или baseline.
- `containers` — bbox контейнеров после фильтрации и объединения мелких
  фрагментов.
- `seedlings` — bbox сеянцев, выбранные из сырых `detections` по
  `seedling_class`.
- `container_analysis` — матрица ячеек и цели удаления, рассчитанные по
  postprocessed `containers`.

Один bbox записывается так:

```json
{
  "class_id": 1,
  "name": "seedlings",
  "confidence": 0.91,
  "box": [10.0, 20.0, 50.0, 80.0],
  "center": [30.0, 50.0],
  "area": 1234.5
}
```

Здесь `box` уже в пикселях, формат `xyxy`, а не нормированный YOLO-формат.

Важно: текущая команда `evaluate-cells` берёт предсказанные объекты из
`detections`, а не из `containers`. Поэтому визуализация по `container_analysis`
может выглядеть лучше, чем диагностика `matched_containers` в `cell_metrics.json`.

## 8. Как строится матрица ячеек

Матрица строится внутри bbox контейнера. Параметры `grid_rows` и `grid_cols`
задают число строк и столбцов, по умолчанию `11 x 11`.

Система координат соответствует изображению:

- `x` растёт слева направо;
- `y` растёт сверху вниз;
- `row = 0` — верхняя строка контейнера;
- `col = 0` — левый столбец контейнера.

Каждая ячейка получается равномерным делением bbox контейнера. Сеянец
назначается в ячейку по центру своего bbox, а не по площади пересечения и не по
маске. Если центр bbox сеянца находится за пределами bbox контейнера, этот
сеянец не попадает ни в одну ячейку.

`matrix` в `container_analysis` — это двумерный список размера
`grid_rows x grid_cols`, где каждое число означает количество сеянцев в ячейке.
Для метрик эти количества сворачиваются в три класса:

```text
0 = empty
1 = single
2 = multiple
```

`removal_targets` формируются только для ячеек класса `multiple`. Текущая
эвристика оставляет сеянец с максимальной площадью bbox (`keep_box`), а все
остальные bbox в этой ячейке считает кандидатами на удаление (`remove_box`,
`remove_center`). Это baseline-правило, а не биологически доказанный критерий
качества сеянца.

Пример одного target:

```json
{
  "row": 3,
  "col": 7,
  "keep_box": [10.0, 20.0, 50.0, 80.0],
  "remove_box": [60.0, 25.0, 90.0, 70.0],
  "remove_center": [75.0, 47.5]
}
```

## 9. Контракт `cell_metrics.json`

Главные поля:

- `cell_accuracy` — общая точность классификации ячеек.
- `cell_macro` — precision/recall/F1 для классов ячеек.
- `cell_confusion_matrix` — матрица ошибок.
- `multi_seedling_cell` — precision/recall/F1 для ячеек с несколькими сеянцами.
- `removal_targets` — precision/recall/F1 целей удаления и ошибка координат.
- `container_recall` — доля GT-контейнеров, сопоставленных с предсказанными.
- `cell_accuracy_bootstrap_ci` — bootstrap 95% CI для image-level точности
  ячеек. В текущей реализации ресэмплируются значения `cell_accuracy` по
  изображениям, где были оценённые ячейки.
- `images` — краткая диагностика по каждому изображению.

Классы ячеек:

```text
0 = empty
1 = single
2 = multiple
```

В `cell_confusion_matrix` строки соответствуют GT-классам, столбцы —
предсказанным классам. CSV-файл `cell_confusion_matrix.csv` сохраняет ту же
матрицу с заголовками.

Цели удаления сопоставляются по расстоянию между центрами `remove_center`.
Порог задаётся `evaluation.target_match_distance_px`.

## 10. Контейнеры и `matched_containers`

При `evaluation.use_ground_truth_containers: false` оценка работает как
end-to-end pipeline:

1. Берёт GT-контейнеры из YOLO-разметки.
2. Берёт предсказанные контейнеры из сырых `detections`.
3. Сопоставляет их по IoU.
4. Считает контейнер matched, если IoU не ниже `evaluation.container_iou`.

По умолчанию `container_iou: 0.5`.

Если контейнер unmatched, cell-level оценка всё равно выполняется: для сетки
используется GT bbox контейнера, а предсказанные сеянцы раскладываются по этой
сетке. Поэтому `matched_containers` и `container_recall` нужно читать как
отдельную диагностику контейнеров, а не как прямую замену `cell_accuracy`.

`matched_containers = 0` не доказывает, что модель вообще не нашла контейнер.
Частые причины:

- bbox визуально близкий, но IoU ниже порога;
- модель нашла несколько частей контейнера вместо одного bbox;
- визуализация построена по `containers`, а оценка идёт по `detections`;
- перепутан порядок классов;
- разметка и модель используют разные правила обведения кассеты.

Если нужно проверить только качество поиска сеянцев и матрицы ячеек при
известной геометрии, используйте:

```yaml
evaluation:
  use_ground_truth_containers: true
```

В таком режиме cell-level метрики нельзя называть полной end-to-end оценкой
детектора контейнеров.

## 11. Baseline

`baseline-green` ищет зелёные компоненты по HSV-порогам и превращает каждый
компонент в bbox сеянца. Если в конфиге задано:

```yaml
baseline:
  use_known_containers: true
```

то bbox контейнеров добавляются из разметки. Это честный baseline для сравнения
поиска сеянцев и матрицы ячеек, но не baseline для детекции контейнера.

Чтобы оценить baseline через `evaluate-cells`, в `evaluation.predictions` нужно
временно указать `predictions.json`, созданный командой `baseline-green`.

Важно: `baseline-green` не запускает Ultralytics validation и не создаёт
`test_metrics.json`, `mAP@50` или `mAP@50-95`. Его корректно сравнивать с YOLO
по task-level метрикам из `cell_metrics.json`: `cell_accuracy`,
`multi_seedling_cell`, `removal_targets`, coordinate error. Object-level mAP для
YOLO берётся из команды `val`; для HSV baseline в текущем контуре это поле
следует оставлять `N/A`, если не реализована отдельная object-detection оценка
baseline-предсказаний.

Все bbox от HSV baseline получают `confidence = 1.0` и имя `green_component`.
Это не вероятность модели, а техническое значение для совместимости формата
`predictions.json`.

## 12. Частые проверки при странных результатах

Если `cell_accuracy` равен `null` или почти нет ячеек:

- проверьте, что в test-разметке есть объекты класса `container`;
- проверьте `evaluation.container_class` и `evaluation.seedling_class`;
- проверьте, что `evaluation.dataset` и `evaluation.split` указывают на тот же
  набор, по которому создавался `predictions.json`.

Если матрица ячеек выглядит сдвинутой:

- проверьте, что bbox контейнера размечен по тем же границам, которые ожидаются
  для сетки;
- проверьте, что `grid_rows` и `grid_cols` соответствуют реальной кассете;
- помните, что сеянец назначается по центру bbox, поэтому широкий или наклонный
  bbox может попасть в соседнюю ячейку.

Если `matched_containers = 0`, но на визуализации контейнер есть:

- сравните bbox из `detections` и `containers` в `predictions.json`;
- временно уменьшите `evaluation.container_iou` только для диагностики;
- проверьте, что визуализация использует те же поля, что `evaluate-cells`;
- при оценке только cell-level задачи включите `use_ground_truth_containers`.
- не ожидайте, что `prediction.min_container_area` или `merge_distance` изменят
  matching в `evaluate-cells`, пока оценка читает сырые `detections`.

Если изменили `prediction.conf`, `prediction.iou` или `prediction.imgsz`:

- заново запустите `predict`;
- затем заново запустите `evaluate-cells`;
- убедитесь, что `evaluation.predictions` указывает на новый `predictions.json`.

Если в таблице baseline появляются `mAP`-значения:

- проверьте, откуда они взяты;
- не переносите `mAP` из YOLO-валидации в строку HSV baseline;
- для текущего `baseline-green` используйте `N/A` в object-level колонках и
  сравнивайте методы по cell-level метрикам.

Если метрики кажутся слишком хорошими:

- убедитесь, что split был сделан до аугментации;
- проверьте `split_manifest.csv`;
- используйте `group_regex` для серий изображений одной кассеты или даты;
- проверьте, что `runs/...` и `prepared_root` относятся именно к текущему
  эксперименту, а не к предыдущему запуску с тем же именем.

Если в `dataset_audit.json` нет missing labels, но вы ожидали их увидеть:

- проверьте, какой набор аудировался: raw или prepared;
- помните, что `prepare` создаёт пустые label-файлы для изображений без исходной
  разметки;
- для исходных missing/orphan labels смотрите raw-аудит, сделанный до `prepare`.

Если после повторного `prepare` неожиданно выросло число train-изображений:

- проверьте, не запускалась ли аугментация поверх уже аугментированного
  `prepared_root`;
- используйте новый `prepared_root` для каждого независимого эксперимента;
- сравните `split_manifest.csv` и `train_augmentation_manifest.csv`.

Если `train` или первый запуск с `yolo11n.pt` падает из-за загрузки весов:

- скачайте веса заранее в доступной среде;
- замените `training.model` на локальный путь к `.pt`;
- проверьте, что `validation.model` и `prediction.model` указывают на ожидаемые
  веса после обучения.

## 13. Ограничения текущего контура

Текущий контур оценивает классы `container` и `seedlings`. Если в статье
обсуждаются сорняки, нужен отдельный класс `weed` или явная экспертная разметка
нежелательных растений.

Пиксельные координаты целей удаления ещё не являются координатами исполнительного
механизма. Для физического удаления нужны калибровка камеры, перевод в
миллиметры, оценка ошибки наведения и отдельный safety protocol.
