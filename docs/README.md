# Документация проекта code-diplome

Это входная точка документации по воспроизводимому контуру
`seedling_experiments`. Все рабочие инструкции, протоколы и методические
заметки должны находиться в этой папке `docs/`.

Корневой `README.md` оставлен только как короткий указатель на этот файл.

## Карта документов

- `DEVELOPER_GUIDE.md` - практический контракт для запуска проекта: формат
  датасета, конфиг, артефакты, `predictions.json`, `cell_metrics.json` и
  диагностика типичных ошибок.
- `EXPERIMENT_PROTOCOL.md` - экспериментальный протокол: честный split,
  аугментация только train, baseline, метрики конечной задачи и трассировка
  артефактов.
- `ARTICLE_EXPERIMENT_PLAN.md` - пошаговый план получения результатов для
  статьи.
- `dataset_collection_recommendations.md` - рекомендации по сбору и разметке
  нового датасета.
- `vision_automated_weed_removal.md` - продуктовое и научное видение системы
  автоматизированного удаления сорняков/лишних сеянцев.
- `necessary_corrections.md` - список методических правок для статьи и
  результатов.

## Назначение проекта

Код предназначен для экспериментов по детекции контейнеризированных сеянцев,
построению матриц заполнения ячеек и оценке целей удаления при множественном
прорастании.

Новые воспроизводимые эксперименты запускаются через пакет
`seedling_experiments`. Старые скрипты в корне проекта оставлены как прототипы и
не должны использоваться для новых результатов, которые нужно защищать в статье.

## Быстрый старт

Все команды запускаются из корня проекта `code-diplome/`, где лежат
`requirements.txt`, `configs/` и пакет `seedling_experiments/`.

```powershell
py -m pip install -r requirements.txt
py -m seedling_experiments --help
```

CLI должен показывать команды:

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

## Входной датасет

Исходный датасет должен быть в YOLO-формате:

```text
raw_seedlings/
  images/
  labels/
```

Для текущих конфигов ожидается порядок классов:

```text
0 = container
1 = seedlings
```

Перед `prepare` желательно сохранить аудит исходного набора:

```powershell
py -m seedling_experiments audit --dataset E:/dataset/raw_seedlings --output E:/dataset/raw_seedlings/raw_dataset_audit.json
```

Это фиксирует missing/orphan labels до подготовки. После `prepare` отсутствующие
labels превращаются в пустые `.txt` в `prepared_root`.

## Основной порядок эксперимента

1. Отредактировать `configs/example_experiment.yaml`: пути, seed, модель,
   пороги, размер сетки и параметры baseline.

2. Сделать split до аугментации, применить аугментацию только к `train` и
   получить аудит датасета:

```powershell
py -m seedling_experiments prepare --config configs/example_experiment.yaml
```

Основные артефакты `prepare`:

```text
prepared_root/config.yaml
prepared_root/run_snapshot.json
prepared_root/data.yaml
prepared_root/split_summary.json
prepared_root/split_manifest.csv
prepared_root/train_augmentation_manifest.csv
prepared_root/split_integrity_report.json
prepared_root/dataset_audit.json
prepared_root/prepare_summary.json
```

Если `dataset.augment_train` выключен, `train_augmentation_manifest.csv` может
содержать только заголовок или не быть значимым для анализа.

3. Обучить YOLO:

```powershell
py -m seedling_experiments train --config configs/example_experiment.yaml
```

4. Проверить модель на независимом test split:

```powershell
py -m seedling_experiments val --config configs/example_experiment.yaml --split test
py -m seedling_experiments predict --config configs/example_experiment.yaml
py -m seedling_experiments evaluate-cells --config configs/example_experiment.yaml
```

Ключевой файл для статьи: `cell_metrics.json`. В нём есть:

- accuracy по ячейкам;
- macro precision/recall/F1 для классов `0 / 1 / >1`;
- precision/recall для множественных ячеек;
- precision/recall целей удаления;
- средняя ошибка координат удаления в пикселях;
- bootstrap 95% CI для accuracy по ячейкам;
- диагностика контейнеров и покрытия `predictions.json`.

Матрица ячеек строится равномерным делением bbox контейнера на `grid_rows x
grid_cols`; каждый сеянец назначается в ячейку по центру своего bbox.
`removal_targets` используют baseline-эвристику: в ячейке `multiple`
оставляется самый большой bbox, остальные считаются кандидатами на удаление.

5. Запустить простой baseline:

```powershell
py -m seedling_experiments baseline-green --config configs/example_experiment.yaml
```

После этого baseline можно оценить той же командой `evaluate-cells`, указав в
конфиге путь к его `predictions.json`. `baseline-green` не создаёт `mAP`; его
сравнивают с YOLO по task-level метрикам из `cell_metrics.json`.

## Важные настройки

Относительные пути в конфиге считаются от текущей рабочей директории, а не от
файла конфига. Для повторяемого эксперимента запускайте команды из
`code-diplome/` и используйте новый пустой `dataset.prepared_root` для каждого
полного `prepare`.

Пороги `validation.conf/iou` относятся к команде `val`, а пороги
`prediction.conf/iou` - к `predictions.json`. После изменения параметров
`prediction.*` нужно заново выполнить `predict`, а затем `evaluate-cells`.

По умолчанию финальная end-to-end оценка контейнеров использует:

```yaml
evaluation:
  container_prediction_source: detections
  strict_predictions: true
  reject_augmented_eval_images: true
  use_ground_truth_containers: false
```

`container_prediction_source: containers` можно использовать для отдельной
диагностики postprocessing, но финальную таблицу нужно подписывать по тому
источнику контейнеров, который реально использован.

Если нужно оценить только качество поиска сеянцев и матрицы ячеек при известной
геометрии кассеты, используйте `evaluation.use_ground_truth_containers: true`.
Такой режим нельзя называть полной end-to-end оценкой детектора контейнеров.

## Быстрые низкоуровневые команды

```powershell
py -m seedling_experiments audit --dataset E:/dataset/raw_seedlings --output E:/dataset/raw_seedlings/raw_dataset_audit.json
py -m seedling_experiments split --source E:/dataset/raw_seedlings --output E:/dataset/prepared_seedlings --seed 42
py -m seedling_experiments check-split --dataset E:/dataset/prepared_seedlings --output E:/dataset/prepared_seedlings/split_integrity_report.json
py -m seedling_experiments audit --dataset E:/dataset/prepared_seedlings --output E:/dataset/prepared_seedlings/dataset_audit.json
```

Для полного воспроизводимого сценария предпочтительнее использовать
`prepare --config`, потому что он сохраняет `config.yaml`, `run_snapshot.json`,
split, train-only аугментацию, integrity-check и аудит одним согласованным
запуском.

## Старые скрипты

- `learning_yolo.py` - исходный минимальный запуск обучения YOLO.
- `matrix_for_container.py` - прототип построения матриц по детекциям.
- `many_seedlings.py` - прототип визуализации множественных сеянцев.
- `filter.py` - прототип фотометрической аугментации.
- `model_info.py` - исходный вывод метрик модели.

Для новых результатов используйте CLI выше: он фиксирует split, аугментации,
параметры запуска, окружение и метрики конечной задачи.
