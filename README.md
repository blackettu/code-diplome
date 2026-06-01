# code-diplome

Код для экспериментов по детекции контейнеризированных сеянцев, построению матриц
заполнения ячеек и оценке целей удаления при множественном прорастании.

В проект добавлен воспроизводимый экспериментальный контур `seedling_experiments`.
Старые скрипты оставлены как прототипы, но новые эксперименты лучше запускать через CLI.

## Установка

```powershell
py -m pip install -r requirements.txt
```

Все команды ниже запускаются из корня репозитория `code-diplome/`. Практический
контракт для внешнего разработчика — формат датасета, структура JSON-артефактов
и диагностика типичных ошибок — описан в `docs/DEVELOPER_GUIDE.md`.

## Рекомендуемый порядок эксперимента

1. Подготовить исходный YOLO-датасет:

```text
raw_seedlings/
  images/
  labels/
```

Перед `prepare` желательно сохранить аудит исходного набора:

```powershell
py -m seedling_experiments audit --dataset E:/dataset/raw_seedlings --output E:/dataset/raw_seedlings/raw_dataset_audit.json
```

Это фиксирует missing/orphan labels до подготовки. После `prepare` отсутствующие
labels превращаются в пустые `.txt` в `prepared_root`.

2. Отредактировать `configs/example_experiment.yaml`: пути, seed, модель, пороги,
размер сетки и параметры baseline.

Важно: относительные пути в конфиге считаются от текущей рабочей директории.
Для повторяемого эксперимента запускайте команды из `code-diplome/` и используйте
новый пустой `dataset.prepared_root` для каждого полного `prepare`. Для каждого
независимого обучения также лучше задавать новое `training.name` и новые
`output_dir`, чтобы не смешивать артефакты в `runs/...`.

3. Сделать split до аугментации, применить аугментацию только к `train` и получить
аудит датасета:

```powershell
py -m seedling_experiments prepare --config configs/example_experiment.yaml
```

Будут созданы:

- `data.yaml` для Ultralytics;
- `split_manifest.csv`;
- `train_augmentation_manifest.csv`;
- `dataset_audit.json`;
- `prepare_summary.json`.

4. Обучить YOLO:

```powershell
py -m seedling_experiments train --config configs/example_experiment.yaml
```

Каждый запуск сохраняет `run_snapshot.json` с конфигом, версией Python и версиями
пакетов. Это нужно для воспроизводимости.

5. Проверить модель на независимом test split:

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
- bootstrap 95% CI для accuracy по ячейкам.

Матрица ячеек строится равномерным делением bbox контейнера на `grid_rows x
grid_cols`; каждый сеянец назначается в ячейку по центру своего bbox.
`removal_targets` сейчас используют baseline-эвристику: в ячейке `multiple`
оставляется самый большой bbox, остальные считаются кандидатами на удаление.

Пороги `validation.conf/iou` относятся к команде `val`, а пороги
`prediction.conf/iou` — к `predictions.json`. После изменения параметров
`prediction.*` нужно заново выполнить `predict`, а затем `evaluate-cells`.

В `cell_metrics.json` также есть диагностические поля по контейнерам:
`container_recall` и `matched_containers` по изображениям. Контейнер считается
matched только если bbox из разметки и bbox из предсказания имеют IoU не ниже
`evaluation.container_iou` из конфига. Визуально похожий bbox может не пройти
этот порог из-за сдвига, другого размера, обрезки или разбиения контейнера на
несколько боксов. В текущей реализации `evaluate-cells` читает сырые
`detections` из `predictions.json`, а не postprocessed поле `containers`.
Если контейнер не matched, cell-level метрики всё равно считаются: для привязки
предсказанных сеянцев к ячейкам используется bbox контейнера из разметки.
Если нужно оценить только качество поиска сеянцев и матрицы ячеек при известной
геометрии кассеты, используйте `evaluation.use_ground_truth_containers: true`.

6. Запустить простой baseline:

```powershell
py -m seedling_experiments baseline-green --config configs/example_experiment.yaml
```

После этого можно оценить baseline тем же `evaluate-cells`, указав в конфиге путь
к его `predictions.json`. Для baseline допустимо использовать известные bbox
контейнеров из разметки, чтобы честно сравнить именно метод поиска сеянцев.
`baseline-green` не создаёт `mAP`; его сравнивают с YOLO по `cell_metrics.json`,
а object-level mAP в текущем контуре берётся только из команды `val` для YOLO.

## Быстрые команды без полного конфига

```powershell
py -m seedling_experiments audit --dataset E:/dataset/raw_seedlings --output E:/dataset/raw_seedlings/raw_dataset_audit.json
py -m seedling_experiments split --source E:/dataset/raw_seedlings --output E:/dataset/prepared_seedlings --seed 42
py -m seedling_experiments audit --dataset E:/dataset/prepared_seedlings --output E:/dataset/prepared_seedlings/dataset_audit.json
```

## Старые скрипты

- `learning_yolo.py` - исходный минимальный запуск обучения YOLO.
- `matrix_for_container.py` - прототип построения матриц по детекциям.
- `many_seedlings.py` - прототип визуализации множественных сеянцев.
- `filter.py` - прототип фотометрической аугментации.
- `model_info.py` - исходный вывод метрик модели.

Для новых результатов, которые нужно защищать в статье, используйте CLI выше:
он фиксирует split, аугментации, параметры запуска и метрики конечной задачи.
