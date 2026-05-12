# code-diplome

Код для экспериментов по детекции контейнеризированных сеянцев, построению матриц
заполнения ячеек и оценке целей удаления при множественном прорастании.

В проект добавлен воспроизводимый экспериментальный контур `seedling_experiments`.
Старые скрипты оставлены как прототипы, но новые эксперименты лучше запускать через CLI.

## Установка

```powershell
py -m pip install -r requirements.txt
```

## Рекомендуемый порядок эксперимента

1. Подготовить исходный YOLO-датасет:

```text
raw_seedlings/
  images/
  labels/
```

2. Отредактировать `configs/example_experiment.yaml`: пути, seed, модель, пороги,
размер сетки и параметры baseline.

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

6. Запустить простой baseline:

```powershell
py -m seedling_experiments baseline-green --config configs/example_experiment.yaml
```

После этого можно оценить baseline тем же `evaluate-cells`, указав в конфиге путь
к его `predictions.json`. Для baseline допустимо использовать известные bbox
контейнеров из разметки, чтобы честно сравнить именно метод поиска сеянцев.

## Быстрые команды без полного конфига

```powershell
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
