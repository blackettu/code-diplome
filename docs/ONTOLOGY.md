# Онтология v0.1

Файл `configs/ontology/ontology_v0_1.yaml` фиксирует стартовые классы объектов,
состояния ячеек, метки действий, метки безопасности и базовые атрибуты.

Классы объектов:

```text
0 = container
1 = crop_seedling
2 = weed
3 = unknown_plant
```

Для совместимости с текущим набором данных у `crop_seedling` есть псевдоним
`seedlings`. Это позволяет проверять существующие `data.yaml`, где второй класс
пока называется `seedlings`, без немедленной миграции всех артефактов.

Проверка:

```powershell
py -m seedling_data validate-ontology --ontology configs/ontology/ontology_v0_1.yaml --class-names container,seedlings
```

Разметка ячеек и точек действия проверяется тем же файлом онтологии:

```powershell
py -m seedling_data audit-cell-annotations --path data/zks_v0_1/manifests/cell_annotations.jsonl --ontology configs/ontology/ontology_v0_1.yaml --grid-rows 11 --grid-cols 11
py -m seedling_data audit-action-points --path data/zks_v0_1/manifests/action_points.jsonl --ontology configs/ontology/ontology_v0_1.yaml --cell-annotations data/zks_v0_1/manifests/cell_annotations.jsonl --min-safe-distance-px 12 --max-uncertainty-px 4
```

Важное ограничение: класс `weed` нельзя использовать в обучении как доказанный
класс без отдельной экспертной разметки. Для неопределённых растений
используется `unknown_plant`, а действия по таким ячейкам должны уходить на
проверку оператором.

Контракт валидатора для `ontology_v0_1` достаточно строг для отчётов и проверок
готовности: идентификаторы классов 0..3 должны оставаться `container`,
`crop_seedling`, `weed` и `unknown_plant`; обязательные состояния ячеек, метки
действий, метки безопасности и базовые атрибуты должны присутствовать;
нормализованные имена классов и псевдонимы не должны конфликтовать. Проверка
имён классов набора данных также отклоняет пустые и повторяющиеся имена.

`CellStateBuilder` различает `weed_only` и `crop_and_weed`. Генерация целей
создаёт `remove_weed` только для явного объекта класса `weed`; ячейки
`crop_and_weed` помечаются как требующие проверки, чтобы действие рядом с
целевым сеянцем не стало полностью автоматическим.
