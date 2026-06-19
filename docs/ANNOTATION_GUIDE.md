# Руководство по разметке v0.1

## Метки объектов

Источник истины для классов: `configs/ontology/ontology_v0_1.yaml`.

```text
0 = container
1 = crop_seedling
2 = weed
3 = unknown_plant
```

Неуверенные растения размечайте как `unknown_plant`; не переводите их молча в
целевые сеянцы или сорняки. Метка `weed` требует экспертного подтверждения.

## Ограничивающие рамки

- Размечайте один `container` вокруг видимой рабочей области кассеты.
- Рамка растения должна охватывать видимую массу растения, а не всю ячейку.
- Не включайте соседние ячейки в рамку растения.
- Для корректных изображений без объектов используйте пустые файлы разметки.
- Не добавляйте аугментированные изображения в папки исходной разметки.

## Разметка ячеек

`cell_annotations.jsonl` хранит один JSON-объект на пару изображение/ячейка:

```json
{"image_id":"tray001","cell_id":"r00_c00","row":0,"col":0,"state":"single_crop","object_ids":["obj_001"],"keep_object_id":"obj_001","human_review_required":false,"annotator_id":"expert_a"}
```

Допустимые значения `state` берутся из `cell_states` онтологии: `empty`,
`single_crop`, `multiple_crop`, `weed_only`, `crop_and_weed`, `unknown`,
`ambiguous` и `image_quality_insufficient`.

Проверка:

```powershell
py -m seedling_data audit-cell-annotations --path data/zks_v0_1/manifests/cell_annotations.jsonl --ontology configs/ontology/ontology_v0_1.yaml --grid-rows 11 --grid-cols 11
```

## Точки действия

`action_points.jsonl` хранит экспертные решения, используемые для оценки целей,
симуляции и наград RL:

```json
{"image_id":"tray001","cell_id":"r00_c00","object_id":"obj_002","target_type":"remove_extra_crop","action_point_px":[120.0,85.0],"point_type":"bbox_center","uncertainty_radius_px":4.0,"min_distance_to_keep_px":12.0,"human_review_required":false,"decision_rule":"expert","annotator_id":"expert_a"}
```

Устанавливайте `human_review_required: true`, если цель неоднозначна, слишком
близко к сохраняемому сеянцу, находится вне откалиброванной области или зависит
от нерешённого биологического вопроса. Точки действия в ячейках
`crop_and_weed` также должны иметь `human_review_required: true`, даже если
целевой объект явно размечен как `weed`.

Цели, пересекающие запрещённую зону, должны содержать `forbidden_zone_ids` и
`human_review_required: true`. Цели рядом с сохраняемым сеянцем и цели с высокой
неопределённостью также нужно отправлять на проверку, а не считать напрямую
исполняемыми.

Проверка:

```powershell
py -m seedling_data audit-action-points --path data/zks_v0_1/manifests/action_points.jsonl --ontology configs/ontology/ontology_v0_1.yaml --cell-annotations data/zks_v0_1/manifests/cell_annotations.jsonl --min-safe-distance-px 12 --max-uncertainty-px 4
```

## Проверки дублей

Запускайте проверку дублей по `image_manifest.csv` после разбиения и до
подготовки отчётов:

```powershell
py -m seedling_data near-duplicates --manifest data/zks_v0_1/manifests/image_manifest.csv --out reports/near_duplicates.json
```

Если используется `--out`, `seedling_data` пишет `artifact_registry.json` и
`artifact_registry.csv` рядом со сформированным отчётом. Храните реестр вместе с
отчётом о дублях для воспроизводимости.

Отчёт получает `ok=false`, если между выборками есть точные или перцептивные
дубли, а также если строки манифеста ссылаются на отсутствующий `file_path`.
Для отсутствующих файлов невозможно проверить перцептивные хэши.
