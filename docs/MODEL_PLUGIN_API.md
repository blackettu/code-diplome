# API подключаемых моделей

Минимальный контракт детектора находится в `seedling_vision.adapters.base`:

```python
class DetectorAdapter:
    def load(self, model_uri: str, device: str | None = None) -> None: ...
    def predict(self, image, context=None) -> DetectionResultV1: ...
    def metadata(self) -> ModelMetadata: ...
```

Реализации первого этапа:

- `RecordedPredictionDetector` читает существующий `predictions.json` и
  возвращает `DetectionResultV1`; используется для офлайн-тестов без GPU.
- `UltralyticsYOLODetector` оборачивает Ultralytics YOLO и возвращает те же
  доменные `DetectionObject`.
- `ONNXDetector` оборачивает среду выполнения `ONNX Runtime` и поддерживает явные контракты
  выхода: `[x1,y1,x2,y2,confidence,class_id]`,
  `[cx,cy,w,h,confidence,class_id]`, `[x1,y1,x2,y2,class_scores...]`,
  `[cx,cy,w,h,class_scores...]` и варианты с objectness score. Выходы
  channel-first вида `[1, channels, boxes]` транспонируются перед разбором.
  `onnxruntime` является необязательной vision-зависимостью и импортируется
  только в `load()`.
- `BaselineGreenDetector` — CPU-базовая линия на HSV-пороге и connected
  components. Она нужна для быстрых тестов и простого сравнения без обучения.
- `MockDetector` нужен для модульных и интеграционных тестов.

Все новые детекторы должны возвращать
`seedling_core.schemas.DetectionResultV1`. Внутренний формат конкретной
библиотеки не должен протекать в слои принятия решений, симуляции или робота.
Если передан `InferenceContext.ontology_version`, адаптеры копируют его в
`model_metadata.ontology_version` результата без изменения постоянных метаданных
детектора.

## Реестр моделей и политик

Первые реестры находятся в:

```text
configs/registry/models_v0_1.yaml
configs/registry/policies_v0_1.yaml
```

Общий формат: `entry_id`, `kind`, `name`, `backend`, необязательный `uri`,
`status`, `tags` и `metadata`. CLI selector проверяет, что конкретный
detector/policy можно однозначно выбрать:

```powershell
py -m seedling_ui selector --registry configs/registry/models_v0_1.yaml --kind detector --name baseline_green
py -m seedling_ui selector --registry configs/registry/policies_v0_1.yaml --kind policy --name risk_aware_rule
py -m seedling_ui selector --registry configs/registry/policies_v0_1.yaml --kind policy --name route_planning
py -m seedling_ui selector --registry configs/registry/models_v0_1.yaml --validate
py -m seedling_ui selector --registry configs/registry/policies_v0_1.yaml --validate
```

`selector --validate` проверяет весь реестр компонентов: повторяющиеся
селекторы, поддерживаемые значения kind/status, импортируемые пути `backend`,
метаданные выходной схемы детекторов и метаданные безопасности политик.
Компоненты RL-политик должны объявлять
`metadata.direct_hardware_access: false`, `metadata.requires_action_mask: true`
и уровень безопасности не выше `dry_run`.

Типизированные записи реестра моделей находятся в:

```text
configs/registry/model_registry_v0_1.yaml
```

Этот реестр следует архитектурному контракту раздела 13.7: `model_id`,
`model_type`, `framework`, `artifact_uri`, `config_uri`, `metrics_uri`,
`dataset_version`, `ontology_version`, `calibration_requirements`,
`input_schema`, `output_schema`, `status` и `safety_level`. Используйте:

```powershell
py -m seedling_ui model-registry --registry configs/registry/model_registry_v0_1.yaml --validate
py -m seedling_ui model-registry --registry configs/registry/model_registry_v0_1.yaml --model-id baseline_green_detector_v0
```

Валидатор держит `production_candidate` заблокированным, пока нет отдельного
процесса анализа безопасности; RL-записи вроде `maskable_ppo_v0_seed42` и
`recurrent_ppo_v0_seed42` остаются `simulation_only` или ограниченными сухим
прогоном. Валидатор отклоняет RL-записи, которые запрашивают прямой доступ к
оборудованию или не содержат метаданные маски действий.

## Миграция старых предсказаний

```powershell
py -m seedling_vision migrate-predictions --predictions runs/predict/predictions.json --out runs/predict/scenes.json --dataset-version zks_v0_1
```

Команда миграции читает существующий формат `predictions.json`, преобразует
детекции в `DetectionResultV1`, сопоставляет детекции растений с `CellState` и
пишет сериализуемые записи `SceneState` со сгенерированными базовыми целями.

## Пакетная обработка, неопределённость и наложения

```powershell
py -m seedling_vision batch-recorded --predictions runs/predict/predictions.json --images tray001.jpg tray002.jpg --out runs/predict/batch_detection_results.json --uncertainty --progress-log runs/predict/batch_progress.jsonl
py -m seedling_vision postprocess-containers --predictions runs/predict/predictions.json --out runs/predict/postprocessed_containers.json --min-area 10000 --merge-distance 50
py -m seedling_vision overlay --predictions runs/predict/predictions.json --image data/zks_v0_1/images/tray001.jpg --out reports/overlays/tray001.jpg --uncertainty
py -m seedling_vision overlay --predictions runs/predict/predictions.json --image data/zks_v0_1/images/tray001.jpg --out reports/overlays/tray001_full.png --include-scene --grid-rows 11 --grid-cols 11 --errors reports/cell_diagnostics/error_taxonomy.json
```

`seedling_vision.uncertainty` добавляет флаги уверенности, размера и края
изображения без изменения кода конкретного детектора. Если детектор отдаёт
числовые `class_score_*` в `DetectionObject.uncertainty`, оценщик также пишет
`class_entropy_normalized`, `class_probability_margin` и флаг
`high_class_entropy`. ONNX-контракты выходов с оценками классов сохраняют эти
сырые оценки, а контракты с objectness дополнительно сохраняют `objectness`.

Построение состояния ячеек переносит неопределённые атрибуты детекции
(`low_confidence`, `high_class_entropy`, флаги размера `bbox` и края
изображения) в флаги риска ячейки, поэтому затронутые ячейки и сгенерированные
цели требуют проверки человеком.

`seedling_vision.overlays` пишет статический отчёт-изображение с `bbox`,
подписями уверенности, необязательными ячейками сетки, необязательными целями
действий и необязательными маркерами ошибок. Режим
`overlay --include-scene` строит ячейки и базовые цели из старого
`predictions.json`; `--scene` может вместо этого использовать существующий
JSON/JSONL `SceneState`.

`--errors` принимает списки JSON, `{"errors": [...]}`,
`{"critical_events": [...]}` или образцы `error_taxonomy.json` и рисует записи,
которые содержат точку, `bbox`, идентификатор цели, ячейки или объекта.
Псевдонимы целей/ячеек/объектов из диагностики, например `pred_target_id`,
`gt_target_id`, `pred_cell_id`, `gt_cell_id`, `pred_object_id` и
`gt_object_id`, также сопоставляются, если переданы сущности сцены.

`batch-recorded --progress-log` пишет JSONL-события прогресса с индексом
изображения, общим числом и итогом `batch_complete`, сохраняя stdout как один
JSON-объект для автоматизации.

Все команды `seedling_vision`, которые пишут файлы, также создают ролевые
`artifact_registry.json` / `artifact_registry.csv` и `*.run_snapshot.json`
рядом с выходным файлом. Так файлы предсказаний, ключи/пути изображений,
необязательные сцены/ошибки для наложения, журналы прогресса, отчёты и точные
аргументы CLI остаются прослеживаемыми.

`seedling_vision.postprocess` держит фильтрацию/слияние контейнеров и
IoU-диагностику отдельно от адаптеров моделей, поэтому оценщики могут явно
выбирать сырые `detections` или постобработанные `containers`.

## Адаптеры политик и робота

Политики принятия решений находятся в `seedling_decision.policies` и
предоставляют:

```python
class DecisionPolicy:
    def reset(self, scene: SceneState) -> None: ...
    def propose_plan(self, scene: SceneState) -> ActionPlan: ...
    def next_action(self, scene: SceneState) -> ActionCommand | None: ...
```

Встроенные базовые политики: `noop`, `human_review`, `raster_scan`,
`nearest_neighbor`, `route_planning` и `risk_aware_rule`. `route_planning`
использует жадный маршрут ближайшего соседа по текущим допустимым целям и
записывает расстояние отдельного шага и накопленное расстояние маршрута в
метаданные команды.

Адаптеры робота находятся в `seedling_robot.adapters` и предоставляют общий
интерфейс `RobotAdapter`. Реализации должны принимать только команды, которые
уже прошли `SafetyGate`.
