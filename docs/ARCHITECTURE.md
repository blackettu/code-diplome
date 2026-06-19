# Архитектурный каркас

Этот слой закрывает первые архитектурные группы задач из
`docs/seedlings_vilga_architecture_codex_rl_simulation.md`:

```text
A-001..A-008
B-001..B-014
C-001..C-010
D-001..D-012
E-001..E-008
F-001..F-011
G-001..G-012
H-001..H-013
I-001/I-002
I-003..I-009 (сухое выполнение ActionPlan и программный HIL-каркас; реальная проверка портала не входит в репозиторий)
J-001..J-006
J-007
K-001..K-008
M-001..M-010
L-001..L-011 через unit/integration/CLI/performance проверки
```

## Принцип

Существующий `seedling_experiments` остаётся исследовательским ядром и
совместимым CLI. Новые продуктовые контракты вынесены в отдельные пакеты:

```text
seedling_core        общие схемы, проверка конфигураций, реестр компонентов и снимки запусков
seedling_data        онтология, манифесты, реестр наборов данных и аудиты разметки
seedling_vision      API адаптеров детекторов
seedling_cells       сетка, состояния ячеек, генерация целей
seedling_calibration преобразования координат изображение/кассета/робот
seedling_decision    интерфейс политик, базовые политики, планирование маршрута, маски действий, SafetyGate
seedling_sim         логический симулятор, шум детекции, ошибка исполнительного механизма
seedling_robot       интерфейс RobotAdapter, симулятор, сухой прогон плана и HIL-адаптер указателя
seedling_rl          RL-среда только для симуляции и оценка базовых политик
seedling_reports     реестр артефактов, CSV-таблицы, проверки готовности, доказательства быстрой проверки и HTML-отчёт эксперимента
seedling_ui          статические офлайн- и replay-просмотрщики
```

## Поток данных

```text
DetectorAdapter
  -> DetectionResultV1 / DetectionObject
  -> CellStateBuilder
  -> ActionTarget generation
  -> DecisionPolicy
  -> SafetyGate
  -> RobotAdapter
  -> logical simulator / dry-run adapter / gated HIL pointer adapter
```

Генерация `ActionTarget` разделяет действия по лишним целевым сеянцам и
сорнякам: ячейки `multiple_crop` создают цели `remove_extra_crop`, ячейки
`weed_only` создают `remove_weed`, а цели `crop_and_weed` требуют проверки.

Слой зрения нормализует выход детектора как `DetectionResultV1`. Аннотация
неопределённости добавочная: флаги уверенности, размера и края изображения
доступны всегда, а нормализованная энтропия класса вычисляется только тогда,
когда адаптеры отдают значения `class_score_*`. ONNX-адаптер сохраняет выходы
оценок классов в таком формате, чтобы правила проверки и безопасности могли
использовать их без знания среды выполнения модели.

`CellStateBuilder` переносит неопределённые атрибуты детекции, например
`low_confidence`, `high_class_entropy`, флаги размера `bbox` и флаги края
изображения, в `risk_flags` ячейки. Такие ячейки требуют проверки человеком, а
сформированные цели наследуют это требование.

RL и реальные адаптеры/адаптеры сухого прогона намеренно не получают доступа во время выполнения к
опасным действиям. Адаптер симулятора выполняет команды только после
`SafetyGate`; реальные адаптеры должны сохранять тот же барьер и оставаться без
опасных `tool profiles` до отдельного анализа безопасности.

HIL-адаптер сейчас покрывает только последовательные проверки указателя за
валидатором `HardwareInLoopReview` и явным флагом `allow_hardware=True`. Он не
подтверждает реальный прогон портала и отклоняет профили реального действия.

## Совместимость

Старые команды остаются прежними:

```powershell
py -m seedling_experiments prepare --config configs/example_experiment.yaml
py -m seedling_experiments evaluate --config configs/example_experiment.yaml
py -m seedling_experiments evaluate-cells --config configs/example_experiment.yaml
```

Для симулятора и RL также доступны унифицированные обёртки быстрой проверки:

```powershell
py -m seedling_experiments sim --config configs/simulation/tray_env_v0.yaml --out reports/unified_sim_smoke.json
py -m seedling_experiments rl --config configs/rl/maskable_ppo_v0.yaml --dry-run --out reports/unified_rl_dry_run.json
```

Эти обёртки пишут `artifact_registry.json` / `artifact_registry.csv` рядом со
сводкой и отдельно помечают входы конфигурации и выходные сводки.

Новая точка входа после `pip install -e .`:

```powershell
seedling-experiments --help
```

`seedling_experiments.grid` теперь является совместимым фасадом над
`seedling_cells.grid`. Для `TrayState.bbox_xyxy_px` используется совместимое
равномерное деление `bbox`; для `TrayState.corners_px` `CellStateBuilder`
строит полигональные ячейки и назначает объекты через алгоритм point-in-polygon, поэтому
объекты внутри внешнего `bbox`, но вне реальной трапеции кассеты, не попадают в
ячейки.

## Реестры и селекторы

Реестры компонентов — это YAML/JSON-файлы, загружаемые через
`seedling_core.registry.ComponentRegistry`. Текущие реестры по умолчанию:

```text
configs/registry/models_v0_1.yaml
configs/registry/policies_v0_1.yaml
configs/registry/model_registry_v0_1.yaml
```

Команда `seedling_ui selector` даёт небольшую быструю проверку и точку для UI
оператора при выборе детектора или политики по `kind`, `name`, `entry_id`,
`status` и тегам. `seedling_ui selector --validate` проверяет весь реестр
компонентов: повторяющиеся селекторы, поддерживаемые значения kind/status,
импортируемые backend-пути, метаданные выходной схемы детектора и метаданные
безопасности политики.

Компоненты RL-политик должны оставаться в области `offline`/`simulation`/`dry-run`,
должны объявлять использование маски действий и не должны иметь прямого доступа
к оборудованию.

`seedling_core.registry.ModelRegistry` — типизированный реестр моделей,
требуемый архитектурным документом. Он хранит `model_id`, `model_type`,
framework, URI артефакта/конфигурации/метрик, версии набора данных и онтологии,
требования калибровки, входные/выходные схемы, статус и уровень безопасности.

Команда `seedling_ui model-registry --validate` отклоняет неподдерживаемые
типы моделей, статусы и уровни безопасности, а также блокирует
`production_candidate`, пока не будет добавлен отдельный поток анализа
безопасности. Для записей `rl_policy` проверка также ограничивает уровень
безопасности значениями `offline_only`, `simulation_only` или `dry_run`,
требует `metadata.direct_hardware_access: false` и
`metadata.requires_action_mask: true`.
