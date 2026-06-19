# Отчёты и реестры

`seedling_reports` строит таблицы воспроизводимости из существующих артефактов
запусков. Документ описывает, какие команды использовать, какие файлы
создаются и какие ограничения нельзя обходить при подготовке статьи или отчёта
готовности.

## Сводки наборов данных

Сводки реестра наборов данных создаются через `seedling_data`:

```powershell
py -m seedling_data registry summarize --dataset zks_v0_1 --out reports/dataset_summary.md
py -m seedling_data registry summarize --dataset zks_v0_1 --out reports/dataset_summary.html
py -m seedling_data audit-post-action --path runs/followup/post_action_observations.jsonl --required-hours 24,48,72
py -m seedling_data summarize-post-action --path runs/followup/post_action_observations.jsonl --out runs/followup/post_action_summary.json
```

Markdown-сводка включает зарегистрированные хэши, числа изображений по выборкам,
распределения идентификаторов классов YOLO, распределения состояний ячеек из
`cell_annotations.jsonl` и распределения типов целей из `action_points.jsonl`,
если такие артефакты присутствуют.

## Основной интерфейс командной строки

```powershell
py -m seedling_reports build --root runs --out reports/experiment_summary --model-registry configs/registry/model_registry_v0_1.yaml
py -m seedling_reports compare-runs --runs reports/experiment_a reports/experiment_b --out reports/compare_ab
py -m seedling_reports compare-runs --runs reports/experiment_a reports/experiment_b --out reports/compare_ab --min-cell-accuracy 0.9 --min-edge-cell-accuracy 0.85 --min-corner-cell-accuracy 0.8
py -m seedling_reports compare-runs --runs reports/experiment_a reports/experiment_b --out reports/compare_ab --min-target-recall 0.8 --max-rl-critical-error-rate 0.01 --min-rl-seed-count 3
py -m seedling_reports compare-runs --runs reports/experiment_a reports/experiment_b --out reports/compare_ab --max-rl-critical-events-count 0 --max-rl-block-rate 0.05 --max-rl-crop-damage-count 0
py -m seedling_reports compare-runs --runs reports/experiment_a reports/experiment_b --out reports/compare_ab --min-hardware-move-success-rate 0.95 --max-safety-unsafe-action-escape-count 0 --max-safety-real-action-guard-blocks 0 --max-safety-unsupported-tool-profile-blocks 0 --max-safety-aborted-runs 0 --max-safety-skipped-commands 0
py -m seedling_reports compare-runs --runs reports/experiment_a reports/experiment_b --out reports/compare_ab --min-post-action-coverage-rate 0.95 --max-post-action-crop-damage-rate 0
py -m seedling_reports compare-runs --runs reports/experiment_a reports/experiment_b --out reports/compare_ab --max-error-budget-total-mm 3.0 --require-error-budget-complete
py -m seedling_reports compare-runs --runs reports/experiment_a reports/experiment_b --out reports/compare_ab --require-run-snapshots --max-run-snapshot-missing-config-hash 0 --max-run-snapshot-missing-command-args 0
py -m seedling_reports scenario-compare --scene runs/predict/scenes.json --policies raster_scan route_planning risk_aware_rule rl_no_model --out reports/scenario_compare
py -m seedling_reports export-plan --scene runs/predict/scenes.json --policy route_planning --out reports/action_plan.json
py -m seedling_reports diagnose-cell-metrics --metrics runs/evaluation/cell_metrics.json --out reports/cell_diagnostics
py -m seedling_reports architecture-audit --doc docs/seedlings_vilga_architecture_codex_rl_simulation.md --out reports/architecture_backlog --repo-root .
py -m seedling_reports software-readiness-smoke --out reports/software_readiness_smoke --repo-root .
py -m seedling_reports readiness-check --root reports/experiment_summary --out reports/readiness --repo-root .
```

## Выходные файлы

```text
artifact_registry.csv
model_registry.csv
dataset_summary.csv
object_level_results.csv
task_level_results.csv
run_snapshot_results.csv
rl_results.csv
hardware_dry_run_results.csv
safety_results.csv
post_action_results.csv
error_budget_results.csv
annotation_feedback_results.csv
readiness_report.json / .csv / .html
architecture_backlog_audit.json / .csv / .html
software_readiness_smoke.json
experiment_report.html
report_summary.json
```

`dataset_summary.csv` строится из `dataset_audit.json` и, если он есть,
`raw_dataset_audit.json`. Колонка `audit_stage` разделяет строки `raw` и
`prepared`, чтобы таблицы статьи не смешивали аудит исходного набора данных и
аудит после разбиения/подготовки.

`model_registry.csv` строится из типизированного реестра моделей
`configs/registry/model_registry_v0_1.yaml` или пути `--model-registry`. Таблица
сохраняет идентификатор и тип модели, framework, URI артефакта/конфигурации/
метрик, версии набора данных и онтологии, схемы входа/выхода, статус, уровень
безопасности, теги и требования калибровки.

`object_level_results.csv` собирает метрики детектора. `task_level_results.csv`
сохраняет метрики ячеек и целей, ошибку координат в пикселях и, если
`evaluate-cells` запускался с `evaluation.calibration`, ошибку в миллиметрах
кассеты через `mean_coordinate_error_mm`.

Для старых запусков YOLO и schema-first оценок `SceneState` таблица также
содержит поля с ценой ошибок: `cost_total`, `critical_error_total`,
`critical_error_rate_per_cell` и сериализованные `critical_error_counts`.
Schema-first строки дополнительно содержат `schema_mode`, `gt_scene_id`,
`pred_scene_id`, метрики краевых/угловых ячеек и precision/recall экспертного
keep/remove.

`rl_results.csv` строится из `rl_eval_metrics.json`, `unified_rl_baselines.json`
и `offline_replay_eval.json`. В нём фиксируются политика, число эпизодов и
действий, средняя награда, доля критических ошибок, доля успешных целей, доля
проверок, действия на кассету, суммарное перемещение, средняя ошибка расстояния,
число критических событий, разрешённые/заблокированные действия, доля
блокировок и число повреждений целевых сеянцев.

`run_snapshot_results.csv` строится из `run_snapshot.json` и
`*.run_snapshot.json`. Он сохраняет имя команды, канонический `config_hash`,
наличие сохранённой конфигурации и аргументов команды, Python/платформу и
сериализованные версии пакетов. `compare-runs` может требовать полноту этих
полей через `--max-run-snapshot-missing-config-hash 0` и
`--max-run-snapshot-missing-command-args 0`.

`hardware_dry_run_results.csv` строится из `dry_run_control_points.json`,
`dry_run_plan_report.json` и `hil_pointer_report.json`. Таблица включает
успешность homing и движений, p50/p95/p99 ошибки позиционирования при наличии
контрольных точек, повторяемость, статус проверки, блокировки безопасности и
ссылки на журналы команд/воспроизведения. Аппаратные счётчики потерянных шагов,
событий концевиков, дрейфа калибровки и потерь телеметрии остаются пустыми,
пока контроллер портала их не отдаёт.

`safety_results.csv` строится из `offline_replay_eval.json`,
`critical_events.json`, отчётов сухого плана, HIL-отчётов и журналов воспроизведения.
Он сводит небезопасные попытки, блокировки `SafetyGate`, обнаруженные уходы
блокировок с повреждением сеянцев, отказы межблокировки и калибровки, блоки
реального действия, неподдерживаемые профили инструмента, прерванные HIL-запуски,
пропущенные команды, требования проверки и нарушения запрещённых зон.

`post_action_results.csv` строится из `post_action_summary.json` или сырого
`post_action_observations.jsonl`. JSONL хранит отложенные наблюдения по
`command_id`/`target_id` с `hours_after_action`, `outcome`, `confidence`,
`image_ref` и метаданными наблюдателя. Стандартные окна проверки: 24/48/72 часа.

`error_budget_results.csv` строится из `error_budget.json` и сохраняет RSS
общей ошибки, полноту, статус порога, отсутствующие компоненты и компоненты
архитектурного бюджета `e_detection`, `e_grid`, `e_calibration`,
`e_mechanics`, `e_focus`, `e_latency`, `e_biological_target`.

`annotation_feedback_results.csv` строится из `feedback.jsonl` и
`annotation_tasks.jsonl`. Он сохраняет связанные идентификаторы изображения,
цели, объекта и ячейки, тип причины/ошибки, приоритет задачи, статус,
идентификатор оператора, комментарии и структурированный JSON
`proposed_correction` для передачи на переразметку.

## Сравнение запусков и сценариев

`compare-runs` читает объектные, задачные, RL, аппаратные и безопасностные
метрики из нескольких папок отчётов. Если CSV-таблицы ещё не собраны, команда
также принимает прямые артефакты `rl_eval_metrics.json`,
`unified_rl_baselines.json`, `offline_replay_eval.json`,
`dry_run_control_points.json`, `dry_run_plan_report.json`,
`hil_pointer_report.json`, `run_snapshot.json` и `error_budget.json`.

Если найден `sweep_stability_summary.json`, сравнение добавляет число случайных зёрен и
колонки стабильности RL для группы с лучшей наградой. Пороговые флаги добавляют
`threshold_status` и `threshold_failures` для критериев статьи: accuracy всей
кассеты, краевых и угловых ячеек, recall целей, критическая ошибка RL, число
случайные зёрна, пределы критических событий/блокировок/повреждений, успешность сухого
прогона/HIL, p95 ошибки позиционирования, уходы небезопасных действий, блоки
реального действия, неподдерживаемые профили инструмента, прерванные HIL,
пропущенные команды, покрытие/повреждения проверки после действия, полный бюджет ошибок и
полнота снимков запуска.

Выходы:

```text
compare_runs.csv
compare_runs.html
compare_runs_summary.json
```

`scenario-compare` запускает детерминированные политики на одном `SceneState`.
Также поддерживаются `rl_no_model` / `rl_policy_adapter` для безопасной
строки быстрой проверки RL-адаптера и `rl_checkpoint:<path>` для установленной контрольной точки
SB3/SB3-Contrib. Выходы:

```text
scenario_compare.json
scenario_compare.html
```

Строки содержат число команд, число целей на проверку/блокировку, порядок целей,
`rl_action_kind` для строк RL-адаптера и `route_distance_mm`, если политика
записывает маршрутные метаданные.

`export-plan` пишет сериализуемую обёртку `ActionPlan`:

```text
action_plan.json
```

Этот файл используется как `--plan` для сухого прогона робота и HIL-указателя.

## Диагностика метрик ячеек

`diagnose-cell-metrics` пересэмплирует строки уровня изображения/группы из
`cell_metrics.json` и пишет:

```text
bootstrap_ci.json
error_taxonomy.json
```

Единица bootstrap задаётся `--group-key`; по умолчанию пересэмплируются
изображения, а не отдельные ячейки. `error_taxonomy.json` хранит числа и
примеры по категориям, а также `group_counts` / `category_groups`, чтобы отказы
можно было смотреть по слоям: `image`, `geometry`, `biology`, `decision`.

Для schema-first файлов `cell_metrics.json` с `scene_state_metrics` таксономия
использует идентификаторы ячеек, идентификаторы целей, богатую матрицу ошибок
состояний ячеек и ошибки координат сопоставленных целей из блока схемы, а не
дублирует совместимую сводку из `images[]`.

## Готовность и аудит архитектуры

`software-readiness-smoke` создаёт воспроизводимый набор программных
доказательств: артефакты калибровки и карты ошибок, сравнение политик на
сценарии, безопасный отчёт сухого выполнения `ActionPlan`, CSV-сводки
hardware/safety, вложенный отчёт готовности и вложенный аудит
`architecture_backlog/` по всем backlog ID.

`architecture-audit` читает все backlog ID из
`seedlings_vilga_architecture_codex_rl_simulation.md` и пишет JSON/CSV/HTML
таблицу доказательств с путями репозитория для каждого пункта. Программный HIL
каркас отделён от реальной проверки портала: `I-009` помечается как
`external_required`.

Для рискованных пунктов аудит проверяет не только наличие путей, но и
содержательные маркеры. Проверяются точки входа пакета, разделение
зависимостей, валидация конфигурации, round-trip схем, миграция предсказаний,
CI, снимки запусков с хэшами конфигураций и аргументами команд, ролевой реестр
артефактов, онтология v0.1, манифесты изображений, проверки дублей, паспорт и
руководство разметки, групповые разбиения, аудиты разметки, реестр наборов
данных, журнал изменений, интерфейсы детекторов, контракты Ultralytics/HSV/записанных/
ONNX, неопределённость, постобработка контейнеров, типизированные метаданные
моделей, наложения, пакетный вывод, построение `CellState`, генерация целей,
маски действий, метрики ячеек и `SceneState`, калибровка, политики решений,
симуляция, RL, `SafetyGate`, телеметрия робота, адаптеры робота, UI/отчёты,
блокировки запретных зон, адаптер сухого прогона, сравнение сценариев, пороги
`compare-runs`, матрица тестов и разделы документации.

`readiness-check` сопоставляет архитектурные критерии готовности с явными
доказательствами для выбранной папки запуска/отчёта. Он смотрит файлы
репозитория через `--repo-root` и артефакты в `--root`: отчёты, журналы команд,
журналы воспроизведения и выходы калибровки. Пункты детекторов и политик дополнительно запускают
валидацию реестров компонентов, включая импортируемость backend, метаданные
выходной схемы детектора и safety-метаданные политик/RL.

Готовность калибровки требует хотя бы один `calibration.json`, который проходит
`CalibrationValidator` с `p95 <= 2.0`, без истечения или warning-проблем, и
имеет соответствующий `error_map` под корнем запуска. Все команды пишут
локальные реестры.

Физическая проверка портала и отложенная биологическая проверка после действия
остаются `external_required`, пока не появятся реальные доказательства. Короткие
заглушки вроде `review: {"ok": true}` намеренно не считаются достаточными для
заявления продуктовой готовности.

## Реестры и снимки запусков

Существующие команды `seedling_experiments` теперь пишут `artifact_registry.json`
и `artifact_registry.csv` рядом с `config.yaml` и `run_snapshot.json`. Лёгкие
обёртки `split`, `audit`, `check-split`, `evaluate-scenes`, `sim` и `rl` также
пишут ролевые локальные реестры рядом со своими выходами. Ошибки записи реестра
не фатальны, чтобы сохранить совместимость старого CLI.

Для команд с конфигурацией (`prepare`, `train`, `val`, `predict`,
`evaluate-cells`, `baseline-green`) исходный путь `--config` записывается как
`input`, если он доступен. Специфичные входы вроде весов модели, `data.yaml`,
папок изображений, файлов предсказаний, артефактов калибровки и файлов
`SceneState` также записываются. Сохранённые `config.yaml`, `run_snapshot.json`
и основные выходы (`predictions.json`, `cell_metrics.json`, матрицы ошибок,
метрики валидации и манифесты аудита/разбиения из `prepare`) записываются как
`output`.

Команды `seedling_reports`, `seedling_calibration`, `seedling_data`,
`seedling_vision`, `seedling_sim`, `seedling_rl`, `seedling_robot` и
`seedling_ui`, которые пишут файлы, используют тот же подход: создают локальные
`artifact_registry.json`, `artifact_registry.csv` и `*.run_snapshot.json`,
разделяя входы, выходы и снимки команды.

`seedling_core.run_snapshot` предоставляет общий сервис снимков. Старое имя
`seedling_experiments.config.save_run_snapshot` сохранено как совместимая
обёртка. `run_snapshot.json` содержит команду, сериализованные `command_args`,
канонический SHA-256 `config_hash`, сохранённую конфигурацию и метаданные
окружения. Для команд без YAML-конфигурации `save_command_snapshot` хэширует
канонический объект из аргументов команды, входных путей, выходных путей и
метаданных.
