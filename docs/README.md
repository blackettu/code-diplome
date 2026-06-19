# Документация проекта code-diplome

Это входная точка документации по воспроизводимому контуру
`seedling_experiments` и новым пакетам `seedling_*`. Рабочие инструкции,
протоколы и методические заметки должны находиться в папке `docs/`.
Корневой `README.md` оставлен только как короткий указатель на этот файл.

## Карта документов

- `DEVELOPER_GUIDE.md` — практический контракт запуска: формат набора данных,
  конфигурация, артефакты, `predictions.json`, `cell_metrics.json` и диагностика
  типичных ошибок.
- `EXPERIMENT_PROTOCOL.md` — экспериментальный протокол: честное разбиение,
  аугментация только обучающей выборки, базовый метод сравнения, метрики
  конечной задачи и трассировка артефактов.
- `ARTICLE_EXPERIMENT_PLAN.md` — пошаговый план получения результатов для
  статьи.
- `dataset_collection_recommendations.md` — рекомендации по сбору и разметке
  нового набора данных.
- `DATASET_CARD_TEMPLATE.md` — шаблон паспорта набора данных с разбиением,
  калибровкой, ограничениями безопасности и полями реестра.
- `DATASET_CHANGELOG_TEMPLATE.md` — шаблон журнала изменений версии набора
  данных.
- `ANNOTATION_GUIDE.md` — правила ограничивающих рамок, разметки ячеек и точек
  действия.
- `vision_automated_weed_removal.md` — продуктовое и научное видение системы
  автоматизированного удаления сорняков и лишних сеянцев.
- `necessary_corrections.md` — методические правки для статьи и результатов.
- `ARCHITECTURE.md` — краткое описание архитектурного слоя: схемы, онтология,
  API адаптеров, ячейки и цели.
- `MODEL_PLUGIN_API.md` — контракт `DetectorAdapter`, базовые детекторы и
  реестр моделей/политик.
- `ONTOLOGY.md` — описание `configs/ontology/ontology_v0_1.yaml`.
- `CALIBRATION_PROTOCOL.md` — артефакт калибровки, цепочка преобразований и
  валидатор.
- `SAFETY_CONCEPT.md` — `SafetyGate`, решения безопасности и запрет обхода
  через политики.
- `SIMULATION_SPEC.md` — логический симулятор, шум детекции и модель ошибки
  исполнительного механизма.
- `RL_SPEC.md` — совместимая с Gymnasium среда, маска действий и оценка базовых
  политик.
- `ROBOT_PROTOCOL.md` — контракт `RobotAdapter`, симулятор, сухой прогон и HIL
  указателем.
- `REPORTING.md` — `artifact_registry`, CSV-таблицы для статьи и HTML-отчёт.
- `VIEWERS.md` — статические просмотрщики офлайн-предсказаний и журналов воспроизведения
  симуляции.
- `FAILURE_MODES.md` — таксономия отказов и блокирующие условия.
- `OPERATOR_MANUAL_DRAFT.md` — черновой рабочий процесс для офлайн, симуляции и
  сухого прогона.
- `CI.md` — локальные `pre-commit` проверки и рабочий процесс GitHub Actions.
- `PERFORMANCE.md` — лёгкие проверки производительности.
- `ARTICLE_REPORT_TEMPLATE.md` — таблицы для статьи и сопоставление
  артефактов.
- `seedlings_vilga_architecture_codex_rl_simulation.md` — полный backlog
  архитектурной пересборки, симуляции и RL.

## Назначение проекта

Код предназначен для экспериментов по детекции контейнеризированных сеянцев,
построению матрицы заполнения ячеек и оценке целей удаления при множественном
прорастании.

Новые воспроизводимые эксперименты запускаются через пакет
`seedling_experiments` и вспомогательные пакеты `seedling_*`. Старые скрипты в
корне проекта оставлены как прототипы и не должны использоваться для новых
результатов, которые нужно защищать в статье.

## Быстрый старт

Все команды запускаются из корня проекта `code-diplome/`, где лежат
`pyproject.toml`, `requirements/`, `configs/` и пакеты `seedling_*`.

```powershell
py -m pip install -e .
py -m pip install -r requirements/base.txt
py -m pip install -r requirements/dev.txt

# Необязательные наборы зависимостей:
py -m pip install -r requirements/vision.txt
py -m pip install -r requirements/rl.txt
py -m pip install -r requirements/robot.txt
```

После установки доступны точки входа:

```powershell
seedling-experiments --help
seedling-data --help
seedling-vision --help
seedling-calibration --help
seedling-sim --help
seedling-rl --help
seedling-robot --help
seedling-reports --help
seedling-ui --help
```

Команда `seedling-experiments` должна показывать:

```text
prepare
train
val
predict
evaluate
evaluate-cells
baseline-green
split
audit
check-split
evaluate-scenes
sim
rl
```

## Команды быстрой проверки

Проверка реестров моделей и политик:

```powershell
py -m seedling_ui selector --registry configs/registry/models_v0_1.yaml --kind detector --name baseline_green
py -m seedling_ui selector --registry configs/registry/policies_v0_1.yaml --kind policy --name risk_aware_rule
py -m seedling_ui selector --registry configs/registry/policies_v0_1.yaml --kind policy --name route_planning
py -m seedling_ui selector --registry configs/registry/models_v0_1.yaml --validate
py -m seedling_ui selector --registry configs/registry/policies_v0_1.yaml --validate
py -m seedling_ui model-registry --registry configs/registry/model_registry_v0_1.yaml --validate
py -m seedling_ui model-registry --registry configs/registry/model_registry_v0_1.yaml --model-id baseline_green_detector_v0
```

Проверка данных:

```powershell
py -m seedling_data validate-ontology --ontology configs/ontology/ontology_v0_1.yaml --class-names container,seedlings
py -m seedling_data image-manifest --dataset-root data/zks_v0_1 --images-dir images --output data/zks_v0_1/manifests/image_manifest.csv --group-regex "(?P<tray_id>tray[0-9]+)" --session-id session01
py -m seedling_data audit-cell-annotations --path data/zks_v0_1/manifests/cell_annotations.jsonl --ontology configs/ontology/ontology_v0_1.yaml --grid-rows 11 --grid-cols 11
py -m seedling_data audit-action-points --path data/zks_v0_1/manifests/action_points.jsonl --ontology configs/ontology/ontology_v0_1.yaml --cell-annotations data/zks_v0_1/manifests/cell_annotations.jsonl --min-safe-distance-px 12 --max-uncertainty-px 4
py -m seedling_data audit-changelog --dataset-root data/zks_v0_1 --dataset-version zks_v0_1
py -m seedling_data near-duplicates --manifest data/zks_v0_1/manifests/image_manifest.csv --out reports/near_duplicates.json
```

Миграция и отчёты зрения:

```powershell
py -m seedling_vision migrate-predictions --predictions runs/predict/predictions.json --out runs/predict/scenes.json --dataset-version zks_v0_1
py -m seedling_vision batch-recorded --predictions runs/predict/predictions.json --images tray001.jpg --out runs/predict/batch_detection_results.json --uncertainty --progress-log runs/predict/batch_progress.jsonl
py -m seedling_vision postprocess-containers --predictions runs/predict/predictions.json --out runs/predict/postprocessed_containers.json --min-area 10000 --merge-distance 50
py -m seedling_vision overlay --predictions runs/predict/predictions.json --image data/zks_v0_1/images/tray001.jpg --out reports/overlays/tray001.jpg --uncertainty
```

Калибровка:

```powershell
py -m seedling_calibration estimate --config data/calibration/session01/calibration_config.yaml --out data/calibration/session01/calibration.json --calibration-id session01
py -m seedling_calibration validate --calibration data/calibration/session01/calibration.json --tray-type 11x11 --max-p95-mm 2.0
py -m seedling_calibration error-map --config data/calibration/session01/calibration_config.yaml --calibration data/calibration/session01/calibration.json --out data/calibration/session01/error_map.json
py -m seedling_calibration error-budget --components configs/calibration/error_budget.example.json --calibration data/calibration/session01/calibration.json --out data/calibration/session01/error_budget.json --max-total-mm 3.0
```

Отчёты:

```powershell
py -m seedling_reports build --root runs --out reports/experiment_summary --model-registry configs/registry/model_registry_v0_1.yaml
py -m seedling_reports compare-runs --runs reports/experiment_a reports/experiment_b --out reports/compare_ab --max-error-budget-total-mm 3.0 --require-error-budget-complete --require-run-snapshots --max-run-snapshot-missing-config-hash 0 --max-run-snapshot-missing-command-args 0
py -m seedling_reports scenario-compare --scene runs/predict/scenes.json --policies raster_scan route_planning risk_aware_rule rl_no_model --out reports/scenario_compare
py -m seedling_reports export-plan --scene runs/predict/scenes.json --policy route_planning --out reports/action_plan.json
py -m seedling_reports diagnose-cell-metrics --metrics runs/evaluation/cell_metrics.json --out reports/cell_diagnostics
py -m seedling_reports architecture-audit --doc docs/seedlings_vilga_architecture_codex_rl_simulation.md --out reports/architecture_backlog --repo-root .
py -m seedling_reports software-readiness-smoke --out reports/software_readiness_smoke --repo-root .
py -m seedling_reports readiness-check --root reports/experiment_summary --out reports/readiness --repo-root .
```

Команда `seedling_reports build` создаёт таблицы статьи и реестры артефактов.
Команды, которые пишут файлы, создают `artifact_registry.json`,
`artifact_registry.csv` и `*.run_snapshot.json`, чтобы входы, выходы,
аргументы команд и хэши конфигураций оставались прослеживаемыми.

Симуляция и RL:

```powershell
py -m seedling_sim run-policy --scene data/sim_scenes/v0/scene_000000.json --policy route_planning --out runs/sim/replay_route_scene000000.json
py -m seedling_sim compare-policies --scenes data/sim_scenes/v0 --policies raster_scan route_planning risk_aware_rule --out reports/policy_comparison.html
py -m seedling_experiments sim --config configs/simulation/tray_env_v0.yaml --out reports/unified_sim_smoke.json
py -m seedling_rl check-env --config configs/simulation/tray_env_v0.yaml
py -m seedling_rl train --config configs/rl/maskable_ppo_v0.yaml --dry-run
py -m seedling_rl evaluate --config configs/simulation/tray_env_v0.yaml --baseline raster_scan --episodes 10
py -m seedling_rl offline-replay-eval --replays runs/sim/replay.json --out runs/rl/offline_replay_eval.json
py -m seedling_experiments rl --config configs/rl/maskable_ppo_v0.yaml --dry-run --out reports/unified_rl_dry_run.json
```

Робот и сухой прогон:

```powershell
py -m seedling_robot dry-run-test --points data/calibration/session01/control_points.json --out reports/dry_run_control_points.json --command-log reports/dry_run_commands.json
py -m seedling_robot validate-hil-review --review configs/robot/hil_review_template.json --out reports/hil_review_validation.json --hil-pointer
py -m seedling_reports export-plan --scene runs/predict/scenes.json --policy route_planning --out reports/action_plan.json
py -m seedling_robot dry-run-plan --scene runs/predict/scenes.json --plan reports/action_plan.json --out reports/dry_run_plan_report.json --command-log reports/dry_run_plan_commands.json --replay reports/dry_run_plan_replay.json --interlock-ok
py -m seedling_robot hil-pointer-run --scene runs/predict/scenes.json --plan reports/action_plan.json --review configs/robot/hil_review_template.json --port COM5 --out reports/hil_pointer_report.json --command-log reports/hil_pointer_commands.json --replay reports/hil_pointer_replay.json --allow-hardware --interlock-ok
```

`readiness-check` оставляет физический портал и отложенную биологическую
проверку как `external_required`, пока в корне артефактов нет реального
доказательства: успешного HIL-прогона указателем с журналом команд/воспроизведения и
полной 24/48/72-часовой проверки после действия с наблюдателем.

## Входной набор данных

Исходный набор данных должен быть в формате YOLO:

```text
raw_seedlings/
  images/
  labels/
```

Для текущих конфигураций ожидается порядок классов:

```text
0 = container
1 = seedlings
```

Перед `prepare` желательно сохранить аудит исходного набора:

```powershell
py -m seedling_experiments audit --dataset E:/dataset/raw_seedlings --output E:/dataset/raw_seedlings/raw_dataset_audit.json
```

Это фиксирует отсутствующие и лишние файлы разметки до подготовки. После
`prepare` отсутствующие `labels` превращаются в пустые `.txt` в
`prepared_root`.

## Основной порядок эксперимента

1. Отредактировать `configs/example_experiment.yaml`: пути, случайное зерно, модель,
   пороги, размер сетки и параметры базового метода сравнения.
2. Выполнить разбиение до аугментации, применить аугментацию только к `train` и
   получить аудит набора данных:

```powershell
py -m seedling_experiments prepare --config configs/example_experiment.yaml
```

Основные артефакты `prepare`:

```text
prepared_root/config.yaml
prepared_root/run_snapshot.json
prepared_root/raw_dataset_audit.json
prepared_root/data.yaml
prepared_root/split_summary.json
prepared_root/split_manifest.csv
prepared_root/train_augmentation_manifest.csv
prepared_root/split_integrity_report.json
prepared_root/dataset_audit.json
prepared_root/prepare_summary.json
```

3. Обучить YOLO:

```powershell
py -m seedling_experiments train --config configs/example_experiment.yaml
```

4. Проверить модель на независимом `test`:

```powershell
py -m seedling_experiments val --config configs/example_experiment.yaml --split test
py -m seedling_experiments predict --config configs/example_experiment.yaml
py -m seedling_experiments evaluate-cells --config configs/example_experiment.yaml
```

Ключевой файл для статьи: `cell_metrics.json`. Он содержит accuracy ячеек,
макро-точность/полнота/F1, точность/полнота множественных ячеек, точность/полнота
целей удаления, метрики от схемы для экспертного `keep/remove`, метрики с учётом стоимости ошибок,
ошибку координат удаления в пикселях и, при наличии калибровки, в миллиметрах.

5. Запустить простой базовый метод сравнения:

```powershell
py -m seedling_experiments baseline-green --config configs/example_experiment.yaml
```

После этого его можно оценить той же командой `evaluate-cells`, указав в
конфигурации путь к его `predictions.json`. `baseline-green` не создаёт `mAP`;
его сравнивают с YOLO по метрикам конечной задачи из `cell_metrics.json`.

## Важные настройки

Относительные пути в конфигурации считаются от текущей рабочей директории, а не
от файла конфигурации. Для повторяемого эксперимента запускайте команды из
`code-diplome/` и используйте новый пустой `dataset.prepared_root` для каждого
полного `prepare`.

Перед запуском `seedling_experiments prepare/train/val/predict/evaluate` CLI
проверяет структуру конфигурации и существование входных путей относительно
текущей рабочей директории. Выходные директории не обязаны существовать заранее.

Пороги `validation.conf/iou` относятся к команде `val`, а пороги
`prediction.conf/iou` — к `predictions.json`. После изменения параметров
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
диагностики postprocessing, но финальную таблицу нужно подписывать по источнику
контейнеров, который реально использован.

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
`prepare --config`, потому что он одним согласованным запуском сохраняет
`config.yaml`, `run_snapshot.json`, разбиение, аугментацию только `train`,
integrity-check и аудит.

## Старые скрипты

- `learning_yolo.py` — исходный минимальный запуск обучения YOLO.
- `matrix_for_container.py` — прототип построения матриц по детекциям.
- `many_seedlings.py` — прототип визуализации множественных сеянцев.
- `filter.py` — прототип фотометрической аугментации.
- `model_info.py` — исходный вывод метрик модели.

Для новых результатов используйте CLI выше: он фиксирует разбиение,
аугментации, параметры запуска, окружение и метрики конечной задачи.
