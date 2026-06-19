# Спецификация RL

Первый слой RL работает только в симуляции. Он не управляет реальным
оборудованием и не обходит `SafetyGate`.

## Среда

`seedling_rl.envs.SeedlingTrayEnv` следует интерфейсу Gymnasium
`reset`/`step`/`render`. Если Gymnasium установлен, среда отдаёт настоящие
пространства Gymnasium; иначе тот же Python API остаётся доступным для тестов и
оценки базовых политик.

Набор модульных тестов включает регрессионную проверку совместимости Gymnasium.
Она запускается при установленных зависимостях разработки и пропускается только
в минимальных установках без Gymnasium.

Среда читает блок `noise:` из `configs/simulation/tray_env_v0.yaml` и перед
сбросом симулятора применяет к сгенерированным логическим сценам эффекты
пропуска детекций, ложных срабатываний, шума класса и сдвига точек.

Пространство действий:

```text
0..max_targets-1 = выбрать target_i
max_targets      = отправить неопределённые цели на проверку
max_targets + 1  = остановиться
```

Наблюдение:

```text
cell_tensor      [rows, cols, 11]
target_features  [max_targets, 8]
target_mask      [max_targets]
robot_state      [3]
action_mask      [max_targets + 2]
```

Недопустимые или небезопасные действия маскируются. Реальные исполнительные
действия не входят в среду MVP.

## Награда

Сигнал награды намеренно привязан к конечной задаче: он поощряет безопасное
удаление цели и штрафует повреждение целевого сеянца, небезопасные или
заблокированные действия, избыточное перемещение и ненужную проверку
оператором. Те же компоненты пишутся в метрики оценки, чтобы политика на правилах
политики и обученные политики можно было сравнивать без изменения контракта
безопасности.

## Базовые политики

Текущая оценка базовых политик поддерживает:

```text
noop
raster_scan
nearest_neighbor
route_planning
risk_aware_rule
```

CLI:

```powershell
py -m seedling_rl check-env --config configs/simulation/tray_env_v0.yaml
py -m seedling_rl evaluate-baselines --config configs/simulation/tray_env_v0.yaml --episodes 100
py -m seedling_rl train --config configs/rl/maskable_ppo_v0.yaml --dry-run
py -m seedling_rl train --config configs/rl/maskable_ppo_v0.yaml
py -m seedling_rl train --config configs/rl/recurrent_ppo_v0.yaml --dry-run
py -m seedling_rl evaluate --config configs/simulation/tray_env_v0.yaml --checkpoint runs/rl/maskable_ppo_v0_seed42/model.zip --algorithm maskable_ppo --episodes 100
py -m seedling_rl evaluate --config configs/simulation/tray_env_v0.yaml --checkpoint runs/rl/recurrent_ppo_v0_seed42/model.zip --algorithm recurrent_ppo --episodes 100
py -m seedling_rl evaluate --config configs/simulation/tray_env_v0.yaml --baseline raster_scan --episodes 100
py -m seedling_rl vector-env --config configs/simulation/tray_env_v0.yaml --n-envs 4
py -m seedling_rl curriculum --config configs/simulation/tray_env_v0.yaml --out runs/rl/curriculum.json
py -m seedling_rl offline-replay-eval --replays runs/sim/replay.json --out runs/rl/offline_replay_eval.json
py -m seedling_rl sweep --config configs/rl/sweep.example.yaml --out runs/rl/sweep_plan.json
py -m seedling_rl sweep-report --plan runs/rl/sweep_plan.json --metrics runs/rl/*/rl_eval_metrics.json --out runs/rl/sweep_report
py -m seedling_experiments rl --config configs/rl/maskable_ppo_v0.yaml --dry-run --out reports/unified_rl_dry_run.json
py -m seedling_experiments rl --config configs/rl/maskable_ppo_v0.yaml --mode evaluate-baselines --episodes 100 --out reports/unified_rl_baselines.json
```

Метрики включают среднюю награду, долю успешных целей, долю критических ошибок,
долю проверок, число действий на кассету, суммарное перемещение и среднюю
ошибку расстояния.

`seedling_rl evaluate` пишет `rl_eval_metrics.json`, `critical_events.json`,
пошаговые `rl_metrics.jsonl` / `rl_metrics.csv` и журналы воспроизведения эпизодов в
`replays/`. Оценка базовых политик использует ту же форму артефактов воспроизведения,
что и оценка контрольной точки, поэтому небезопасные действия, события повреждения
сеянцев и пути воспроизведения можно аудитировать без обученной модели.

`train` использует Stable-Baselines3 `PPO` или SB3-Contrib `MaskablePPO` /
`RecurrentPPO`, когда эти необязательные зависимости установлены. `--dry-run`
проверяет YAML, создаёт папку запуска, `training_log.jsonl`, `rl_metrics.jsonl`,
`rl_metrics.csv` и метаданные реестра без импорта SB3. Это сохраняет базовую
установку CV и среды выполнения лёгкой и одновременно оставляет реальный путь обучения RL.

Если конфигурация обучения задаёт `n_envs > 1`, реальное обучение строит
SB3 `DummyVecEnv` через `make_vectorized_env`; `n_envs: 1` оставляет путь одной
среды, используемый лёгкими тестами.

Лёгкий журнал в `seedling_rl.callbacks` записывает `reward`, `critical_error`,
`review_rate`, `movement_mm`, `total_distance_mm` и `mean_distance_error_mm`.
При реальном обучении SB3 он подключается как необязательный callback; при
оценке и сухом запуске те же артефакты пишутся напрямую.

Для частичной наблюдаемости и отложенной проверки конфигурация обучения может
использовать:

```yaml
algorithm: recurrent_ppo
policy: MultiInputLstmPolicy
recurrent: true
```

Этот путь использует SB3-Contrib `RecurrentPPO` только при реальном обучении;
сухая проверка фиксирует рекуррентный режим без импорта SB3. Оценка контрольной точки для
`--algorithm recurrent_ppo` сохраняет состояние LSTM между шагами и передаёт
маркер `episode_start` при сбросе эпизода.

`vector-env` пишет метаданные векторизованной среды без импорта SB3, а
`curriculum` пишет staged-планы генерации сцен для проверки до запуска тяжёлого
обучения. `offline-replay-eval` читает журналы воспроизведения без шагов среды и пишет
`offline_replay_eval.json` с числами разрешённых/заблокированных действий,
причинами блокировок, блокировками на уровне адаптера, критическими событиями,
долей проверок, успехом целей, повреждениями целевых сеянцев, наградой и
метриками движения/ошибки расстояния.

`sweep` раскрывает детерминированные сетки гиперпараметров и случайные зёрна в план сухого
запуска; фактическое обучение всё равно требует необязательные зависимости RL.
`sweep-report` читает полученные `rl_eval_metrics.json` и пишет
`sweep_stability_summary.json` / `.csv` с числом случайных зёрен, средним, стандартным
отклонением, минимумом, максимумом и размахом для награды, критической ошибки и
успеха целей.

Команды `seedling_rl`, которые пишут файлы, создают локальные
`artifact_registry.json`, `artifact_registry.csv` и снимки команд рядом с
выходами. Папки обучения/оценки пишут `run_snapshot.json`; лёгкие обёртки пишут
`*.run_snapshot.json` на каждый выходной файл. Реестры записывают входы
конфигураций/воспроизведений/планов/метрик и JSON/CSV/снимки с ролями входа/выхода.

`RLPolicyAdapter` подключает загруженную RL-модель через API `DecisionPolicy`.
Он возвращает только `ActionPlan`; выполнение всё равно требует `SafetyGate`.
Если адаптер создан для рекуррентной модели, он сохраняет recurrent-состояние
между вызовами `propose_plan()` и очищает его на `reset()`.

Когда модель выбирает проверку, остановку, недопустимый индекс или
замаскированную цель, адаптер записывает `rl_action_kind`, `rl_action_index`,
`review_reasons_by_target` и `blocked_reasons_by_target` в
`ActionPlan.metadata`, чтобы интерфейс оператора и отчёты могли объяснить
решение.
