# Протокол робота

Текущий слой робота — это абстрактный контракт, а не драйвер реального портала.

`RobotAdapter` предоставляет:

```text
connect()
home()
move_to(point_mm, speed)
mark_or_act(profile)
emergency_stop()
telemetry()
```

`SpeedProfile` хранит скорости XY/Z в мм/с. `ToolProfile` хранит запрошенный
режим инструмента; программные сухие прогоны и HIL-сценарии с указателем сейчас
принимают только `pointer_only`.

Реализованные адаптеры:

- `SimulatorRobotAdapter` проверяет каждый `ActionCommand` через `SafetyGate`
  до перемещения или симулированной маркировки. Он может работать с
  `LogicalTraySimulator` и добавлять события `robot_execution` в `ReplayLogger`.
- `DryRunSerialAdapter` — безопасная заглушка последовательного протокола. Он
  не открывает последовательный порт и не двигает оборудование. Адаптер
  сериализует записи `MotionCommand`, требует homing, подтверждённую
  межблокировку и исправные концевые выключатели, принудительно использует
  `pointer_only`, отдаёт телеметрию и пишет журналы команд, включая
  заблокированные решения `SafetyGate`.
- `RealGantrySerialAdapter` — HIL-адаптер указателя с защитными проверками. Он
  отказывается открывать последовательный порт без действительного
  `HardwareInLoopReview` и явного `allow_hardware=True`. Он принимает только
  профили инструмента `pointer_only`. Путь `execute_command` проверяет
  `ActionCommand` через `SafetyGate`, отправляет только команды `MOVE` и
  `POINT`, а заблокированные решения безопасности записывает в журнал/воспроизведение
  вместо молчаливого пропуска.

Реальные и сухие последовательные адаптеры должны сохранять одно правило:

```text
DecisionPolicy cannot execute hardware commands.
RobotAdapter cannot bypass SafetyGate.
RLPolicyAdapter must only produce plans through DecisionPolicy.
```

Опасные профили инструмента и реальное исполнительное действие намеренно не
реализованы до отдельного анализа безопасности.

Артефакт HIL-проверки:

```json
{
  "review_id": "hil-review-001",
  "reviewer": "operator",
  "approved_at": "2026-06-17T00:00:00Z",
  "expires_at": "2026-07-17T00:00:00Z",
  "gantry_id": "vilga-gantry-dev",
  "allowed_modes": ["hardware_in_loop_pointer"],
  "allowed_tool_profiles": ["pointer_only"],
  "calibration_id": "session01",
  "max_error_p95_mm": 2.0,
  "positioning_error_p95_mm": 1.0,
  "interlock_required": true,
  "limit_switch_required": true,
  "emergency_stop_tested": true,
  "allow_real_actuation": false
}
```

Загрузчик отклоняет артефакты с неизвестным `schema_version`. Валидатор
отклоняет просроченные проверки, неподдерживаемые режимы и профили инструмента,
отсутствующие требования межблокировки или концевых выключателей, а также любой
артефакт с `allow_real_actuation=true`. Если проверка содержит
`positioning_error_p95_mm`, значение должно быть не больше `max_error_p95_mm`;
иначе проверка падает с `positioning_error_p95_mm_exceeds_limit`.

Обычная проверка предупреждает, если нет доказательства ошибки позиционирования;
HIL-выполнение указателя считает это причиной `positioning_error_p95_mm_required`
и останавливается до создания последовательного адаптера. HIL-выполнение также
требует, чтобы проверка разрешала `hardware_in_loop_pointer` и `pointer_only`;
иначе runner прерывается с `mode_not_authorized:*` или
`tool_profile_not_authorized:*`. Для отсутствующего `calibration_id` или записи
проверки аварийного стопа выдаются предупреждения.

Повтор движения:

```python
from seedling_robot import replay_motion_commands
```

`replay_motion_commands` повторяет сериализованные `MotionCommand` через
`RobotAdapter`; с `DryRunSerialAdapter` это только записывает сухие команды. По
умолчанию повтор останавливается после первого неуспешного движения; передавайте
`stop_on_failure=False` только для явной диагностической проверки.

CLI-безопасный путь повтора использует адаптер симулятора и пишет JSON-отчёт с
локальным реестром артефактов:

```powershell
py -m seedling_robot motion-replay --commands reports/dry_run_commands.json --out reports/motion_replay.json
```

Вход команд может быть списком JSON-объектов `MotionCommand`, `{"commands": [...]}`,
`{"motion_commands": [...]}` или строками журнала команд с вложенным объектом
`command`. Строки журнала сухого прогона `move_to` преобразуются в
синтетические `MotionCommand`; служебные строки вроде `home` пропускаются.

Каждый адаптер записывает оцененный `SafetyDecision.result` обратно в
`ActionCommand.safety_gate_result`; `MotionCommand.metadata.safety_gate_result`
и `safety_token` переносят то же решение в журналы команд и файлы воспроизведения.

Пути команд сухого прогона:

```powershell
py -m seedling_robot dry-run-test --points data/calibration/session01/control_points.json --out reports/dry_run_control_points.json --command-log reports/dry_run_commands.json
py -m seedling_robot validate-hil-review --review configs/robot/hil_review_template.json --out reports/hil_review_validation.json --hil-pointer
py -m seedling_reports export-plan --scene runs/predict/scenes.json --policy route_planning --out reports/action_plan.json
py -m seedling_robot dry-run-plan --scene runs/predict/scenes.json --plan reports/action_plan.json --out reports/dry_run_plan_report.json --command-log reports/dry_run_plan_commands.json --replay reports/dry_run_plan_replay.json --interlock-ok
py -m seedling_robot hil-pointer-run --scene runs/predict/scenes.json --plan reports/action_plan.json --review configs/robot/hil_review_template.json --port COM5 --out reports/hil_pointer_report.json --command-log reports/hil_pointer_commands.json --replay reports/hil_pointer_replay.json --allow-hardware --interlock-ok
```

Файл точек имеет формат JSON:

```json
{"points":[{"point_id":"p1","expected_mm":[10.0,20.0,0.0]}]}
```

Сформированный `dry_run_control_points.json` содержит исходные строки точек и
`error_summary_mm` с `min`, `mean`, `p50`, `p95`, `p99` и `max`, поэтому
артефакт полезен ещё до того, как `seedling_reports build` пересчитает те же
колонки ошибки позиционирования для таблиц статьи.

Эта команда не открывает последовательный порт и не двигает реальное
оборудование. Она сериализует движения указателя в сухом прогоне и пишет отчёт
ошибок для проверки оператором.

CLI `dry-run-plan` загружает `SceneState` и экспортированный `ActionPlan`,
проверяет каждую команду через `SafetyGate`, сериализует команды указателя
только для сухого прогона и пишет отчёт, журнал команд и необязательное воспроизведение.
Он останавливается после первой неуспешной команды, записывает `aborted=true`,
считает невыполненные команды как `skipped` / `blocked` и никогда не открывает
реальный последовательный порт. Для успешного выполнения команда требует явный
флаг `--interlock-ok`; без него `SafetyGate` блокирует движение как
`BLOCK_INTERLOCK`. Команды, требующие подтверждения человека, остаются
заблокированными без `--operator-confirmed`.

HIL-выполнение указателя использует тот же контракт `ActionCommand`, что и
симуляция и сухие адаптеры. Действительный артефакт проверки и явное
`allow_hardware=True` обязательны до открытия последовательного порта. Адаптер
отклоняет неподдерживаемые профили инструмента до движения и записывает
`blocked_tool_profile`; `action_type=remove_*`, `laser_fire`, `cut` и `burn`
остаются заблокированными стандартными ограничениями безопасности.

CLI `hil-pointer-run` загружает `SceneState` и `ActionPlan`, проводит каждую
команду через `SafetyGate`, пишет JSON-отчёт и может сохранять журнал
последовательных команд и JSON-журнала воспроизведения для разбора ошибок. HIL-прогоны указателя
останавливаются после первой неуспешной команды, отправляют `ESTOP`, помечают
отчёт `aborted=true` и считают оставшиеся команды как `skipped` / `blocked`.

Если HIL-артефакт недействителен или просрочен, runner прерывается до создания
последовательного адаптера и всё равно пишет `hil_pointer_report.json` с
`aborted=true`, `abort_reason=invalid_hil_review` и ошибками проверки.

Для доказательства продуктовой готовности `readiness-check` принимает HIL
только если это успешный прогон `hardware_in_loop_pointer` минимум с одной
командой, `executed == commands`, ноль пропущенных/заблокированных команд, нет
прерывания, существуют журнал команд и воспроизведение, а встроенная проверка содержит
`review_id`, `gantry_id`, `positioning_error_p95_mm`, требования
межблокировки/концевых выключателей и `emergency_stop_tested: true`.
Минимальный отчёт `review: {"ok": true}` всё ещё считается `external_required`.

CLI-команды робота, которые пишут отчёты, также создают ролевые
`artifact_registry.json`, `artifact_registry.csv` и `*.run_snapshot.json` в
папке отчёта. Реестр записывает входы точек/проверки/сцены/плана и выходы
отчёта, журнала команд, воспроизведения и снимка команды, чтобы доказательства
сухого прогона/HIL можно было цитировать в экспериментальных отчётах.
