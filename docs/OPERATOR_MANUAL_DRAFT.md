# Черновик руководства оператора

## Режимы

```text
offline_only        только просмотр файлов и отчётов
simulation_only     запуск логического симулятора и просмотр replay
dry_run_pointer     сериализация безопасных команд указателя без опасного действия
supervised_mark     будущая неразрушающая маркировка с подтверждением оператора
supervised_real     вне области проекта до анализа безопасности
```

## Процедура сухого прогона

1. Проверьте калибровку и пределы безопасности.
2. Сгенерируйте или загрузите артефакты сцены/предсказаний.
3. Запустите политику, чтобы создать `ActionPlan`.
4. Передайте каждый `ActionCommand` через `SafetyGate`.
5. Используйте `DryRunSerialAdapter` только в режиме `pointer_only`.
6. Проверьте `command_log.json`, журналы воспроизведения и причины блокировки.
7. Запишите обратную связь по ошибочной цели, пропущенной цели, ошибочному
   состоянию ячейки, проблеме калибровки или неопределённости модели.

CLI для полного сухого прогона плана:

```powershell
py -m seedling_robot dry-run-plan --scene runs/predict/scenes.json --plan reports/action_plan.json --out reports/dry_run_plan_report.json --command-log reports/dry_run_plan_commands.json --replay reports/dry_run_plan_replay.json --interlock-ok
```

Не указывайте `--operator-confirmed`, пока человек не проверит команды, которым
требуется подтверждение. Если флаг межблокировки не передан, движение
блокируется, а отчёт записывает причину `SafetyGate`.

## Процедура HIL с указателем

1. Подготовьте JSON-артефакт `HardwareInLoopReview` для портала, сессии
   калибровки, измеренной `positioning_error_p95_mm`, проверки оператором и
   даты истечения.
2. Проверьте артефакт:

```powershell
py -m seedling_robot validate-hil-review --review configs/robot/hil_review_template.json --out reports/hil_review_validation.json --hil-pointer
```

3. Используйте `RealGantrySerialAdapter` только для проверок
   `hardware_in_loop_pointer` и `pointer_only`. Адаптер всё равно блокирует
   профили реального действия и требует явный флаг `allow_hardware=True` перед
   открытием последовательного порта.
4. Выполняйте только те `ActionCommand`, которые прошли `SafetyGate`; после
   каждой успешной или заблокированной команды проверяйте журнал команд и
   журнал воспроизведения. HIL-планы с указателем останавливаются после первой неуспешной
   команды, отправляют `ESTOP` и помечают отчёт `aborted=true`; не продолжайте
   пропущенные команды без новой проверки.

CLI:

```powershell
py -m seedling_robot hil-pointer-run --scene runs/predict/scenes.json --plan reports/action_plan.json --review configs/robot/hil_review_template.json --port COM5 --out reports/hil_pointer_report.json --command-log reports/hil_pointer_commands.json --replay reports/hil_pointer_replay.json --allow-hardware --interlock-ok
```

5. Держите `supervised_real` вне области проекта, пока концепция безопасности,
   механические межблокировки, аварийный стоп и проверка калибровки не будут
   подтверждены на реальном стенде.

Команды сухого прогона робота и HIL пишут `artifact_registry.json` /
`artifact_registry.csv` рядом с отчётом. Храните эти файлы вместе с журналом
команд и журнала воспроизведения: это запись воспроизводимости для проверки оператором и таблиц
статьи.

## Обратная связь

Обратная связь хранится как JSONL:

```powershell
py -m seedling_ui feedback --feedback reports/feedback.jsonl --image-id tray001.jpg --target-id target_001 --error-type wrong_target --comment "operator marked target as wrong"
py -m seedling_ui feedback --feedback reports/feedback.jsonl --image-id tray001.jpg --cell-id r00_c00 --error-type bad_cell_state --priority high --proposed-correction-json '{"cell_state":"weed_only"}' --comment "operator corrected cell state"
py -m seedling_ui feedback --feedback reports/feedback.jsonl --summary
py -m seedling_ui feedback --feedback reports/feedback.jsonl --summary --annotation-tasks-out reports/annotation_tasks.jsonl
```

Файлы обратной связи являются входами для разметки и аудита, а не прямым
обновлением модели. `annotation_tasks.jsonl` передаётся на последующую
переразметку и сохраняет `priority` вместе со структурированным
`proposed_correction`.

## Проверка после действия

Для любого биологического пилота под наблюдением записывайте отложенные
наблюдения отдельно от журнала движения и воспроизведения. Окна наблюдения по умолчанию:
24, 48 и 72 часа.

```powershell
py -m seedling_data audit-post-action --path runs/followup/post_action_observations.jsonl --required-hours 24,48,72
py -m seedling_data summarize-post-action --path runs/followup/post_action_observations.jsonl --out runs/followup/post_action_summary.json
```

Каждая строка JSONL должна ссылаться на `command_id`, `target_id`, `scene_id`,
`tray_id`, `observed_at`, `hours_after_action` и результат, например
`target_removed`, `target_survived`, `crop_damage`, `regrowth`, `not_visible`
или `uncertain`. Для доказательств готовности каждая строка также должна
содержать `image_ref` и `observer_id`; сформированные сводки сохраняют
`observer_ids`.

Внешний gate закрывается только полным покрытием окон 24/48/72 часа без
пропусков наблюдений, итогом `target_removed` для всех команд, нулевым
повреждением целевых сеянцев/повторным ростом и отсутствием последних исходов
`uncertain` / `not_visible`. Эти записи являются артефактами отложенной
проверки, а не разрешением запускать реальный исполнительный механизм без
отдельного протокола безопасности и биологии.
