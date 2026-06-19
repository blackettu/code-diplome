# Статические просмотрщики

`seedling_ui` предоставляет статические HTML-просмотрщики, которые работают без
локального веб-приложения. Команды просмотра и обратной связи, которые пишут
файлы, создают `artifact_registry.json` / `artifact_registry.csv` и
`*.run_snapshot.json` рядом с выходом. Так исходные предсказания, сцены,
журналы воспроизведения и JSONL-файлы обратной связи остаются прослеживаемыми без
перезаписи снимков от других UI-команд в той же папке.

Просмотрщик офлайн-предсказаний:

```powershell
py -m seedling_ui offline-viewer --predictions runs/predict/predictions.json --out reports/offline_viewer.html
py -m seedling_ui offline-viewer --predictions runs/predict/scenes.json --image-name scene_001 --out reports/offline_scene_viewer.html
```

Просмотрщик воспроизведения симуляции:

```powershell
py -m seedling_ui replay-viewer --scene data/sim_scenes/v0/scene_000001.json --replay runs/sim/replay.json --out reports/replay_viewer.html
py -m seedling_sim play-policy --scene data/sim_scenes/v0/scene_000001.json --policy route_planning --render html --out reports/policy_replay.html
```

Экспорт отчёта по кассете или эпизоду:

```powershell
py -m seedling_ui report-export --scene runs/predict/scenes.json --out reports/tray_report.md
py -m seedling_ui report-export --scene data/sim_scenes/v0/scene_000001.json --replay runs/sim/replay.json --out reports/episode_report.html
```

Офлайн-просмотрщик принимает старый `predictions.json`, одиночный JSON
`SceneState` или пакет сцен вида `{"scenes": [...]}`. Для входов `SceneState` он
рисует полигоны кассеты/ячеек, детекции, цели действия, строки состояния ячеек
и причины проверки/блокировки. Для старых предсказаний он показывает причины
проверки/блокировки целей, если они присутствуют в `removal_targets` или во
встроенных `safety_decision`.

Просмотрщик воспроизведения содержит управление запуском, паузой, шагом и сбросом, сводку
безопасности с числами разрешённых, заблокированных и требующих проверки
действий, агрегированные причины, а также пошаговые решения `SafetyGate`.
Оператор или рецензент может увидеть, почему цель была разрешена или
отклонена.

`report-export` пишет Markdown или HTML в зависимости от расширения выходного
файла. Он сводит `SceneState` или `SimScene`, строки целей и, при наличии,
счётчики событий воспроизведения, награды и блокировки безопасности для проверки на
уровне кассеты или эпизода. Строки целей `SceneState` включают причины
проверки/блокировки из флагов цели, запрещённых зон и флагов риска ячейки;
разделы воспроизведения включают причины по событиям и агрегированные счётчики.

Обратная связь по разметке:

```powershell
py -m seedling_ui feedback --feedback reports/feedback.jsonl --image-id tray001.jpg --target-id target_001 --error-type wrong_target --comment "operator rejected target"
py -m seedling_ui feedback --feedback reports/feedback.jsonl --image-id tray001.jpg --cell-id r00_c00 --error-type bad_cell_state --priority high --proposed-correction-json '{"cell_state":"weed_only"}' --comment "operator corrected cell state"
py -m seedling_ui feedback --feedback reports/feedback.jsonl --summary
py -m seedling_ui feedback --feedback reports/feedback.jsonl --summary --annotation-tasks-out reports/annotation_tasks.jsonl
```

Файл обратной связи имеет формат JSONL. `--annotation-tasks-out` преобразует
строки обратной связи в открытые задачи разметки для последующей переразметки
или аудита. Строки могут содержать `priority` и `proposed_correction`;
сформированные задачи сохраняют оба поля для разметчиков.

Офлайн-просмотрщики также включают статическую панель обратной связи. Она
позволяет оператору выбрать отрисованную цель, ячейку или детекцию, выбрать тип
ошибки и скопировать готовую строку JSONL. Строка использует те же поля,
которые читает `seedling_ui feedback` и экспорт задач разметки.

Выбор модели или политики:

```powershell
py -m seedling_ui selector --registry configs/registry/models_v0_1.yaml --kind detector --name baseline_green
py -m seedling_ui selector --registry configs/registry/policies_v0_1.yaml --kind policy --name risk_aware_rule
py -m seedling_ui selector --registry configs/registry/policies_v0_1.yaml --kind policy --name route_planning
py -m seedling_ui selector --registry configs/registry/models_v0_1.yaml --validate
py -m seedling_ui selector --registry configs/registry/policies_v0_1.yaml --validate
```

Отчёты сравнения сценариев формируются через `seedling_reports`:

```powershell
py -m seedling_reports scenario-compare --scene runs/predict/scenes.json --policies raster_scan route_planning risk_aware_rule rl_no_model --out reports/scenario_compare
```
