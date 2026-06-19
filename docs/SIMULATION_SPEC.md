# Спецификация симуляции

`seedling_sim` реализует логический слой симуляции, который нужен до интеграции
Gymnasium/RL:

- JSON-схема `SimScene` для растений, целей, сетки и положения робота.
- `LogicalTraySimulator` с одношаговым выполнением цели.
- `SimSceneGenerator` для пустых, одиночных, множественных, сорных и неизвестных
  логических кассет.
- `DetectionNoiseModel` для пропусков детекции, ложных срабатываний, шума
  класса и сдвига `bbox`.
- `SimSceneDetectionNoiseModel`, который передаёт те же параметры шума в
  логические сцены для `SeedlingTrayEnv`.
- `ActuatorErrorModel` для ошибки XY, дрейфа, задержки и настраиваемых смещений
  по зонам.
- Необязательная интеграция `PlantResponseModel` как заглушки биологического
  ответа без логики дозы лазера. Стандартный путь `LogicalTraySimulator`
  остаётся детерминированным; при передаче модели ответа каждый результат шага
  получает `plant_response` и может моделировать отсутствие реакции или
  необходимость проверки неизвестных растений.
- Статические SVG/HTML-рендеры для сцен и журналов воспроизведения.
- Преобразование реальных сцен предсказаний `SceneState -> SimScene`.
- Синтетический PNG-рендер с наборами доменной рандомизации для визуальной
  проверки.

Этот симулятор намеренно не является физической или биологической моделью
лазера. Это детерминированная и тестируемая песочница для порядка действий
политик, проверок безопасности целей и будущей работы с воспроизведением/RL.

`configs/simulation/tray_env_v0.yaml` передаёт блок `noise:` в
`SeedlingTrayEnv`. Для сгенерированных логических сцен пропуски детекции
удаляют наблюдаемые растения/цели, шум классификации превращает цели в
неизвестные растения с обязательной проверкой, шум центра `bbox` сдвигает точки
растений/целей в миллиметрах, а ложные срабатывания добавляют низкоуверенные
неизвестные растения с целями для проверки.

Текущий CLI:

```powershell
py -m seedling_sim generate-scenes --config configs/simulation/tray_env_v0.yaml --count 100 --out data/sim_scenes/v0
py -m seedling_sim render-scene --scene data/sim_scenes/v0/scene_000000.json --out reports/sim_scene_000000.html
py -m seedling_sim image-backed-scene --scene-state runs/predict/scenes.json --out data/sim_scenes/image_backed/tray001.json --cell-size-mm 33,33
py -m seedling_sim render-synthetic --scene data/sim_scenes/v0/scene_000000.json --out reports/sim_scene_000000.png --preset greenhouse_default
py -m seedling_sim randomization-presets
py -m seedling_sim validate-scene --scene reports/sim_scene.json
py -m seedling_sim step --scene reports/sim_scene.json --target-id target_001
py -m seedling_sim run-policy --scene data/sim_scenes/v0/scene_000000.json --policy route_planning --out runs/sim/replay_route_scene000000.json
py -m seedling_sim compare-policies --scenes data/sim_scenes/v0 --policies raster_scan route_planning risk_aware_rule --out reports/policy_comparison.html
py -m seedling_sim play-policy --scene data/sim_scenes/v0/scene_000000.json --policy route_planning --render html --out reports/policy_replay.html
py -m seedling_experiments sim --config configs/simulation/tray_env_v0.yaml --out reports/unified_sim_smoke.json
```

`run-policy` пишет JSON-журнал воспроизведения с событиями `policy_start`, `robot_execution` и
`policy_stop`. `compare-policies` пишет JSON- или HTML-сводку по базовым
политикам; для HTML также создаётся JSON-файл с теми же строками. `play-policy`
рендерит одно воспроизведение для проверки оператором. Эти команды поддерживают
детерминированные базовые политики, но не прямое физическое действие.

Команды `seedling_sim`, которые пишут файлы, создают локальные
`artifact_registry.json`, `artifact_registry.csv` и `*.run_snapshot.json` рядом
со сценой, рендером, воспроизведением или сравнением. В них отдельно записываются входные
конфигурации/сцены, сгенерированные артефакты и аргументы команды.

Наборы доменной рандомизации сейчас покрывают нейтральный рендеринг, типичную
теплицу, слабое шумное освещение и влажный субстрат. Наборы могут менять
освещение, размытие фокуса, шум изображения, дрожание растений и
`calibration_drift_mm`, который сдвигает растения и маркеры целей относительно
сетки кассеты. Это визуальные стресс-факторы для офлайн-проверки и будущих
экспериментов обучения, а не проверенная замена реальным данным.
