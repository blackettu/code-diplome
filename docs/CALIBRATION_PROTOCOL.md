# Протокол калибровки

`seedling_calibration` задаёт первый контракт калибровки:

- `CalibrationConfig` описывает контрольные цели камеры/кассеты и геометрию
  сетки.
- `CalibrationArtifact` хранит `image_to_tray_homography`,
  `tray_to_robot_transform`, `tool_offset_mm`, масштаб px/mm и сводку ошибок.
- `CalibrationValidator` проверяет срок действия, тип кассеты и пределы ошибок.

Поток координат:

```text
image_px -> tray_mm -> robot_frame_mm -> tool offset
```

Используйте `attach_calibration_to_target` для одного `ActionTarget` или
`attach_calibration_to_scene` для полного `SceneState`. Помощник уровня сцены
заполняет у каждой цели `action_point_mm` и `robot_point_mm`, а также записывает
`CalibrationArtifact.calibration_id` в `SceneState.tray.calibration_id`. Сам по
себе он не выставляет `SceneState.safety.calibration_valid=true`.

Текущий CLI:

```powershell
py -m seedling_calibration estimate --config data/calibration/session01/calibration_config.yaml --out data/calibration/session01/calibration.json --calibration-id session01
py -m seedling_calibration validate --calibration data/calibration/session01/calibration.json --tray-type 11x11 --max-p95-mm 2.0 --out data/calibration/session01/calibration_validation.json
py -m seedling_calibration error-map --config data/calibration/session01/calibration_config.yaml --calibration data/calibration/session01/calibration.json --out data/calibration/session01/error_map.json
py -m seedling_calibration error-map --config data/calibration/session01/calibration_config.yaml --calibration data/calibration/session01/calibration.json --out reports/error_map_session01.html
py -m seedling_calibration error-budget --components configs/calibration/error_budget.example.json --calibration data/calibration/session01/calibration.json --out data/calibration/session01/error_budget.json --max-total-mm 3.0
py -m seedling_calibration estimate-intrinsics --observations configs/calibration/intrinsics_observations.example.json --camera-id cam01 --out data/calibration/session01/intrinsics.json
py -m seedling_calibration validate-intrinsics --intrinsics data/calibration/session01/intrinsics.json --max-reprojection-error-px 1.0 --out data/calibration/session01/intrinsics_validation.json
```

`estimate` использует оценку гомографии DLT на numpy минимум по четырём парам
контрольных точек изображение/кассета. Если есть опорные точки робота, команда
также оценивает `tray_to_robot_transform`; иначе преобразование робота по
умолчанию тождественное.

`error-map` пишет остатки по контрольным точкам и сводку ошибок в миллиметрах.
Если путь выхода заканчивается на `.html`, команда создаёт статический отчёт
карты ошибок калибровки по кассете/сетке с ожидаемыми точками, предсказанными
точками и таблицей остатков.

Для `readiness-check` доказательство калибровки принимается только тогда, когда
артефакт `calibration.json` проходит `CalibrationValidator`, не имеет истечения
или предупреждающих проблем, сохраняет `error_summary_mm.p95 <= 2.0` и
сопровождается артефактом `error_map` в том же корне запуска/отчёта.

`error-budget` пишет `error_budget.json` с архитектурным RSS-бюджетом ошибок:

```text
e_total^2 = e_detection^2 + e_grid^2 + e_calibration^2 + e_mechanics^2
          + e_focus^2 + e_latency^2 + e_biological_target^2
```

Вход компонентов — JSON-объект или `{"components_mm": {...}}` с теми же именами
компонентов в миллиметрах. Пропущенные компоненты сохраняются как
`missing_components` и делают отчёт неполным. Если передан артефакт калибровки,
а `e_calibration` отсутствует, команда берёт это значение из
`calibration.error_summary_mm.p95`.

Внутренние параметры камеры хранятся в `CameraIntrinsicsArtifact`. Схема
наблюдений поддерживает цели `chessboard`, `aruco` и `fiducials`. Каждое
наблюдение хранит сопоставленные `image_points_px`, плоские `object_points_mm`,
`image_size_px` и необязательные метаданные; пример находится в
`configs/calibration/intrinsics_observations.example.json`. Оценка использует
OpenCV `calibrateCamera`, когда установлен `opencv-python`; проверка и
сериализация не требуют OpenCV.

Команды, которые пишут файлы, также создают ролевые `artifact_registry.json`,
`artifact_registry.csv` и `*.run_snapshot.json` в выходной папке. Реестр
записывает конфигурации калибровки, артефакты калибровки, наблюдения внутренних
параметров, сформированные отчёты и снимки команд как входные/выходные
артефакты для воспроизводимых экспериментальных отчётов.

Автономное реальное действие остаётся вне области проекта, пока ошибка
калибровки, межблокировки и биологическая безопасность не будут проверены
отдельно.
