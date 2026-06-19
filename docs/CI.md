# CI и проверки разработчика

Локальные проверки:

```powershell
py -m pip install -e .[dev]
pre-commit run --all-files
py -B -m unittest discover -s tests
```

Рабочий процесс CI в `.github/workflows/ci.yml` устанавливает пакет с
зависимостями для разработки, запускает
`pre-commit run --all-files --show-diff-on-failure`, затем выполняет полный
набор `unittest`.

Конфигурация `pre-commit` проверяет синтаксис YAML/JSON, пробелы в конце строк,
наличие завершающей новой строки, правила Ruff и форматирование Ruff.
Зависимости для разработки включают Gymnasium, поэтому регрессионная проверка
среды `SeedlingTrayEnv` запускается в CI без установки зависимостей для обучения
Stable-Baselines.
