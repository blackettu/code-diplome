# code-diplome

Вся рабочая документация проекта находится в папке `docs/`.

Начните с [docs/README.md](docs/README.md): там есть карта документов, быстрый старт,
актуальный порядок эксперимента и ссылки на подробные протоколы.

Пакет можно установить в режиме разработки из корня репозитория:

```powershell
py -m pip install -e .
py -m pip install -r requirements/base.txt
seedling-experiments --help
```

Дополнительные наборы зависимостей разделены в `requirements/`:
`vision.txt`, `training.txt`, `rl.txt`, `robot.txt` и `dev.txt`.
