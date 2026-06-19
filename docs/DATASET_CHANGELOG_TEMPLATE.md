# Шаблон журнала изменений набора данных

Используйте отдельный раздел для каждой версии набора данных. После заморозки
версии записи должны оставаться неизменными; новые факты добавляются в новую
версию, а не правкой старой.

## zks_v0_1

- Статус: черновик / заморожен / архивирован
- Дата:
- Автор:
- Корень исходного набора данных: data/zks_v0_1
- Версия онтологии:
- Версия руководства по разметке:
- Версия разбиения:
- Версия калибровки:

### Добавлено

-

### Изменено

-

### Удалено

-

### Известные проблемы

-

### Проверка

```powershell
py -m seedling_data audit-changelog --dataset-root data/zks_v0_1 --dataset-version zks_v0_1
py -m seedling_data registry validate --dataset zks_v0_1
py -m seedling_data near-duplicates --manifest data/zks_v0_1/manifests/image_manifest.csv --out reports/near_duplicates.json
```
