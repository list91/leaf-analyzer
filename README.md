# Plant Leaf Analyzer

Система для автоматического поиска, сегментации и диагностики листьев растений на фото. Выделяет только больные листья.

## Быстрый старт

1. Установите зависимости:
   ```powershell
   pip install -r requirements.txt
   ```
2. Положите изображения в `images/test`.
3. Запустите детекцию:
   ```powershell
   python tools\detect_leaves.py
   ```
   Результаты будут в `results/detection`.

## Кастомизация
- Порог уверенности: `detection_threshold` в `detect_leaves.py`
- Фильтр по размеру: `min_width`, `min_height` в том же файле

## Структура
- `tools/` — скрипты
- `images/` — датасеты
- `checkpoints/` — модели
- `results/` — результаты

## Обучение модели (опционально)
```powershell
python tools\cnn_train.py
```