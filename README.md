# Leaf Analyzer

## Описание проекта
Инструмент для обнаружения и визуализации объектов на изображениях растений. 

## Возможности
- Автоматическое обнаружение контуров
- Выделение границ объектов
- Фильтрация и оптимизация детекции
- Гибкая настройка параметров

## Требования
- Python 3.10+
- OpenCV
- NumPy
- Pillow

## Установка
1. Клонируйте репозиторий
2. Создайте виртуальное окружение
3. Установите зависимости: `pip install -r requirements.txt`

## Использование
```bash
python tools/detect_plants.py
```

## Параметры
- `min_area`: Минимальная площадь контура
- Настройка цвета и толщины контуров

## Лицензия
MIT
