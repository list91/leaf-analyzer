import os
import sys
from PIL import Image, ImageDraw, ImageFilter
import math
from collections import defaultdict
import cv2
import numpy as np

def is_leaf_pixel(r, g, b):
    """
    Определяет, является ли пиксель частью фона (не листа)
    """
    return not (
        g > 60 and 
        g > r * 1.1 and 
        g > b * 1.1 and 
        g < 250
    )

def merge_nearby_regions(regions, max_distance=50):
    """
    Оптимизированное объединение близких областей
    """
    if not regions:
        return []
    
    # Создаем сетку для быстрого поиска соседей
    grid_size = max_distance
    grid = defaultdict(list)
    
    # Распределяем точки по сетке
    for region_id, region in enumerate(regions):
        # Используем центр масс региона
        cx = sum(x for x, _ in region) // len(region)
        cy = sum(y for _, y in region) // len(region)
        
        # Добавляем индекс региона в соответствующие ячейки сетки
        grid_x = cx // grid_size
        grid_y = cy // grid_size
        
        # Добавляем в текущую и соседние ячейки
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                grid[(grid_x + dx, grid_y + dy)].append(region_id)
    
    # Находим связные компоненты
    merged = []
    used = set()
    
    for region_id, region in enumerate(regions):
        if region_id in used:
            continue
            
        # Новый объединенный регион
        current = set(region)
        used.add(region_id)
        
        # Центр текущего региона
        cx = sum(x for x, _ in region) // len(region)
        cy = sum(y for _, y in region) // len(region)
        
        # Проверяем только регионы в соседних ячейках
        grid_x = cx // grid_size
        grid_y = cy // grid_size
        
        nearby_regions = set()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nearby_regions.update(grid[(grid_x + dx, grid_y + dy)])
        
        # Объединяем близкие регионы
        for other_id in nearby_regions:
            if other_id in used:
                continue
                
            other_region = regions[other_id]
            other_cx = sum(x for x, _ in other_region) // len(other_region)
            other_cy = sum(y for _, y in other_region) // len(other_region)
            
            # Проверяем расстояние между центрами
            if math.sqrt((cx - other_cx)**2 + (cy - other_cy)**2) < max_distance:
                current.update(other_region)
                used.add(other_id)
        
        merged.append(list(current))
    
    return merged

def detect_and_highlight_plants(image_path):
    """
    Обнаружение и выделение фона вокруг листьев
    """
    try:
        # Открытие и подготовка изображения
        image = Image.open(image_path)
        image_rgb = image.convert('RGB')
        image_blurred = image_rgb.filter(ImageFilter.GaussianBlur(2))
        draw = ImageDraw.Draw(image_rgb, 'RGBA')
        width, height = image_rgb.size
        
        # Поиск связных областей фона
        leaf_regions = []
        visited = set()
        
        def explore_region(start_x, start_y):
            """
            Исследование связной области фона
            """
            region = []
            to_check = [(start_x, start_y)]
            
            while to_check:
                x, y = to_check.pop()
                
                if (x < 0 or x >= width or 
                    y < 0 or y >= height or 
                    (x, y) in visited):
                    continue
                
                visited.add((x, y))
                
                # Проверяем пиксель только если его еще не посещали
                r, g, b = image_blurred.getpixel((x, y))
                
                if is_leaf_pixel(r, g, b):
                    region.append((x, y))
                    
                    # Проверяем соседей сразу во всех направлениях
                    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) not in visited:
                            to_check.append((nx, ny))
            
            return region
        
        # Оптимизированный поиск областей
        step = 2  # Увеличиваем шаг для ускорения
        for x in range(0, width, step):
            for y in range(0, height, step):
                if (x, y) not in visited:
                    r, g, b = image_blurred.getpixel((x, y))
                    if is_leaf_pixel(r, g, b):
                        region = explore_region(x, y)
                        if len(region) > width * height * 0.001:
                            leaf_regions.append(region)
        
        # Объединение и отрисовка
        merged_regions = merge_nearby_regions(leaf_regions)
        
        for region in merged_regions:
            # Создаем маски для контура и заливки
            outline_mask = Image.new('L', (width, height), 0)
            fill_mask = Image.new('L', (width, height), 0)
            outline_draw = ImageDraw.Draw(outline_mask)
            fill_draw = ImageDraw.Draw(fill_mask)
            
            # Рисуем точки на обеих масках
            for x, y in region:
                outline_draw.point((x, y), fill=255)
                fill_draw.point((x, y), fill=255)
            
            # Создаем контур путем вычитания размытой маски
            blurred = outline_mask.filter(ImageFilter.GaussianBlur(0.1))  # Очень легкое размытие
            outline = Image.blend(outline_mask, blurred, -0.5)  # Уменьшаем интенсивность смешивания
            
            # Слегка размываем маску заливки для сглаживания
            fill_mask = fill_mask.filter(ImageFilter.GaussianBlur(1))
            
            # Сначала рисуем контур (очень прозрачный светло-фиолетовый)
            draw.bitmap((0, 0), outline, fill=(200, 100, 255, 8))
            
            # Затем заливаем внутреннюю область (почти невидимый)
            draw.bitmap((0, 0), fill_mask, fill=(200, 100, 255, 8))
        
        # Сохранение результата
        results_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'images', 
            'results'
        ))
        os.makedirs(results_dir, exist_ok=True)
        
        output_path = os.path.join(
            results_dir, 
            f'highlighted_{os.path.basename(image_path)}'
        )
        image_rgb.save(output_path)
        
        print(f"Обработано изображение: {image_path}")
        print(f"Результат сохранен: {output_path}")
        print(f"Найдено областей фона: {len(merged_regions)}")
        
        return True
        
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return False

def detect_and_highlight_objects(image_path, min_area=100):
    """
    Обнаружение и выделение границ объектов на изображении
    
    :param image_path: Путь к исходному изображению
    :param min_area: Минимальная площадь контура для отрисовки
    :return: Путь к обработанному изображению
    """
    try:
        # Загрузка изображения
        image = cv2.imread(image_path)
        
        # Преобразование в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применение размытия для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Обнаружение краев с помощью алгоритма Кэнни
        edges = cv2.Canny(blurred, 50, 200)
        
        # Нахождение контуров
        contours, _ = cv2.findContours(
            edges, 
            cv2.RETR_EXTERNAL,  # Внешние контуры
            cv2.CHAIN_APPROX_SIMPLE  # Упрощенное представление контуров
        )
        
        # Отфильтровываем и рисуем контуры
        for contour in contours:
            # Вычисляем площадь контура
            area = cv2.contourArea(contour)
            
            # Рисуем только значимые контуры
            if area > min_area:
                # Аппроксимация контура для сглаживания
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # Рисование контура
                cv2.drawContours(
                    image, 
                    [approx], 
                    0,  # Индекс контура
                    (200, 100, 255),  # Цвет (светло-фиолетовый)
                    2  # Толщина линии
                )
        
        # Создаем директорию для результатов
        results_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'images', 
            'results'
        ))
        os.makedirs(results_dir, exist_ok=True)
        
        # Сохраняем результат
        output_path = os.path.join(
            results_dir, 
            f'objects_highlighted_{os.path.basename(image_path)}'
        )
        cv2.imwrite(output_path, image)
        
        print(f"Обработано изображение: {image_path}")
        print(f"Найдено контуров: {len(contours)}")
        print(f"Результат сохранен: {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return None

def process_directory(directory):
    """
    Обработка всех изображений в указанной директории
    
    :param directory: Путь к директории с изображениями
    """
    # Поддерживаемые расширения изображений
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # Получаем список файлов
    image_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and 
        os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    # Обработка каждого изображения
    for image_path in image_files:
        detect_and_highlight_objects(image_path)

def main():
    base_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'images', 
        'normal_plants'
    ))
    
    try:
        images = [
            os.path.join(base_path, filename) 
            for filename in os.listdir(base_path) 
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
    except FileNotFoundError:
        print(f"Директория не найдена: {base_path}")
        sys.exit(1)
    
    processed_images = []
    for image_path in images:
        if detect_and_highlight_plants(image_path):
            processed_images.append(image_path)
    
    print("\n--- Результаты ---")
    print(f"Всего изображений: {len(images)}")
    print(f"Обработано изображений: {len(processed_images)}")

if __name__ == '__main__':
    main()