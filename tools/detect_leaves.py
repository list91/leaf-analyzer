import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
from scipy import ndimage

from cnn_model import load_model
from cnn_visualize import preprocess_image, predict_image

def detect_leaves(image_path, model_path, output_path=None, device='cpu', 
                  min_blob_size=300, detection_threshold=0.7):
    """
    Обнаружение и классификация отдельных листьев на изображении
    с использованием алгоритма водораздела (watershed)
    
    Args:
        image_path: путь к изображению
        model_path: путь к модели
        output_path: путь для сохранения результата
        device: устройство для вычислений
        min_blob_size: минимальный размер области для классификации (в пикселях)
        detection_threshold: порог для обнаружения листьев
    """
    # Загрузка модели
    print(f"Загрузка модели из {model_path}...")
    model, _ = load_model(model_path, device)
    model.eval()
    
    # Загрузка изображения
    print(f"Обработка изображения: {image_path}")
    original_image = Image.open(image_path).convert('RGB')
    image_np = np.array(original_image)
    
    # Создаем директорию для отладочных изображений
    debug_dir = os.path.join(os.path.dirname(output_path), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Конвертация в HSV для лучшей сегментации зеленых областей
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Расширенный диапазон для зеленого цвета (листья)
    lower_green = np.array([20, 30, 30])
    upper_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Сохраняем исходную маску
    cv2.imwrite(os.path.join(debug_dir, "1_initial_mask.png"), green_mask)
    
    # Применяем морфологические операции для улучшения сегментации
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(os.path.join(debug_dir, "2_opening.png"), opening)
    
    # Дилатация для увеличения областей
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imwrite(os.path.join(debug_dir, "3_sure_bg.png"), sure_bg)
    
    # Дистанционное преобразование для определения центров листьев
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    cv2.imwrite(os.path.join(debug_dir, "4_sure_fg.png"), sure_fg)
    
    # Определение неизвестной области
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imwrite(os.path.join(debug_dir, "5_unknown.png"), unknown)
    
    # Маркировка маркеров для водораздела
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Применение алгоритма водораздела
    markers = cv2.watershed(image_np, markers)
    image_np[markers == -1] = [0, 0, 255]  # Отмечаем границы красным цветом
    
    # Сохраняем изображение с маркерами
    watershed_image = image_np.copy()
    cv2.imwrite(os.path.join(debug_dir, "6_watershed.png"), cv2.cvtColor(watershed_image, cv2.COLOR_RGB2BGR))
    
    # Создаем нормализованную маску для визуализации маркеров
    markers_normalized = np.uint8(255 * (markers - markers.min()) / (markers.max() - markers.min() + 1e-8))
    colored_markers = cv2.applyColorMap(markers_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(debug_dir, "7_markers.png"), colored_markers)
    
    # Сохраняем маску для отладки
    mask_path = os.path.join(os.path.dirname(output_path), "mask_" + os.path.basename(image_path))
    cv2.imwrite(mask_path, markers_normalized)
    
    # Создаем копию оригинального изображения для отрисовки результатов
    result_image = original_image.copy()
    draw = ImageDraw.Draw(result_image)
    
    # Попытка загрузки шрифта (для Windows)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Классификация каждого сегмента (каждого потенциального листа)
    leaf_count = 0
    healthy_count = 0
    diseased_count = 0
    
    # Уникальные маркеры (исключая фон (1) и границы (-1))
    unique_markers = np.unique(markers)
    unique_markers = unique_markers[unique_markers > 1]  # Исключаем фон и границы
    
    print(f"Найдено {len(unique_markers)} потенциальных листьев")
    
    for marker in unique_markers:
        # Создаем маску для текущего маркера
        marker_mask = (markers == marker).astype(np.uint8) * 255
        
        # Проверяем размер области
        if np.sum(marker_mask) / 255 < min_blob_size:
            continue
        
        # Находим ограничивающий прямоугольник
        contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Добавляем отступы к прямоугольнику
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image_np.shape[1] - x, w + 2*padding)
        h = min(image_np.shape[0] - y, h + 2*padding)
        
        # Вырезаем область с листом
        leaf_image = image_np[y:y+h, x:x+w]
        
        # Конвертируем в PIL Image
        leaf_pil = Image.fromarray(leaf_image)
        
        # Сохраняем временное изображение для классификации
        temp_path = os.path.join(os.path.dirname(image_path), f"temp_leaf_{marker}.jpg")
        leaf_pil.save(temp_path)
        
        try:
            # Классификация с использованием пути к файлу
            pred_class, confidence, _ = predict_image(model, temp_path, device)
            
            # Удаляем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            # Если уверенность ниже порога, пропускаем
            if confidence < detection_threshold:
                continue
                
            leaf_count += 1
            
            # Определение класса и цвета прямоугольника
            if pred_class == 0:  # Здоровый
                color = "green"
                label_text = f"Здоровый ({confidence:.2f})"
                healthy_count += 1
            else:  # Больной
                color = "red"
                label_text = f"Больной ({confidence:.2f})"
                diseased_count += 1
                
            print(f"Лист {leaf_count}: {label_text}")
            
            # Расчет толщины линии на основе уверенности (от 1 до 6 пикселей)
            line_width = int(1 + confidence * 5)  # min=1, max=6
            
            # Рисуем прямоугольник и текст с изменяемой толщиной
            draw.rectangle([x, y, x+w, y+h], outline=color, width=line_width)
            
            # Добавляем черный фон для текста для лучшей видимости
            text_w, text_h = font.getsize(label_text) if hasattr(font, 'getsize') else draw.textbbox((0, 0), label_text, font=font)[2:4]
            draw.rectangle([x, y, x+text_w, y+text_h], fill=color)
            draw.text((x, y), label_text, fill="white", font=font)
            
        except Exception as e:
            print(f"Ошибка при классификации области {marker}: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            continue
    
    # Добавляем итоговую статистику
    summary_text = f"Всего листьев: {leaf_count} (Здоровых: {healthy_count}, Больных: {diseased_count})"
    draw.text((10, 10), summary_text, fill="black", font=font)
    
    # Сохраняем результат
    if output_path:
        result_image.save(output_path)
        print(f"Результат сохранен в {output_path}")
    
    # Возвращаем результат
    return result_image, leaf_count, healthy_count, diseased_count

def process_directory(input_dir, model_path, output_dir=None, device='cpu'):
    """
    Обработка всех изображений в директории
    
    Args:
        input_dir: директория с изображениями
        model_path: путь к модели
        output_dir: директория для сохранения результатов
        device: устройство для вычислений
    """
    # Создание выходной директории
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Получение списка изображений
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"В директории {input_dir} не найдено изображений")
        return
    
    print(f"Найдено {len(image_files)} изображений")
    
    # Обработка каждого изображения
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_dir, image_file)
        
        if output_dir:
            output_path = os.path.join(output_dir, f"detected_{image_file}")
        else:
            output_path = None
        
        print(f"Обработка изображения {i+1}/{len(image_files)}: {image_file}")
        
        try:
            result_image, leaf_count, healthy_count, diseased_count = detect_leaves(
                image_path, model_path, output_path, device, detection_threshold=0.7
            )
            
            print(f"Найдено листьев: {leaf_count} (Здоровых: {healthy_count}, Больных: {diseased_count})")
        except Exception as e:
            print(f"Ошибка при обработке {image_file}: {str(e)}")

if __name__ == '__main__':
    # Путь к данным
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'test')
    
    # Получение последней модели
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(checkpoint_dir, latest_model)
    
    # Директория для результатов
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'detection')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Используется модель: {latest_model}")
    
    # Обработка всех изображений в тестовой директории
    process_directory(test_dir, model_path, results_dir)
