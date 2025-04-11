import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

def find_red_marker(image):
    """
    Находит красную метку на изображении
    
    :param image: Изображение в формате OpenCV (BGR)
    :return: Координаты (x, y) красной метки или None, если метка не найдена
    """
    # Конвертируем в HSV для лучшего выделения красного цвета
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Определяем диапазоны красного цвета в HSV
    # Красный цвет в HSV имеет значения около 0 и 180 (круговая шкала)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Создаем маски для обоих диапазонов красного цвета
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Объединяем маски
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Применяем морфологические операции для удаления шума
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Сохраняем маску для отладки
    debug_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'images', 
        'debug'
    ))
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, 'red_mask.png'), mask)
    
    # Находим контуры красных областей
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Если контуры найдены, берем самый большой
    if contours:
        # Сортируем контуры по площади (от большего к меньшему)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Берем самый большой контур (предполагаем, что это наша метка)
        largest_contour = contours[0]
        
        # Находим центр масс контура
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    
    return None

def segment_leaf_grabcut(image, marker_position, rect_size=150):
    """
    Сегментирует лист с использованием алгоритма GrabCut,
    используя красную метку как центр области интереса
    
    :param image: Изображение в формате OpenCV (BGR)
    :param marker_position: Координаты (x, y) красной метки
    :param rect_size: Размер прямоугольника вокруг метки
    :return: Маска сегментированного листа
    """
    # Создаем прямоугольник вокруг метки
    x, y = marker_position
    half_size = rect_size // 2
    
    # Убедимся, что прямоугольник не выходит за границы изображения
    height, width = image.shape[:2]
    rect_x = max(0, x - half_size)
    rect_y = max(0, y - half_size)
    rect_width = min(width - rect_x, rect_size)
    rect_height = min(height - rect_y, rect_size)
    
    rect = (rect_x, rect_y, rect_width, rect_height)
    
    # Подготавливаем маску и модели для GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Инициализируем маску: область внутри прямоугольника - вероятный передний план
    cv2.rectangle(mask, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), cv2.GC_PR_FGD, -1)
    
    # Отмечаем область вокруг метки как точно передний план
    marker_radius = 20
    cv2.circle(mask, marker_position, marker_radius, cv2.GC_FGD, -1)
    
    # Запускаем GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    
    # Создаем маску, где 0 и 2 - фон, 1 и 3 - передний план
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Применяем морфологические операции для улучшения маски
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    
    # Сохраняем маску для отладки
    debug_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'images', 
        'debug'
    ))
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, 'leaf_mask.png'), mask2 * 255)
    
    return mask2

def extract_leaf_contour(mask):
    """
    Извлекает контур листа из маски
    
    :param mask: Бинарная маска листа
    :return: Контур листа
    """
    # Находим контуры в маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Если контуры найдены, берем самый большой
    if contours:
        # Сортируем контуры по площади (от большего к меньшему)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Берем самый большой контур (предполагаем, что это наш лист)
        largest_contour = contours[0]
        
        # Сглаживаем контур для более эстетичного вида
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return approx_contour
    
    return None

def process_image(image_path):
    """
    Обрабатывает изображение: находит красную метку, сегментирует лист и выделяет его контур
    
    :param image_path: Путь к изображению
    :return: True, если обработка прошла успешно, иначе False
    """
    try:
        print(f"Обрабатываем изображение: {image_path}")
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return False
        
        # Создаем копию для результата
        result_image = image.copy()
        
        # Находим красную метку
        marker_position = find_red_marker(image)
        if marker_position is None:
            print(f"Красная метка не найдена на изображении: {image_path}")
            
            # Альтернативный метод: поиск ярко-красных пикселей
            red_pixels = np.where(
                (image[:,:,2] > 200) & 
                (image[:,:,1] < 100) & 
                (image[:,:,0] < 100)
            )
            
            if len(red_pixels[0]) > 0:
                # Берем средние координаты красных пикселей
                y_coords = red_pixels[0]
                x_coords = red_pixels[1]
                
                marker_position = (int(np.mean(x_coords)), int(np.mean(y_coords)))
                print(f"Найдена красная метка альтернативным методом: {marker_position}")
            else:
                return False
        
        print(f"Найдена красная метка: {marker_position}")
        
        # Отмечаем найденную метку на изображении для отладки
        debug_image = image.copy()
        cv2.circle(debug_image, marker_position, 10, (0, 255, 255), -1)
        
        debug_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'images', 
            'debug'
        ))
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f'marker_{os.path.basename(image_path)}'), debug_image)
        
        # Сегментируем лист с использованием GrabCut
        leaf_mask = segment_leaf_grabcut(image, marker_position)
        
        # Извлекаем контур листа
        leaf_contour = extract_leaf_contour(leaf_mask)
        if leaf_contour is None:
            print(f"Не удалось извлечь контур листа: {image_path}")
            return False
        
        # Рисуем контур на изображении фиолетовым цветом
        cv2.drawContours(result_image, [leaf_contour], 0, (255, 0, 255), 3)
        
        # Сохраняем результат
        results_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'images', 
            'results'
        ))
        os.makedirs(results_dir, exist_ok=True)
        
        output_path = os.path.join(
            results_dir, 
            f'leaf_contour_{os.path.basename(image_path)}'
        )
        
        cv2.imwrite(output_path, result_image)
        
        print(f"Результат сохранен: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    
    if not image_files:
        print(f"В директории {directory} не найдено изображений")
        return
    
    print(f"Найдено {len(image_files)} изображений для обработки")
    
    # Обработка каждого изображения
    success_count = 0
    for image_path in image_files:
        if process_image(image_path):
            success_count += 1
    
    print(f"Успешно обработано {success_count} из {len(image_files)} изображений")

def main():
    # Путь к директории с изображениями
    images_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'images', 
        'normal_plants_target_point'
    ))
    
    # Проверяем существование директории
    if not os.path.exists(images_dir):
        print(f"Директория не найдена: {images_dir}")
        print("Создаем директорию...")
        os.makedirs(images_dir, exist_ok=True)
        print(f"Пожалуйста, поместите изображения с красными метками в директорию: {images_dir}")
        return
    
    # Обработка всех изображений
    print(f"Начинаем обработку изображений из директории: {images_dir}")
    process_directory(images_dir)
    
    print("Обработка завершена.")

if __name__ == "__main__":
    main()
