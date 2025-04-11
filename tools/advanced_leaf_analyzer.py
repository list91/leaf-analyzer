#!/usr/bin/env python3
# Адаптировано из: https://github.com/oguztoraman/leaf-segmentation

import cv2
import numpy as np
import time as t
import os
from scipy import ndimage
import argparse

class LeafSegmentation:
    """
    Класс для сегментации листьев на изображениях
    Использует алгоритм Оцу для создания маски и морфологические операции для улучшения результата
    """
    # Параметры фильтрации
    filter_size = (11, 11)
    filter_sigma = 5

    # Параметры порогового значения
    otsu_threshold_min = 0
    otsu_threshold_max = 255
    
    # Параметры морфологических операций
    structuring_element = cv2.MORPH_RECT
    structuring_element_size = (15, 15)
    
    def __init__(self, debug=False):
        """
        Инициализация сегментатора листьев
        
        :param debug: Флаг для сохранения отладочных изображений
        """
        self.debug = debug
        
        # Создаем директории для результатов и отладки
        self.results_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'images', 
            'results'
        ))
        os.makedirs(self.results_dir, exist_ok=True)
        
        if self.debug:
            self.debug_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                '..', 
                'images', 
                'debug'
            ))
            os.makedirs(self.debug_dir, exist_ok=True)

    def process_directory(self, input_dir):
        """
        Обрабатывает все изображения в указанной директории
        
        :param input_dir: Путь к директории с изображениями
        """
        # Проверяем существование директории
        if not os.path.exists(input_dir):
            print(f"Директория не найдена: {input_dir}")
            return
        
        # Получаем список всех изображений в директории
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Начинаем обработку изображений из директории: {input_dir}")
        
        success_count = 0
        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            if self.process_image(image_path):
                success_count += 1
        
        print(f"Успешно обработано {success_count} из {len(image_files)} изображений")
        print("Обработка завершена.")

    def process_image(self, image_path):
        """
        Обрабатывает одно изображение
        
        :param image_path: Путь к изображению
        :return: True, если обработка прошла успешно, иначе False
        """
        try:
            print(f"Обработка изображения: {image_path}")
            
            # Загружаем изображение
            image = cv2.imread(image_path)
            if image is None:
                print(f"Не удалось загрузить изображение: {image_path}")
                return False
            
            # Находим красную метку на изображении и создаем маску вокруг нее
            marker_position = self.find_red_marker(image)
            if marker_position is None:
                print(f"Не удалось найти красную метку на изображении: {image_path}")
                return False
            
            print(f"Найдена красная метка в позиции: {marker_position}")
            
            # Сегментируем лист
            start_time = t.time()
            segmented_image = self.perform_segmentation(image)
            if segmented_image is None:
                print(f"Не удалось сегментировать лист на изображении: {image_path}")
                return False
            
            # Извлекаем контур листа из сегментированного изображения
            contour = self.extract_leaf_contour(segmented_image)
            if contour is None:
                print(f"Не удалось извлечь контур листа: {image_path}")
                return False
            
            # Рисуем контур на оригинальном изображении
            result_image = image.copy()
            cv2.drawContours(result_image, [contour], 0, (255, 0, 255), 2)
            
            # Отмечаем маркер на изображении
            cv2.circle(result_image, marker_position, 5, (0, 0, 255), -1)
            
            # Сохраняем результат
            result_path = os.path.join(self.results_dir, f'leaf_contour_{os.path.basename(image_path)}')
            cv2.imwrite(result_path, result_image)
            
            end_time = t.time()
            print(f"Сохранен контур листа: {result_path}")
            print(f"Время обработки: {end_time - start_time:.4f} секунд")
            
            return True
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def find_red_marker(self, image):
        """
        Находит красную метку на изображении
        
        :param image: Изображение в формате OpenCV (BGR)
        :return: Координаты (x, y) красной метки или None, если метка не найдена
        """
        try:
            # Преобразуем изображение в HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Определяем диапазон красного цвета в HSV
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            # Создаем маски для обоих диапазонов красного цвета
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            # Объединяем маски
            mask = mask1 + mask2
            
            if self.debug:
                cv2.imwrite(os.path.join(self.debug_dir, 'red_marker_mask.png'), mask)
            
            # Находим контуры на маске
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("Не найдены контуры красной метки")
                return None
            
            # Находим самый большой контур (предположительно, это красная метка)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Проверяем, что контур достаточно большой, чтобы быть меткой
            if cv2.contourArea(largest_contour) < 50:
                print(f"Контур слишком маленький: {cv2.contourArea(largest_contour)}")
                return None
            
            # Находим центр контура
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                print("Момент контура равен нулю")
                return None
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            print(f"Найден центр красной метки: ({cx}, {cy})")
            
            # Отладочное изображение с отмеченной красной меткой
            if self.debug:
                debug_image = image.copy()
                cv2.drawContours(debug_image, [largest_contour], 0, (0, 255, 0), 2)
                cv2.circle(debug_image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(self.debug_dir, 'red_marker_debug.png'), debug_image)
            
            return (cx, cy)
        except Exception as e:
            print(f"Ошибка при поиске красной метки: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def histogram_equalization(self, img):
        """
        Выравнивание гистограммы изображения для улучшения контраста
        
        :param img: Входное изображение
        :return: Изображение с выравненной гистограммой
        """
        return cv2.equalizeHist(img)

    def gaussian_filter(self, img):
        """
        Применяет размытие по Гауссу к изображению
        
        :param img: Входное изображение
        :return: Размытое изображение
        """
        filtered = np.copy(img)
        filtered = cv2.GaussianBlur(img, self.filter_size, self.filter_sigma)
        return filtered

    def create_mask_using_otsu(self, img):
        """
        Создает маску, используя алгоритм порогового значения Оцу
        
        :param img: Входное изображение
        :return: Бинарная маска
        """
        threshold, mask = cv2.threshold(img, 
                                       self.otsu_threshold_min, 
                                       self.otsu_threshold_max,
                                       cv2.THRESH_OTSU)
        x, y = mask.shape
        for i in range(0, x, 1):
            for j in range(0, y, 1):
                if mask[i, j] == self.otsu_threshold_max:
                    mask[i, j] = self.otsu_threshold_min
                else:
                    mask[i, j] = self.otsu_threshold_max 
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'otsu_mask.png'), mask)
        
        return mask
        
    def improve_mask(self, mask):
        """
        Улучшает маску с помощью морфологических операций
        
        :param mask: Входная маска
        :return: Улучшенная маска
        """
        SE = cv2.getStructuringElement(self.structuring_element,
                                       self.structuring_element_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, SE)
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'improved_mask.png'), mask)
        
        return mask
    
    def perform_segmentation(self, img):
        """
        Выполняет сегментацию листа на изображении
        
        :param img: Входное изображение
        :return: Сегментированное изображение, где только лист остается видимым
        """
        img_copy = np.copy(img)
        
        # Находим область вокруг красной метки
        marker_position = self.find_red_marker(img)
        if marker_position is None:
            print("Не удалось найти красную метку для определения цвета листа")
            return None
        
        # Создаем маску для области вокруг метки
        x, y = marker_position
        marker_radius = 30
        marker_area_mask = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(marker_area_mask, marker_position, marker_radius, 255, -1)
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'marker_area_mask_for_color.png'), marker_area_mask)
        
        # Преобразуем изображение в HSV для анализа цвета
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Получаем средний цвет в области вокруг метки
        marker_hsv = cv2.mean(hsv, marker_area_mask)[:3]
        
        # Определяем диапазон цветов на основе среднего цвета в области метки
        h_range = 40
        s_range = 70
        v_range = 70
        
        lower_bound = np.array([
            max(0, marker_hsv[0] - h_range),
            max(0, marker_hsv[1] - s_range),
            max(0, marker_hsv[2] - v_range)
        ], dtype=np.uint8)
        
        upper_bound = np.array([
            min(180, marker_hsv[0] + h_range),
            min(255, marker_hsv[1] + s_range),
            min(255, marker_hsv[2] + v_range)
        ], dtype=np.uint8)
        
        # Создаем маску на основе диапазона цветов
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'color_range_mask.png'), color_mask)
        
        # Применяем морфологические операции для улучшения маски
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'color_mask_improved.png'), color_mask)
        
        # Комбинируем подход на основе цвета с методом Оцу
        blue, green, red = cv2.split(img)
        
        # Используем синий канал для сегментации
        img_blue = self.histogram_equalization(blue)
        img_blue = self.gaussian_filter(img_blue)
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'blue_channel.png'), blue)
            cv2.imwrite(os.path.join(self.debug_dir, 'equalized_blue.png'), img_blue)
        
        mask_blue = self.create_mask_using_otsu(img_blue)
        mask_blue = self.improve_mask(mask_blue)
        
        # Объединяем маски: цветовую и на основе метода Оцу
        combined_mask = cv2.bitwise_or(color_mask, mask_blue)
        
        # Находим компонент, содержащий маркер
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        
        marker_label = labels[y, x] if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] else 0
        
        # Если маркер не попал в сегментированную область, находим ближайший компонент
        if marker_label == 0:
            min_dist = float('inf')
            closest_label = 0
            
            for i in range(1, num_labels):
                dist = np.sqrt((centroids[i][0] - x)**2 + (centroids[i][1] - y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_label = i
            
            if closest_label != 0:
                marker_label = closest_label
        
        # Создаем финальную маску только для компонента с маркером
        final_mask = np.zeros_like(combined_mask)
        
        if marker_label != 0:
            final_mask[labels == marker_label] = 255
            
            # Применяем морфологические операции для улучшения маски
            kernel = np.ones((7, 7), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Заполняем дыры в маске
            final_mask = ndimage.binary_fill_holes(final_mask).astype(np.uint8) * 255
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'final_mask.png'), final_mask)
        
        # Применяем финальную маску к изображению
        segmented = np.copy(img)
        segmented[final_mask == 0] = 0
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'segmented_image.png'), segmented)
        
        return segmented

    def extract_leaf_contour(self, segmented_image):
        """
        Извлекает контур листа из сегментированного изображения
        
        :param segmented_image: Сегментированное изображение
        :return: Контур листа
        """
        try:
            # Преобразуем в оттенки серого
            gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
            
            # Применяем порог, чтобы получить бинарное изображение
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            if self.debug:
                cv2.imwrite(os.path.join(self.debug_dir, 'binary_image.png'), binary)
            
            # Находим контуры
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                print("Контуры не найдены")
                return None
            
            # Находим самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Если контур слишком маленький, возвращаем None
            if cv2.contourArea(largest_contour) < 100:
                print(f"Контур слишком маленький: {cv2.contourArea(largest_contour)}")
                return None
            
            # Сглаживаем контур с помощью аппроксимации
            epsilon = 0.001 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if self.debug:
                # Создаем изображение для визуализации контура
                contour_image = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
                cv2.drawContours(contour_image, [largest_contour], 0, (0, 0, 255), 2)  # Исходный контур (красный)
                cv2.drawContours(contour_image, [approx_contour], 0, (255, 0, 255), 2)  # Сглаженный контур (фиолетовый)
                cv2.imwrite(os.path.join(self.debug_dir, 'contour_comparison.png'), contour_image)
            
            return approx_contour
        except Exception as e:
            print(f"Ошибка при извлечении контура: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Улучшенный анализатор листьев с красными метками')
    parser.add_argument('--dir', type=str, help='Путь к директории с изображениями')
    parser.add_argument('--image', type=str, help='Путь к отдельному изображению')
    parser.add_argument('--debug', action='store_true', help='Сохранять отладочные изображения')
    
    args = parser.parse_args()
    
    # Создаем анализатор листьев
    segmentator = LeafSegmentation(debug=args.debug)
    
    # Определяем путь к изображениям
    if args.image:
        # Обрабатываем одно изображение
        segmentator.process_image(args.image)
    else:
        # Определяем директорию с изображениями
        images_dir = args.dir if args.dir else os.path.abspath(os.path.join(
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
        segmentator.process_directory(images_dir)
        
        print("Обработка завершена.")

if __name__ == "__main__":
    main()
