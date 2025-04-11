import os
import cv2
import numpy as np
import argparse
from PIL import Image, ImageDraw
from scipy import interpolate

class LeafAnalyzer:
    """
    Класс для анализа и сегментации листьев на изображениях с красными метками
    """
    
    def __init__(self, method='grabcut', debug=False, segmentation_params=None):
        """
        Инициализация анализатора листьев
        
        :param method: Метод сегментации ('grabcut', 'watershed', 'color', 'all')
        :param debug: Флаг для сохранения отладочных изображений
        :param segmentation_params: Словарь с параметрами сегментации
        """
        self.method = method
        self.debug = debug
        
        # Параметры сегментации по умолчанию
        self.segmentation_params = {
            'h_range': 40,
            's_range': 70,
            'v_range': 70,
            'contour_points': 1000
        }
        
        # Обновляем параметры, если они переданы
        if segmentation_params:
            self.segmentation_params.update(segmentation_params)
        
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
    
    def segment_leaf_grabcut(self, image, marker_position, rect_size=150):
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
        
        # Адаптивный размер прямоугольника в зависимости от размера изображения
        height, width = image.shape[:2]
        adaptive_size = int(min(width, height) / 1.5)  # Преобразуем в целое число
        rect_size = min(rect_size * 3, adaptive_size)
        
        half_size = rect_size // 2
        
        # Убедимся, что прямоугольник не выходит за границы изображения
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
        marker_radius = max(30, rect_size // 8)
        cv2.circle(mask, marker_position, marker_radius, cv2.GC_FGD, -1)
        
        # Определяем цвет объекта на основе области вокруг метки
        # Создаем маску для области вокруг метки
        marker_area_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(marker_area_mask, marker_position, marker_radius * 2, 255, -1)
        
        # Преобразуем изображение в HSV для анализа цвета
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Получаем средний цвет в области вокруг метки
        marker_hsv = cv2.mean(hsv, marker_area_mask)[:3]
        
        # Определяем диапазон цветов на основе среднего цвета в области метки
        h_range = self.segmentation_params['h_range']  
        s_range = self.segmentation_params['s_range']
        v_range = self.segmentation_params['v_range']
        
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
        
        # Отмечаем области с похожим цветом как вероятный передний план
        mask[color_mask > 0] = cv2.GC_PR_FGD
        
        # Запускаем GrabCut с большим количеством итераций
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_MASK)
        
        # Создаем маску, где 0 и 2 - фон, 1 и 3 - передний план
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Применяем морфологические операции для улучшения маски
        kernel = np.ones((5, 5), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        
        # Находим все контуры в маске
        contours, _ = cv2.findContours(mask2.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Если контуры найдены, находим самый большой и заполняем его
        if contours:
            # Сортируем контуры по площади (от большего к меньшему)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Находим контур, содержащий маркер
            marker_contour = None
            for contour in contours:
                if cv2.pointPolygonTest(contour, marker_position, False) >= 0:
                    marker_contour = contour
                    break
            
            # Если контур с маркером не найден, берем самый большой
            if marker_contour is None and contours:
                marker_contour = contours[0]
            
            if marker_contour is not None:
                # Создаем новую маску и заполняем контур
                final_mask = np.zeros_like(mask2)
                cv2.drawContours(final_mask, [marker_contour], 0, 1, -1)
                mask2 = final_mask
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'leaf_mask_grabcut.png'), mask2 * 255)
            cv2.imwrite(os.path.join(self.debug_dir, 'marker_area_mask.png'), marker_area_mask)
            cv2.imwrite(os.path.join(self.debug_dir, 'color_mask.png'), color_mask)
        
        return mask2
    
    def segment_leaf_watershed(self, image, marker_position):
        """
        Сегментирует лист с использованием алгоритма Watershed
        
        :param image: Изображение в формате OpenCV (BGR)
        :param marker_position: Координаты (x, y) красной метки
        :return: Маска сегментированного листа
        """
        # Преобразуем изображение в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применяем размытие по Гауссу
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Применяем пороговую обработку для получения бинарного изображения
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Выполняем морфологическое открытие для удаления шума
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Определяем фоновую область
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Определяем область переднего плана с помощью дистанционного преобразования
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        
        # Определяем неизвестную область
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Маркируем области для watershed
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Определяем цвет объекта на основе области вокруг метки
        x, y = marker_position
        marker_radius = 30
        marker_area_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(marker_area_mask, marker_position, marker_radius, 255, -1)
        
        # Преобразуем изображение в HSV для анализа цвета
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Получаем средний цвет в области вокруг метки
        marker_hsv = cv2.mean(hsv, marker_area_mask)[:3]
        
        # Определяем диапазон цветов на основе среднего цвета в области метки
        h_range = self.segmentation_params['h_range']  
        s_range = self.segmentation_params['s_range']
        v_range = self.segmentation_params['v_range']
        
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
        
        # Отмечаем маркер как передний план
        marker_region = np.zeros_like(markers, dtype=np.int32)
        cv2.circle(marker_region, marker_position, marker_radius, 2, -1)
        markers = np.maximum(markers, marker_region)
        
        # Применяем watershed
        cv2.watershed(image, markers)
        
        # Создаем маску для листа (области, помеченные как 2)
        leaf_mask = np.zeros(markers.shape, dtype=np.uint8)
        leaf_mask[markers == 2] = 255
        
        # Комбинируем с цветовой маской для уточнения результата
        leaf_mask = cv2.bitwise_and(leaf_mask, color_mask)
        
        # Находим все контуры в маске
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Если контуры найдены, находим самый большой и заполняем его
        if contours:
            # Сортируем контуры по площади (от большего к меньшему)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Находим контур, содержащий маркер
            marker_contour = None
            for contour in contours:
                if cv2.pointPolygonTest(contour, marker_position, False) >= 0:
                    marker_contour = contour
                    break
            
            # Если контур с маркером не найден, берем самый большой
            if marker_contour is None and contours:
                marker_contour = contours[0]
            
            if marker_contour is not None:
                # Создаем новую маску и заполняем контур
                final_mask = np.zeros_like(leaf_mask)
                cv2.drawContours(final_mask, [marker_contour], 0, 255, -1)
                leaf_mask = final_mask
        
        # Применяем морфологические операции для улучшения маски
        kernel = np.ones((5, 5), np.uint8)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'leaf_mask_watershed.png'), leaf_mask)
            cv2.imwrite(os.path.join(self.debug_dir, 'watershed_markers.png'), (markers * 30).astype(np.uint8))
            cv2.imwrite(os.path.join(self.debug_dir, 'color_mask_watershed.png'), color_mask)
            cv2.imwrite(os.path.join(self.debug_dir, 'marker_area_mask_watershed.png'), marker_area_mask)
        
        return leaf_mask
    
    def segment_leaf_color(self, image, marker_position):
        """
        Сегментирует лист с использованием цветовой сегментации
        
        :param image: Изображение в формате OpenCV (BGR)
        :param marker_position: Координаты (x, y) красной метки
        :return: Маска сегментированного листа
        """
        # Преобразуем изображение в HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Создаем маску для области вокруг метки
        x, y = marker_position
        marker_radius = 30
        marker_area_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(marker_area_mask, marker_position, marker_radius, 255, -1)
        
        # Получаем средний цвет в области вокруг метки
        marker_hsv = cv2.mean(hsv, marker_area_mask)[:3]
        
        # Определяем диапазон цветов на основе среднего цвета в области метки
        h_range = self.segmentation_params['h_range']  
        s_range = self.segmentation_params['s_range']
        v_range = self.segmentation_params['v_range']
        
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
        
        # Применяем морфологические операции для улучшения маски
        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'marker_area_mask.png'), marker_area_mask)
            cv2.imwrite(os.path.join(self.debug_dir, 'color_mask.png'), color_mask)
        
        # Находим связные компоненты в маске цвета
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        
        # Находим компонент, содержащий маркер
        marker_label = 0
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
            marker_label = labels[y, x]
        
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
        
        # Создаем маску только для компонента с маркером
        leaf_mask = np.zeros_like(color_mask, dtype=np.uint8)
        
        if marker_label != 0:
            leaf_mask[labels == marker_label] = 255
            
            # Находим все контуры в маске
            contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Если контуры найдены, находим самый большой и заполняем его
            if contours:
                # Сортируем контуры по площади (от большего к меньшему)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Берем самый большой контур
                largest_contour = contours[0]
                
                # Создаем новую маску и заполняем контур
                final_mask = np.zeros_like(leaf_mask)
                cv2.drawContours(final_mask, [largest_contour], 0, 255, -1)
                leaf_mask = final_mask
        
        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, 'leaf_mask_color.png'), leaf_mask)
        
        return leaf_mask
    
    def extract_leaf_contour(self, mask):
        """
        Извлекает контур листа из маски
        
        :param mask: Бинарная маска листа
        :return: Контур листа
        """
        # Преобразуем маску в формат uint8 и масштабируем до 255
        if mask.dtype != np.uint8:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.copy()
        
        # Применяем морфологические операции для улучшения контура
        kernel = np.ones((7, 7), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=5)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Применяем размытие по Гауссу для сглаживания краев
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (9, 9), 0)
        
        # Применяем пороговую обработку для получения четкой маски
        _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Находим самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Если контур слишком маленький, возвращаем None
        if cv2.contourArea(largest_contour) < 100:
            return None
        
        # Сглаживаем контур с помощью аппроксимации
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Применяем сплайн-интерполяцию для дополнительного сглаживания
        # Преобразуем контур в список точек
        points = approx_contour.reshape(-1, 2)
        
        # Если точек слишком мало, возвращаем исходный контур
        if len(points) < 5:
            return largest_contour
        
        # Создаем замкнутый сплайн
        tck, u = interpolate.splprep([points[:, 0], points[:, 1]], s=0, per=True)
        
        # Генерируем новые точки вдоль сплайна (больше точек для более гладкого контура)
        u_new = np.linspace(0, 1.0, self.segmentation_params['contour_points'])
        smooth_points = interpolate.splev(u_new, tck)
        
        # Преобразуем обратно в формат контура OpenCV
        smooth_contour = np.column_stack((smooth_points[0], smooth_points[1])).astype(np.int32)
        smooth_contour = smooth_contour.reshape(-1, 1, 2)
        
        if self.debug:
            # Создаем изображение для визуализации контура
            contour_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(contour_image, [largest_contour], 0, (0, 0, 255), 2)  # Исходный контур (красный)
            cv2.drawContours(contour_image, [smooth_contour], 0, (255, 0, 255), 2)  # Сглаженный контур (фиолетовый)
            cv2.imwrite(os.path.join(self.debug_dir, 'contour_comparison.png'), contour_image)
        
        return smooth_contour
    
    def process_image(self, image_path):
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
            
            # Предварительная обработка изображения для улучшения сегментации
            # Увеличиваем контраст
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            if self.debug:
                cv2.imwrite(os.path.join(self.debug_dir, f'enhanced_{os.path.basename(image_path)}'), enhanced_image)
            
            # Находим красную метку
            marker_position = self.find_red_marker(image)
            if marker_position is None:
                print(f"Красная метка не найдена на изображении: {image_path}")
                return False
            
            print(f"Найдена красная метка: {marker_position}")
            
            # Отмечаем найденную метку на изображении для отладки
            if self.debug:
                debug_image = image.copy()
                cv2.circle(debug_image, marker_position, 10, (0, 255, 255), -1)
                cv2.imwrite(os.path.join(self.debug_dir, f'marker_{os.path.basename(image_path)}'), debug_image)
            
            # Сегментируем лист с использованием выбранного метода
            if self.method == 'grabcut':
                leaf_mask = self.segment_leaf_grabcut(enhanced_image, marker_position)
            elif self.method == 'watershed':
                leaf_mask = self.segment_leaf_watershed(enhanced_image, marker_position)
            elif self.method == 'color':
                leaf_mask = self.segment_leaf_color(enhanced_image, marker_position)
            elif self.method == 'all':
                # Создаем копии изображения для каждого метода
                grabcut_image = result_image.copy()
                watershed_image = result_image.copy()
                color_image = result_image.copy()
                
                # Запускаем все методы
                grabcut_mask = self.segment_leaf_grabcut(enhanced_image, marker_position)
                watershed_mask = self.segment_leaf_watershed(enhanced_image, marker_position)
                color_mask = self.segment_leaf_color(enhanced_image, marker_position)
                
                # Извлекаем контуры
                grabcut_contour = self.extract_leaf_contour(grabcut_mask)
                watershed_contour = self.extract_leaf_contour(watershed_mask)
                color_contour = self.extract_leaf_contour(color_mask)
                
                # Рисуем контуры на соответствующих изображениях
                if grabcut_contour is not None:
                    cv2.drawContours(grabcut_image, [grabcut_contour], 0, (255, 0, 255), 3)
                    output_path = os.path.join(self.results_dir, f'leaf_contour_grabcut_{os.path.basename(image_path)}')
                    cv2.imwrite(output_path, grabcut_image)
                    print(f"Результат GrabCut сохранен: {output_path}")
                
                if watershed_contour is not None:
                    cv2.drawContours(watershed_image, [watershed_contour], 0, (255, 0, 255), 3)
                    output_path = os.path.join(self.results_dir, f'leaf_contour_watershed_{os.path.basename(image_path)}')
                    cv2.imwrite(output_path, watershed_image)
                    print(f"Результат Watershed сохранен: {output_path}")
                
                if color_contour is not None:
                    cv2.drawContours(color_image, [color_contour], 0, (255, 0, 255), 3)
                    output_path = os.path.join(self.results_dir, f'leaf_contour_color_{os.path.basename(image_path)}')
                    cv2.imwrite(output_path, color_image)
                    print(f"Результат Color сохранен: {output_path}")
                
                # Создаем сравнительное изображение
                if self.debug:
                    # Создаем изображение с масками
                    masks_image = np.zeros((image.shape[0], image.shape[1] * 3, 3), dtype=np.uint8)
                    
                    # Преобразуем маски в цветные изображения для визуализации
                    grabcut_mask_color = cv2.cvtColor(grabcut_mask * 255, cv2.COLOR_GRAY2BGR)
                    watershed_mask_color = cv2.cvtColor(watershed_mask * 255, cv2.COLOR_GRAY2BGR)
                    color_mask_color = cv2.cvtColor(color_mask * 255, cv2.COLOR_GRAY2BGR)
                    
                    # Размещаем маски рядом
                    masks_image[0:image.shape[0], 0:image.shape[1]] = grabcut_mask_color
                    masks_image[0:image.shape[0], image.shape[1]:image.shape[1]*2] = watershed_mask_color
                    masks_image[0:image.shape[0], image.shape[1]*2:image.shape[1]*3] = color_mask_color
                    
                    # Добавляем подписи
                    cv2.putText(masks_image, "GrabCut", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(masks_image, "Watershed", (image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(masks_image, "Color", (image.shape[1] * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Сохраняем сравнительное изображение
                    comparison_path = os.path.join(self.debug_dir, f'comparison_masks_{os.path.basename(image_path)}')
                    cv2.imwrite(comparison_path, masks_image)
                    print(f"Сравнение масок сохранено: {comparison_path}")
                
                return True
            else:
                print(f"Неизвестный метод сегментации: {self.method}")
                return False
            
            # Извлекаем контур листа
            leaf_contour = self.extract_leaf_contour(leaf_mask)
            if leaf_contour is None:
                print(f"Не удалось извлечь контур листа: {image_path}")
                return False
            
            # Рисуем контур на изображении фиолетовым цветом
            cv2.drawContours(result_image, [leaf_contour], 0, (255, 0, 255), 3)
            
            # Сохраняем результат с тем же именем файла, что и исходное изображение
            output_path = os.path.join(
                self.results_dir, 
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
    
    def process_directory(self, directory):
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
            if self.process_image(image_path):
                success_count += 1
        
        print(f"Успешно обработано {success_count} из {len(image_files)} изображений")

def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Анализатор листьев с красными метками')
    parser.add_argument('--dir', type=str, help='Путь к директории с изображениями')
    parser.add_argument('--image', type=str, help='Путь к отдельному изображению')
    parser.add_argument('--method', type=str, default='grabcut', 
                        choices=['grabcut', 'watershed', 'color', 'all'],
                        help='Метод сегментации (grabcut, watershed, color, all)')
    parser.add_argument('--debug', action='store_true', help='Сохранять отладочные изображения')
    parser.add_argument('--color-range', type=int, default=40, 
                        help='Диапазон оттенка цвета (H) для сегментации (по умолчанию 40)')
    parser.add_argument('--sat-range', type=int, default=70, 
                        help='Диапазон насыщенности (S) для сегментации (по умолчанию 70)')
    parser.add_argument('--val-range', type=int, default=70, 
                        help='Диапазон яркости (V) для сегментации (по умолчанию 70)')
    parser.add_argument('--contour-points', type=int, default=1000, 
                        help='Количество точек для сглаженного контура (по умолчанию 1000)')
    
    args = parser.parse_args()
    
    # Создаем словарь параметров сегментации
    segmentation_params = {
        'h_range': args.color_range,
        's_range': args.sat_range,
        'v_range': args.val_range,
        'contour_points': args.contour_points
    }
    
    # Если выбран метод "all", запускаем все методы последовательно
    if args.method == 'all':
        methods = ['grabcut', 'watershed', 'color']
        for method in methods:
            print(f"\nЗапуск метода: {method}")
            analyzer = LeafAnalyzer(method=method, debug=args.debug, segmentation_params=segmentation_params)
            
            # Определяем путь к изображениям
            if args.image:
                # Обрабатываем одно изображение
                analyzer.process_image(args.image)
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
                print(f"Начинаем обработку изображений из директории: {images_dir}")
                analyzer.process_directory(images_dir)
        
        print("\nОбработка всеми методами завершена.")
    else:
        # Создаем анализатор листьев
        analyzer = LeafAnalyzer(method=args.method, debug=args.debug, segmentation_params=segmentation_params)
        
        # Определяем путь к изображениям
        if args.image:
            # Обрабатываем одно изображение
            analyzer.process_image(args.image)
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
            print(f"Начинаем обработку изображений из директории: {images_dir}")
            analyzer.process_directory(images_dir)
            
            print("Обработка завершена.")
        
if __name__ == "__main__":
    main()
