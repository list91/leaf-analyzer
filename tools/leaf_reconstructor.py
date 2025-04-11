import cv2
import numpy as np
import scipy.stats as stats
from skimage import morphology
from scipy.interpolate import interp1d

class LeafReconstructor:
    def __init__(self, debug=False):
        """
        Инициализация реконструктора листьев
        
        :param debug: Флаг для вывода промежуточных изображений
        """
        self.debug = debug
    
    def _preprocess(self, image):
        """
        Предварительная обработка изображения
        
        :param image: Входное изображение
        :return: Preprocessed image
        """
        # Преобразование в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Адаптивная пороговая обработка
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        return thresh
    
    def _extract_contours(self, binary_image):
        """
        Извлечение контуров листа
        
        :param binary_image: Бинаризованное изображение
        :return: Список контуров
        """
        contours, _ = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Сортировка контуров по площади
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        return contours
    
    def _skeletonize(self, binary_image):
        """
        Скелетизация изображения
        
        :param binary_image: Бинаризованное изображение
        :return: Скелет листа
        """
        # Скелетизация с помощью scikit-image
        skeleton = morphology.skeletonize(binary_image > 0)
        
        return skeleton.astype(np.uint8) * 255
    
    def _probabilistic_reconstruction(self, skeleton, original_contour):
        """
        Вероятностная реконструкция формы листа
        
        :param skeleton: Скелет листа
        :param original_contour: Исходный контур
        :return: Реконструированный контур
        """
        # Извлечение точек скелета
        skeleton_points = np.column_stack(np.where(skeleton > 0))
        
        # Байесовская оценка формы
        def bayesian_shape_prior(points):
            """
            Байесовский метод оценки формы
            """
            # Оценка параметров распределения
            mu = np.mean(points, axis=0)
            cov = np.cov(points.T)
            
            # Многомерное нормальное распределение
            mvn = stats.multivariate_normal(mu, cov)
            
            return mvn
        
        # Создание байесовской модели
        shape_model = bayesian_shape_prior(skeleton_points)
        
        # Генерация новых точек
        reconstructed_points = shape_model.rvs(size=len(original_contour))
        
        return reconstructed_points.astype(np.int32)
    
    def reconstruct_leaf(self, image):
        """
        Основной метод реконструкции листа
        
        :param image: Входное изображение листа
        :return: Реконструированное изображение
        """
        # Предобработка
        binary = self._preprocess(image)
        
        # Извлечение контуров
        contours = self._extract_contours(binary)
        
        if not contours:
            return image
        
        # Берем самый большой контур (основной лист)
        main_contour = contours[0]
        
        # Скелетизация
        skeleton = self._skeletonize(binary)
        
        # Вероятностная реконструкция
        reconstructed_contour = self._probabilistic_reconstruction(
            skeleton, 
            main_contour
        )
        
        # Создаем маску для реконструкции
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [reconstructed_contour], -1, 255, -1)
        
        # Морфологические операции для сглаживания
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Наложение реконструированной маски
        result = cv2.bitwise_and(image, image, mask=mask)
        
        # Отладочная визуализация
        if self.debug:
            cv2.imshow('Skeleton', skeleton)
            cv2.imshow('Reconstructed Mask', mask)
            cv2.imshow('Result', result)
            cv2.waitKey(0)
        
        return result

def process_directory(input_dir, output_dir):
    """
    Пакетная обработка изображений в директории
    
    :param input_dir: Директория с исходными изображениями
    :param output_dir: Директория для сохранения результатов
    """
    import os
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем реконструктор
    reconstructor = LeafReconstructor(debug=False)
    
    # Обработка каждого изображения
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f'reconstructed_{filename}')
            
            # Загрузка и реконструкция
            image = cv2.imread(input_path)
            result = reconstructor.reconstruct_leaf(image)
            
            # Сохранение результата
            cv2.imwrite(output_path, result)
            print(f"Обработано: {filename}")

if __name__ == "__main__":
    # Пример использования
    input_directory = r"c:\sts\projects\diplom\plant-preprocesing\images\normal_plants"
    output_directory = r"c:\sts\projects\diplom\plant-preprocesing\images\reconstructed"
    
    process_directory(input_directory, output_directory)
