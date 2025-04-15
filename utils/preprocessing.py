import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from utils.logger import preprocessing_logger as logger

class LeafPreprocessor:
    def __init__(self):
        logger.info("Инициализация LeafPreprocessor")
        self.debug = False  # Отключаем подробные логи по умолчанию
        # Изменяем нормализацию, чтобы учитывать только пиксели листа
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def set_debug(self, debug):
        """Включение/выключение подробных логов"""
        self.debug = debug

    def create_leaf_mask(self, image):
        """
        Создание маски листа (не-черного фона)
        
        Args:
            image: PIL Image или numpy array
        Returns:
            leaf_mask: бинарная маска листа
        """
        if self.debug:
            logger.debug("Создание маски листа")
        
        # Конвертируем в numpy array если получили PIL Image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Конвертируем в HSV для лучшей сегментации
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Создаем маску для выделения не-черных пикселей
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Инвертируем маску, чтобы получить маску листа
        leaf_mask = cv2.bitwise_not(black_mask)
        
        # Применяем морфологические операции для улучшения маски
        kernel = np.ones((3,3), np.uint8)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        
        return leaf_mask

    def normalize_leaf(self, image_tensor, mask_tensor):
        """
        Нормализация только пикселей листа, игнорируя фон
        
        Args:
            image_tensor: тензор изображения [C, H, W]
            mask_tensor: тензор маски [H, W]
        Returns:
            normalized_tensor: нормализованный тензор
        """
        if self.debug:
            logger.debug("Нормализация пикселей листа")
        
        # Создаем маску для каждого канала
        mask_3d = mask_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # Вычисляем среднее и стд только для пикселей листа
        means = []
        stds = []
        for channel in range(3):
            leaf_pixels = image_tensor[channel][mask_tensor > 0]
            if len(leaf_pixels) > 0:
                means.append(leaf_pixels.mean().item())
                stds.append(leaf_pixels.std().item())
            else:
                means.append(0)
                stds.append(1)
        
        # Нормализуем только пиксели листа
        normalized = image_tensor.clone()
        for channel in range(3):
            leaf_pixels_mask = mask_3d[channel] > 0
            if leaf_pixels_mask.any():
                normalized[channel][leaf_pixels_mask] = (normalized[channel][leaf_pixels_mask] - means[channel]) / (stds[channel] if stds[channel] > 0 else 1)
        
        return normalized

    def preprocess_for_training(self, image_path):
        """
        Предобработка изображения для обучения, игнорируя черный фон
        
        Args:
            image_path: путь к изображению
        Returns:
            tensor: тензор для подачи в модель
        """
        if self.debug:
            logger.info(f"Начало предобработки изображения: {image_path}")
        
        # Загружаем изображение
        if self.debug:
            logger.debug("Загрузка изображения")
        image = Image.open(image_path).convert('RGB')
        
        # Создаем маску листа
        leaf_mask = self.create_leaf_mask(image)
        
        # Применяем базовые преобразования
        if self.debug:
            logger.debug("Применение преобразований")
        image_tensor = self.base_transform(image)
        
        # Преобразуем маску в тензор того же размера
        mask_tensor = torch.from_numpy(cv2.resize(leaf_mask, (224, 224))) > 0
        
        # Нормализуем только пиксели листа
        normalized_tensor = self.normalize_leaf(image_tensor, mask_tensor)
        
        # Устанавливаем пиксели фона в 0
        normalized_tensor[:, ~mask_tensor] = 0
        
        if self.debug:
            logger.info("Предобработка завершена")
        return normalized_tensor

    def preprocess_batch(self, image_paths):
        """
        Предобработка батча изображений
        
        Args:
            image_paths: список путей к изображениям
        Returns:
            tensors: батч тензоров
        """
        if self.debug:
            logger.info(f"Начало предобработки батча из {len(image_paths)} изображений")
        tensors = []
        for i, path in enumerate(image_paths):
            if self.debug:
                logger.debug(f"Обработка изображения {i+1}/{len(image_paths)}")
            tensor = self.preprocess_for_training(path)
            tensors.append(tensor)
        if self.debug:
            logger.info("Предобработка батча завершена")
        return torch.stack(tensors)

def visualize_preprocessing(image_path, save_path=None):
    """
    Визуализация этапов предобработки изображения
    
    Args:
        image_path: путь к исходному изображению
        save_path: путь для сохранения результата (опционально)
    """
    if self.debug:
        logger.info(f"Начало визуализации предобработки: {image_path}")
    preprocessor = LeafPreprocessor()
    
    # Загружаем оригинальное изображение
    if self.debug:
        logger.debug("Загрузка оригинального изображения")
    original = Image.open(image_path).convert('RGB')
    
    # Получаем маску листа
    if self.debug:
        logger.debug("Создание маски листа")
    leaf_mask = preprocessor.create_leaf_mask(original)
    
    # Применяем маску к оригинальному изображению
    if self.debug:
        logger.debug("Применение маски")
    result = cv2.bitwise_and(np.array(original), np.array(original), mask=leaf_mask)
    
    # Конвертируем обратно в PIL Image
    result_image = Image.fromarray(result)
    
    # Сохраняем результат
    if save_path:
        if self.debug:
            logger.info(f"Сохранение результата в: {save_path}")
        result_image.save(save_path)
    
    if self.debug:
        logger.info("Визуализация завершена")
    return result_image
