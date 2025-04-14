import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

class LeafDataset(Dataset):
    """
    Датасет для изображений листьев растений
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or self._default_transform()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Применение трансформаций
        if self.transform:
            image = self.transform(image)
            
        # Получение метки
        label = self.labels[idx]
        
        return image, label
    
    @staticmethod
    def _default_transform():
        """
        Стандартные преобразования для изображений
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def create_leaf_mask(image):
    """
    Создание маски листа, отделяющей его от фона
    
    Args:
        image: изображение в формате PIL или numpy array
    Returns:
        leaf_mask: бинарная маска листа
    """
    # Конвертируем в numpy array если получили PIL Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Конвертируем в HSV для лучшей сегментации
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Создаем маску для зеленого цвета (для листьев)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Создаем маску для цвета кожи (для исключения рук)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 150, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Объединяем маски: берем зеленую маску и исключаем кожу
    leaf_mask = green_mask & ~skin_mask
    
    # Применяем морфологические операции для улучшения маски
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    
    return leaf_mask

class AdvancedLeafDataset(LeafDataset):
    """
    Расширенный датасет с предобработкой изображений листьев
    """
    def __getitem__(self, idx):
        # Загрузка изображения
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Создаем маску листа
        leaf_mask = create_leaf_mask(image)
        
        # Применяем маску к изображению
        image_np = np.array(image)
        masked_image = cv2.bitwise_and(image_np, image_np, mask=leaf_mask)
        
        # Конвертируем обратно в PIL
        masked_image_pil = Image.fromarray(masked_image)
        
        # Применение трансформаций
        if self.transform:
            masked_image_pil = self.transform(masked_image_pil)
            
        # Получение метки
        label = self.labels[idx]
        
        return masked_image_pil, label

def prepare_data_loaders(data_dir, batch_size=16, test_size=0.2, val_size=0.1, advanced_preprocessing=True, max_images_per_class=100):
    """
    Подготовка DataLoader'ов для обучения, валидации и тестирования
    
    Args:
        data_dir: директория с данными
        batch_size: размер батча
        test_size: доля тестовой выборки
        val_size: доля валидационной выборки
        advanced_preprocessing: использовать ли расширенную предобработку
        max_images_per_class: максимальное количество изображений на класс
        
    Returns:
        train_loader, val_loader, test_loader: загрузчики данных
    """
    # Сбор путей к изображениям и меток
    disease_dir = os.path.join(data_dir, 'disease_plants')
    normal_dir = os.path.join(data_dir, 'normal_plants')
    
    disease_files = [f for f in os.listdir(disease_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    normal_files = [f for f in os.listdir(normal_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Ограничиваем количество изображений для каждого класса
    if max_images_per_class > 0:
        print(f"Ограничение набора данных: максимум {max_images_per_class} изображений на класс")
        
        # Перемешиваем списки файлов для случайного выбора
        import random
        random.seed(42)  # Для воспроизводимости
        random.shuffle(disease_files)
        random.shuffle(normal_files)
        
        disease_files = disease_files[:max_images_per_class]
        normal_files = normal_files[:max_images_per_class]
        
    print(f"Будет использовано {len(disease_files)} изображений больных растений")
    print(f"Будет использовано {len(normal_files)} изображений здоровых растений")
    
    disease_images = [(os.path.join(disease_dir, f), 1) for f in disease_files]
    normal_images = [(os.path.join(normal_dir, f), 0) for f in normal_files]
    
    all_images = disease_images + normal_images
    image_paths, labels = zip(*all_images)
    
    # Разделение на train, val и test
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Из оставшихся данных выделяем валидационную выборку
    train_size = 1 - (val_size / (1 - test_size))
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, train_size=train_size, random_state=42, stratify=y_temp
    )
    
    # Определяем трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Выбираем класс датасета в зависимости от параметра advanced_preprocessing
    dataset_class = AdvancedLeafDataset if advanced_preprocessing else LeafDataset
    
    # Создание датасетов
    train_dataset = dataset_class(X_train, y_train, transform=transform)
    
    # Для валидации и теста не используем аугментацию
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_dataset = dataset_class(X_val, y_val, transform=val_transform)
    test_dataset = dataset_class(X_test, y_test, transform=val_transform)
    
    # Создание DataLoader'ов
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
