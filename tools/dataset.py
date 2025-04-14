import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from preprocessing import LeafPreprocessor
from logger import dataset_logger as logger

class LeafDataset(Dataset):
    def __init__(self, image_paths, labels):
        logger.info(f"Инициализация LeafDataset с {len(image_paths)} изображениями")
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = LeafPreprocessor()
        self.preprocessor.set_debug(False)  # Отключаем подробные логи

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # Используем новый препроцессор
        image = self.preprocessor.preprocess_for_training(image_path)
        label = self.labels[idx]
        return image, label

def prepare_data_loaders(data_dir, batch_size=8, test_size=0.2, val_size=0.1, 
                        disease_files=None, normal_files=None):
    """
    Подготовка DataLoader'ов для обучения, валидации и тестирования
    """
    logger.info(f"Подготовка данных из директории: {data_dir}")
    logger.info(f"Параметры: batch_size={batch_size}, test_size={test_size}, val_size={val_size}")
    
    # Сбор путей к изображениям и меток
    disease_dir = os.path.join(data_dir, 'disease_plants')
    normal_dir = os.path.join(data_dir, 'normal_plants')
    
    logger.info("Сканирование директорий с изображениями...")
    
    if disease_files is None:
        disease_files = [f for f in os.listdir(disease_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if normal_files is None:
        normal_files = [f for f in os.listdir(normal_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    disease_images = [(os.path.join(disease_dir, f), 1) for f in disease_files]
    normal_images = [(os.path.join(normal_dir, f), 0) for f in normal_files]
    
    all_images = disease_images + normal_images
    image_paths, labels = zip(*all_images)
    
    logger.info(f"Найдено всего изображений: {len(all_images)}")
    logger.info(f"Больных растений: {len(disease_images)}")
    logger.info(f"Здоровых растений: {len(normal_images)}")
    
    # Разделение на train, val и test
    logger.info("Разделение данных на выборки...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Из оставшихся данных выделяем валидационную выборку
    train_size = 1 - (val_size / (1 - test_size))
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, train_size=train_size, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Размер обучающей выборки: {len(X_train)}")
    logger.info(f"Размер валидационной выборки: {len(X_val)}")
    logger.info(f"Размер тестовой выборки: {len(X_test)}")
    
    # Создание DataLoader'ов
    logger.info("Создание DataLoader'ов...")
    train_dataset = LeafDataset(X_train, y_train)
    val_dataset = LeafDataset(X_val, y_val)
    test_dataset = LeafDataset(X_test, y_test)
    
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
    
    logger.info("DataLoader'ы успешно созданы")
    return train_loader, val_loader, test_loader
