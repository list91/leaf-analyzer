import torch
import torch.nn as nn
import torchvision.models as models
from utils.logger import model_logger as logger

class LeafClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        logger.info("Инициализация LeafClassifier")
        logger.info(f"Параметры: num_classes={num_classes}, pretrained={pretrained}")
        
        # Используем MobileNetV3-Small как базовую модель
        logger.info("Загрузка базовой модели MobileNetV3-Small")
        self.model = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Заменяем последний слой на наш классификатор
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        logger.info(f"Изменен последний слой: in_features={in_features}, out_features={num_classes}")
        
    def forward(self, x):
        return self.model(x)

def create_model(device='cpu'):
    """
    Создание и инициализация модели
    """
    logger.info(f"Создание модели для устройства: {device}")
    model = LeafClassifier(num_classes=2)
    model = model.to(device)
    return model

def save_checkpoint(model, optimizer, epoch, loss, accuracy, path):
    """
    Сохранение чекпоинта модели
    """
    logger.info(f"Сохранение чекпоинта в {path}")
    logger.info(f"Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, path)
    logger.info("Чекпоинт успешно сохранен")

def load_checkpoint(model, optimizer, path):
    """
    Загрузка чекпоинта модели
    """
    logger.info(f"Загрузка чекпоинта из {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    
    logger.info(f"Загружен чекпоинт: Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    return epoch, loss, accuracy
