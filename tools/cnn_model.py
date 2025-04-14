import torch
import torch.nn as nn
import torchvision.models as models

class LeafCNN(nn.Module):
    """
    Сверточная нейронная сеть для классификации листьев растений
    Использует предобученную MobileNetV3 в качестве основы
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(LeafCNN, self).__init__()
        
        # Используем MobileNetV3-Small как базовую модель (легкая и эффективная)
        self.model = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Заменяем последний слой на наш классификатор
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def save_model(model, optimizer, epoch, metrics, path):
    """
    Сохранение модели и состояния обучения
    
    Args:
        model: модель для сохранения
        optimizer: оптимизатор
        epoch: текущая эпоха
        metrics: словарь с метриками (loss, accuracy и т.д.)
        path: путь для сохранения
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)
    
def load_model(path, device='cpu'):
    """
    Загрузка модели из файла
    
    Args:
        path: путь к файлу модели
        device: устройство для загрузки модели ('cpu' или 'cuda')
        
    Returns:
        model: загруженная модель
        checkpoint: словарь с состоянием обучения
    """
    checkpoint = torch.load(path, map_location=device)
    
    model = LeafCNN(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint
