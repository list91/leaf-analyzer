import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

from cnn_model import LeafCNN, save_model, load_model
from cnn_dataset import prepare_data_loaders

class TrainingMonitor:
    """
    Класс для мониторинга процесса обучения
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.start_time = time.time()
        
    def update(self, train_loss, train_acc, val_loss, val_acc):
        """Обновление метрик"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
    def plot_metrics(self, save_path=None):
        """Построение графиков метрик"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График потерь
        ax1.plot(self.train_losses, label='Потери на обучении')
        ax1.plot(self.val_losses, label='Потери на валидации')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Потери')
        ax1.set_title('Динамика функции потерь')
        ax1.legend()
        ax1.grid(True)
        
        # График точности
        ax2.plot(self.train_accs, label='Точность на обучении')
        ax2.plot(self.val_accs, label='Точность на валидации')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('Точность (%)')
        ax2.set_title('Динамика точности')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Обучение модели на одной эпохе
    
    Args:
        model: модель
        train_loader: загрузчик данных для обучения
        criterion: функция потерь
        optimizer: оптимизатор
        device: устройство для вычислений
        
    Returns:
        train_loss, train_acc: средние потери и точность на эпохе
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()
        
        # Статистика
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Вывод прогресса
        print(f"\rОбработано изображений: {total}/{len(train_loader.dataset)}", end="")
    
    # Вычисляем средние значения
    train_loss = running_loss / total
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

def validate(model, val_loader, criterion, device):
    """
    Валидация модели
    
    Args:
        model: модель
        val_loader: загрузчик данных для валидации
        criterion: функция потерь
        device: устройство для вычислений
        
    Returns:
        val_loss, val_acc: средние потери и точность на валидации
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Статистика
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Вычисляем средние значения
    val_loss = running_loss / total
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def test_model(model, test_loader, device):
    """
    Тестирование модели
    
    Args:
        model: модель
        test_loader: загрузчик данных для тестирования
        device: устройство для вычислений
        
    Returns:
        test_acc: точность на тестовой выборке
        confusion_matrix: матрица ошибок
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Прямой проход
            outputs = model(inputs)
            
            # Статистика
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Сохраняем предсказания и метки для матрицы ошибок
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Вычисляем точность
    test_acc = 100. * correct / total
    
    # Вычисляем матрицу ошибок
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return test_acc, cm

def train_and_evaluate(data_dir, num_epochs=20, batch_size=16, learning_rate=1e-4, 
                      advanced_preprocessing=True, checkpoint_dir=None, max_images_per_class=100):
    """
    Полный цикл обучения и оценки модели
    
    Args:
        data_dir: директория с данными
        num_epochs: количество эпох обучения
        batch_size: размер батча
        learning_rate: скорость обучения
        advanced_preprocessing: использовать ли расширенную предобработку
        checkpoint_dir: директория для сохранения чекпоинтов
        max_images_per_class: максимальное количество изображений на класс
        
    Returns:
        model: обученная модель
        monitor: монитор обучения с метриками
    """
    # Определение устройства
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство: {device}")
    
    # Подготовка данных
    print("Подготовка данных...")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_dir, batch_size=batch_size, 
        advanced_preprocessing=advanced_preprocessing,
        max_images_per_class=max_images_per_class
    )
    print(f"Размер обучающей выборки: {len(train_loader.dataset)}")
    print(f"Размер валидационной выборки: {len(val_loader.dataset)}")
    print(f"Размер тестовой выборки: {len(test_loader.dataset)}")
    
    # Создание модели
    print("Создание модели...")
    model = LeafCNN(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Создание директории для сохранения чекпоинтов
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Инициализация монитора обучения
    monitor = TrainingMonitor()
    
    # Параметры для early stopping
    best_val_acc = 0.0
    early_stopping_counter = 0
    early_stopping_patience = 10
    
    # Цикл обучения
    print(f"Начало обучения на {num_epochs} эпохах...")
    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch+1}/{num_epochs}")
        
        # Обучение на одной эпохе
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Валидация
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Вывод результатов
        print(f"\nПотери на обучении: {train_loss:.4f}, Точность: {train_acc:.2f}%")
        print(f"Потери на валидации: {val_loss:.4f}, Точность: {val_acc:.2f}%")
        
        # Обновление learning rate
        scheduler.step(val_loss)
        
        # Обновление монитора
        monitor.update(train_loss, train_acc, val_loss, val_acc)
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            
            # Сохранение чекпоинта
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            )
            
            save_model(
                model, optimizer, epoch, 
                {'loss': val_loss, 'accuracy': val_acc}, 
                checkpoint_path
            )
            
            print(f"Сохранена новая лучшая модель! Точность: {val_acc:.2f}%")
        else:
            early_stopping_counter += 1
            print(f"Модель не улучшилась. Early stopping: {early_stopping_counter}/{early_stopping_patience}")
        
        # Проверка условия early stopping
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping! Прекращение обучения.")
            break
    
    # Построение графиков обучения
    plots_dir = os.path.join(os.path.dirname(checkpoint_dir), 'logs')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    monitor.plot_metrics(save_path=plot_path)
    
    # Тестирование модели
    print("\nТестирование модели...")
    test_acc, confusion_matrix = test_model(model, test_loader, device)
    print(f"Точность на тестовой выборке: {test_acc:.2f}%")
    print("Матрица ошибок:")
    print(confusion_matrix)
    
    # Общее время обучения
    total_time = time.time() - monitor.start_time
    print(f"Общее время обучения: {total_time/60:.2f} минут")
    
    return model, monitor

if __name__ == '__main__':
    # Путь к данным
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    
    # Запуск обучения и оценки
    model, monitor = train_and_evaluate(
        data_dir, 
        num_epochs=20,  
        batch_size=16,  
        learning_rate=1e-4,
        advanced_preprocessing=True,
        max_images_per_class=200  # Увеличиваем до 200 изображений на класс (вдвое больше)
    )
