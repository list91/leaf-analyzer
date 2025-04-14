import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime
import random
import sys
import time

from dataset import prepare_data_loaders
from model import create_model, save_checkpoint

class TrainingStatus:
    def __init__(self, num_epochs, total_batches, total_images):
        self.num_epochs = num_epochs
        self.total_batches = total_batches
        self.total_images = total_images
        self.current_epoch = 0
        self.current_batch = 0
        self.current_loss = 0.0
        self.current_acc = 0.0
        self.best_acc = 0.0
        self.last_message = ""
        self.processed_images = 0
        
    def update_status(self, epoch, batch, loss, acc, best_acc, batch_size=0):
        self.current_epoch = epoch
        self.current_batch = batch
        self.current_loss = loss
        self.current_acc = acc
        self.best_acc = best_acc
        if batch_size > 0:
            self.processed_images += batch_size
        
    def print_status(self):
        # Очищаем предыдущий вывод
        sys.stdout.write("\033[K")  # Очистка текущей строки
        sys.stdout.write("\033[F")  # Перемещение курсора вверх
        sys.stdout.write("\033[K")  # Очистка предыдущей строки
        
        # Статус обучения в верхней строке
        status = (f"Эпоха: {self.current_epoch}/{self.num_epochs} | "
                 f"Батч: {self.current_batch}/{self.total_batches} | "
                 f"Изображений: {self.processed_images}/{self.total_images} | "
                 f"Loss: {self.current_loss:.4f} | "
                 f"Точность: {self.current_acc:.2f}% | "
                 f"Ошибка: {100-self.current_acc:.2f}% | "
                 f"Лучшая точность: {self.best_acc:.2f}%")
        
        print(status)
        if self.last_message:
            print(self.last_message)
        sys.stdout.flush()
    
    def set_message(self, message):
        self.last_message = message
        self.print_status()

def train_epoch(model, train_loader, criterion, optimizer, device, status):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Обновляем статус
        current_loss = running_loss / batch_idx
        current_acc = 100. * correct / total
        status.update_status(
            status.current_epoch,
            batch_idx,
            current_loss,
            current_acc,
            status.best_acc,
            batch_size=inputs.size(0)
        )
        status.print_status()
        time.sleep(0.1)  # Небольшая задержка для читаемости вывода
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, status):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    status.set_message(f"Валидация - Loss: {val_loss:.4f}, Точность: {val_acc:.2f}%, Ошибка: {100-val_acc:.2f}%")
    return val_loss, val_acc

def train_model(data_dir, num_epochs=5, batch_size=8, learning_rate=1e-4, max_images_per_class=100):
    # Определение устройства
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Ограничиваем датасет
    disease_files, normal_files = [], []
    for class_dir in ['disease_plants', 'normal_plants']:
        files = [f for f in os.listdir(os.path.join(data_dir, class_dir)) 
                if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)
        if class_dir == 'disease_plants':
            disease_files = files[:max_images_per_class]
        else:
            normal_files = files[:max_images_per_class]
    
    # Подготовка данных
    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_dir, batch_size=batch_size,
        disease_files=disease_files,
        normal_files=normal_files
    )
    
    # Инициализация статуса
    total_images = len(disease_files) + len(normal_files)
    status = TrainingStatus(
        num_epochs=num_epochs,
        total_batches=len(train_loader),
        total_images=total_images
    )
    
    # Создание модели и оптимизатора
    model = create_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Создание директории для сохранения чекпоинтов
    checkpoint_dir = os.path.join(os.path.dirname(data_dir), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_acc = 0.0
    early_stopping_counter = 0
    early_stopping_patience = 10
    
    print("\n")  # Начальные пустые строки для статуса
    print("\n")
    
    for epoch in range(num_epochs):
        status.current_epoch = epoch + 1
        status.processed_images = 0  # Сбрасываем счетчик изображений в начале эпохи
        
        # Обучение
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, status
        )
        
        # Валидация
        val_loss, val_acc = validate(model, val_loader, criterion, device, status)
        
        # Обновление learning rate
        scheduler.step(val_loss)
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            status.best_acc = best_val_acc
            early_stopping_counter = 0
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            )
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc, checkpoint_path
            )
            status.set_message(f"Сохранена новая лучшая модель! Точность: {val_acc:.2f}%")
        else:
            early_stopping_counter += 1
            status.set_message(f"Модель не улучшилась. Early stopping: {early_stopping_counter}/{early_stopping_patience}")
        
        if early_stopping_counter >= early_stopping_patience:
            status.set_message("Early stopping! Прекращение обучения.")
            break
    
    # Финальное тестирование
    test_loss, test_acc = validate(model, test_loader, criterion, device, status)
    status.set_message(
        f"Обучение завершено! "
        f"Точность на тесте: {test_acc:.2f}%, "
        f"Ошибка на тесте: {100-test_acc:.2f}%, "
        f"Лучшая точность: {best_val_acc:.2f}%"
    )
    
    return model, best_val_acc

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    train_model(data_dir, num_epochs=5, batch_size=8, max_images_per_class=100)
