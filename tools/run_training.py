import os
import torch
from train import train_model

def main():
    # Установка параметров обучения
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    num_epochs = 50
    batch_size = 8
    learning_rate = 1e-4
    
    # Проверка наличия GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство: {device}")
    
    # Проверка данных
    disease_dir = os.path.join(data_dir, 'disease_plants')
    normal_dir = os.path.join(data_dir, 'normal_plants')
    
    disease_count = len([f for f in os.listdir(disease_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    normal_count = len([f for f in os.listdir(normal_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Количество изображений больных растений: {disease_count}")
    print(f"Количество изображений здоровых растений: {normal_count}")
    
    # Запуск обучения
    print("\nЗапуск обучения модели...")
    model, best_val_acc = train_model(
        data_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    print(f"\nОбучение завершено!")
    print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
