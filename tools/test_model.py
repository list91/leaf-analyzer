import os
import torch
import argparse
import matplotlib
matplotlib.use('Agg')  # Использование не-интерактивного бэкенда
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import glob
from datetime import datetime

from preprocessing import LeafPreprocessor
from model import create_model, load_checkpoint

def load_best_model(checkpoints_dir, device='cpu'):
    """
    Загрузка лучшей модели из директории с чекпоинтами
    """
    # Находим последний чекпоинт
    checkpoints = glob.glob(os.path.join(checkpoints_dir, 'best_model_*.pth'))
    if not checkpoints:
        raise FileNotFoundError("Чекпоинты не найдены в указанной директории")
    
    # Сортируем по дате создания (последний будет самым новым)
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"Загрузка модели из {latest_checkpoint}")
    
    # Создаем модель и оптимизатор
    model = create_model(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Загружаем веса
    epoch, loss, accuracy = load_checkpoint(model, optimizer, latest_checkpoint)
    print(f"Загружена модель: эпоха {epoch}, точность {accuracy:.2f}%")
    
    return model

def predict_image(model, image_path, device='cpu'):
    """
    Предсказание класса для изображения
    """
    # Предобработка изображения
    preprocessor = LeafPreprocessor()
    preprocessor.set_debug(False)  # Отключаем логи
    image_tensor = preprocessor.preprocess_for_training(image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Добавляем размерность батча
    
    # Предсказание
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Классы: 0 - здоровый, 1 - больной
    class_names = ['Здоровое растение', 'Больное растение']
    prediction = class_names[predicted.item()]
    confidence = confidence.item() * 100
    
    return prediction, confidence

def visualize_prediction(image_path, prediction, confidence):
    """
    Визуализация предсказания
    """
    try:
        # Загрузка изображения
        image = Image.open(image_path).convert('RGB')
        
        # Создание маски листа для визуализации
        preprocessor = LeafPreprocessor()
        preprocessor.set_debug(False)  # Отключаем логи
        image_np = np.array(image)
        mask = preprocessor.create_leaf_mask(image_np)
        
        # Применение маски к изображению
        result = cv2.bitwise_and(image_np, image_np, mask=mask)
        
        # Отображение
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Исходное изображение")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title(f"Предсказание: {prediction}\nУверенность: {confidence:.2f}%")
        plt.axis('off')
        
        # Сохранение результата
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"prediction_{filename.split('.')[0]}.png")
        plt.savefig(output_path)
        print(f"Результат сохранен в {output_path}")
        
        plt.close()  # Закрываем фигуру, чтобы освободить память
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")

def test_model_on_images(model, test_dir, device='cpu', limit=5):
    """
    Тестирование модели на изображениях из директории
    """
    # Получаем список изображений
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(test_dir, ext)))
    
    if not image_files:
        print(f"Изображения не найдены в директории {test_dir}")
        return
    
    # Ограничиваем количество изображений
    if limit > 0 and limit < len(image_files):
        image_files = image_files[:limit]
    
    print(f"Тестирование модели на {len(image_files)} изображениях...")
    
    # Предсказания для каждого изображения
    for i, image_path in enumerate(image_files, 1):
        print(f"Изображение {i}/{len(image_files)}: {os.path.basename(image_path)}")
        try:
            prediction, confidence = predict_image(model, image_path, device)
            print(f"Предсказание: {prediction}, Уверенность: {confidence:.2f}%")
            
            # Визуализация
            visualize_prediction(image_path, prediction, confidence)
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Тестирование модели на изображениях')
    parser.add_argument('--test_dir', type=str, default=None, 
                        help='Директория с тестовыми изображениями')
    parser.add_argument('--image', type=str, default=None,
                        help='Путь к одному изображению для тестирования')
    parser.add_argument('--limit', type=int, default=5,
                        help='Ограничение количества тестируемых изображений')
    args = parser.parse_args()
    
    # Определение устройства
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")
    
    # Загрузка модели
    checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    model = load_best_model(checkpoints_dir, device)
    
    # Тестирование
    if args.image:
        # Тестирование на одном изображении
        prediction, confidence = predict_image(model, args.image, device)
        print(f"Предсказание: {prediction}, Уверенность: {confidence:.2f}%")
        visualize_prediction(args.image, prediction, confidence)
    elif args.test_dir:
        # Тестирование на директории
        test_model_on_images(model, args.test_dir, device, args.limit)
    else:
        # По умолчанию используем тестовую директорию
        test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'test')
        if os.path.exists(test_dir):
            test_model_on_images(model, test_dir, device, args.limit)
        else:
            print("Тестовая директория не найдена. Используем директорию с изображениями больных растений.")
            test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'disease_plants')
            test_model_on_images(model, test_dir, device, args.limit)

if __name__ == '__main__':
    main()
