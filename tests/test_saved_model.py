import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

from tools.cnn_model import load_model
from tools.cnn_visualize import preprocess_image, predict_image

def test_random_images(data_dir, model_path, num_images=10, device='cpu'):
    """
    Тестирование модели на случайных изображениях
    
    Args:
        data_dir: директория с данными
        model_path: путь к модели
        num_images: количество изображений для тестирования
        device: устройство для вычислений
    """
    # Загрузка модели
    print(f"Загрузка модели из {model_path}...")
    model, _ = load_model(model_path, device)
    model.eval()
    
    # Получение списка изображений
    disease_dir = os.path.join(data_dir, 'disease_plants')
    normal_dir = os.path.join(data_dir, 'normal_plants')
    
    disease_files = [os.path.join(disease_dir, f) for f in os.listdir(disease_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Выбор случайных изображений
    random.seed(42)  # Для воспроизводимости
    selected_disease = random.sample(disease_files, num_images // 2)
    selected_normal = random.sample(normal_files, num_images // 2)
    
    all_images = selected_disease + selected_normal
    true_labels = [1] * len(selected_disease) + [0] * len(selected_normal)
    
    # Перемешивание
    combined = list(zip(all_images, true_labels))
    random.shuffle(combined)
    all_images, true_labels = zip(*combined)
    
    # Предсказания
    predictions = []
    confidences = []
    
    print(f"Тестирование на {len(all_images)} изображениях...")
    
    # Создание фигуры для визуализации
    fig, axes = plt.subplots(len(all_images) // 2, 4, figsize=(20, 3 * len(all_images) // 2))
    axes = axes.flatten()
    
    for i, (image_path, true_label) in enumerate(zip(all_images, true_labels)):
        # Предсказание
        pred_class, confidence, preprocessed_image = predict_image(model, image_path, device)
        predictions.append(pred_class)
        confidences.append(confidence)
        
        # Определение класса
        true_class_name = "Больной" if true_label == 1 else "Здоровый"
        pred_class_name = "Больной" if pred_class == 1 else "Здоровый"
        
        # Цвет для правильных/неправильных предсказаний
        color = "green" if pred_class == true_label else "red"
        
        # Визуализация
        ax_orig = axes[i*2]
        ax_proc = axes[i*2+1]
        
        # Оригинальное изображение
        original = Image.open(image_path).convert('RGB')
        ax_orig.imshow(np.array(original))
        ax_orig.set_title(f"Истинный класс: {true_class_name}")
        ax_orig.axis('off')
        
        # Предобработанное изображение с предсказанием
        ax_proc.imshow(np.array(preprocessed_image))
        ax_proc.set_title(f"Предсказание: {pred_class_name}\nУверенность: {confidence:.2f}", color=color)
        ax_proc.axis('off')
    
    plt.tight_layout()
    
    # Сохранение результата
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'test_results.png'))
    
    # Вычисление метрик
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['Здоровый', 'Больной'])
    
    print("\nМатрица ошибок:")
    print(cm)
    print("\nОтчет о классификации:")
    print(report)
    
    # Вычисление точности
    accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
    print(f"\nОбщая точность: {accuracy:.2%}")
    
    return predictions, true_labels, confidences

def test_on_batch(data_dir, model_path, batch_size=100, device='cpu'):
    """
    Тестирование модели на большом батче изображений
    
    Args:
        data_dir: директория с данными
        model_path: путь к модели
        batch_size: размер батча для тестирования
        device: устройство для вычислений
    """
    # Загрузка модели
    print(f"Загрузка модели из {model_path}...")
    model, _ = load_model(model_path, device)
    model.eval()
    
    # Получение списка изображений
    disease_dir = os.path.join(data_dir, 'disease_plants')
    normal_dir = os.path.join(data_dir, 'normal_plants')
    
    disease_files = [os.path.join(disease_dir, f) for f in os.listdir(disease_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Выбор случайных изображений
    random.seed(42)  # Для воспроизводимости
    selected_disease = random.sample(disease_files, batch_size // 2)
    selected_normal = random.sample(normal_files, batch_size // 2)
    
    all_images = selected_disease + selected_normal
    true_labels = [1] * len(selected_disease) + [0] * len(selected_normal)
    
    # Перемешивание
    combined = list(zip(all_images, true_labels))
    random.shuffle(combined)
    all_images, true_labels = zip(*combined)
    
    # Предсказания
    predictions = []
    confidences = []
    
    print(f"Тестирование на {len(all_images)} изображениях...")
    
    for i, (image_path, true_label) in enumerate(zip(all_images, true_labels)):
        # Предсказание
        pred_class, confidence, _ = predict_image(model, image_path, device)
        predictions.append(pred_class)
        confidences.append(confidence)
        
        # Вывод прогресса
        if (i+1) % 10 == 0:
            print(f"\rОбработано {i+1}/{len(all_images)} изображений", end="")
    
    print("\n")
    
    # Вычисление метрик
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['Здоровый', 'Больной'])
    
    print("\nМатрица ошибок:")
    print(cm)
    print("\nОтчет о классификации:")
    print(report)
    
    # Вычисление точности
    accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
    print(f"\nОбщая точность: {accuracy:.2%}")
    
    # Визуализация распределения уверенности
    plt.figure(figsize=(12, 6))
    
    # Разделение на правильные и неправильные предсказания
    correct_conf = [conf for conf, p, t in zip(confidences, predictions, true_labels) if p == t]
    incorrect_conf = [conf for conf, p, t in zip(confidences, predictions, true_labels) if p != t]
    
    plt.hist(correct_conf, bins=20, alpha=0.7, label=f'Правильные ({len(correct_conf)})')
    plt.hist(incorrect_conf, bins=20, alpha=0.7, label=f'Неправильные ({len(incorrect_conf)})')
    
    plt.xlabel('Уверенность')
    plt.ylabel('Количество')
    plt.title('Распределение уверенности модели')
    plt.legend()
    plt.grid(True)
    
    # Сохранение графика
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'confidence_distribution.png'))
    
    return predictions, true_labels, confidences

if __name__ == '__main__':
    # Путь к данным
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    
    # Получение последней модели
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(checkpoint_dir, latest_model)
    
    print(f"Используется модель: {latest_model}")
    
    # Тестирование на случайных изображениях с визуализацией
    test_random_images(data_dir, model_path, num_images=10)
    
    # Тестирование на большом батче
    test_on_batch(data_dir, model_path, batch_size=100)
