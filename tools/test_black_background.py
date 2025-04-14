import os
import torch
import argparse
import numpy as np
import cv2
from PIL import Image
import glob
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix

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
    
    return prediction, confidence, predicted.item()

def test_black_background_influence(model, test_dir, device='cpu'):
    """
    Тестирование влияния черного фона на классификацию
    """
    # Получаем список изображений
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(test_dir, ext)))
    
    if not image_files:
        print(f"Изображения не найдены в директории {test_dir}")
        return
    
    print(f"Тестирование влияния черного фона на {len(image_files)} изображениях...")
    
    # Подготовка для статистики
    original_predictions = []
    black_bg_predictions = []
    white_bg_predictions = []
    true_labels = []
    
    # Директория для сохранения результатов
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'black_bg_test')
    os.makedirs(results_dir, exist_ok=True)
    
    # Определяем истинные метки из пути (disease_plants или normal_plants)
    for i, image_path in enumerate(image_files, 1):
        try:
            # Определяем истинную метку
            true_label = 1 if 'disease_plants' in image_path else 0
            true_labels.append(true_label)
            
            # Предсказание для оригинального изображения
            prediction, confidence, pred_label = predict_image(model, image_path, device)
            original_predictions.append(pred_label)
            
            # Загружаем изображение
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
            
            # Создаем маску листа
            preprocessor = LeafPreprocessor()
            preprocessor.set_debug(False)
            mask = preprocessor.create_leaf_mask(image_np)
            
            # Создаем изображение с черным фоном
            black_bg = np.zeros_like(image_np)
            black_bg[mask > 0] = image_np[mask > 0]
            
            # Создаем изображение с белым фоном
            white_bg = np.ones_like(image_np) * 255
            white_bg[mask > 0] = image_np[mask > 0]
            
            # Сохраняем изображения
            filename = os.path.basename(image_path).split('.')[0]
            
            black_bg_path = os.path.join(results_dir, f"{filename}_black_bg.png")
            cv2.imwrite(black_bg_path, cv2.cvtColor(black_bg, cv2.COLOR_RGB2BGR))
            
            white_bg_path = os.path.join(results_dir, f"{filename}_white_bg.png")
            cv2.imwrite(white_bg_path, cv2.cvtColor(white_bg, cv2.COLOR_RGB2BGR))
            
            # Предсказания для изображений с разным фоном
            _, _, black_pred = predict_image(model, black_bg_path, device)
            black_bg_predictions.append(black_pred)
            
            _, _, white_pred = predict_image(model, white_bg_path, device)
            white_bg_predictions.append(white_pred)
            
            print(f"Изображение {i}/{len(image_files)}: {os.path.basename(image_path)}")
            print(f"  Истинная метка: {'Больное' if true_label == 1 else 'Здоровое'}")
            print(f"  Оригинал: {'Больное' if pred_label == 1 else 'Здоровое'}")
            print(f"  Черный фон: {'Больное' if black_pred == 1 else 'Здоровое'}")
            print(f"  Белый фон: {'Больное' if white_pred == 1 else 'Здоровое'}")
            
        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")
    
    # Расчет метрик
    original_accuracy = accuracy_score(true_labels, original_predictions) * 100
    black_bg_accuracy = accuracy_score(true_labels, black_bg_predictions) * 100
    white_bg_accuracy = accuracy_score(true_labels, white_bg_predictions) * 100
    
    # Матрицы ошибок
    original_cm = confusion_matrix(true_labels, original_predictions)
    black_bg_cm = confusion_matrix(true_labels, black_bg_predictions)
    white_bg_cm = confusion_matrix(true_labels, white_bg_predictions)
    
    # Согласованность предсказаний
    original_black_agreement = np.mean(np.array(original_predictions) == np.array(black_bg_predictions)) * 100
    original_white_agreement = np.mean(np.array(original_predictions) == np.array(white_bg_predictions)) * 100
    black_white_agreement = np.mean(np.array(black_bg_predictions) == np.array(white_bg_predictions)) * 100
    
    # Вывод результатов
    print("\nРезультаты тестирования влияния фона:")
    print(f"Точность (оригинал): {original_accuracy:.2f}%")
    print(f"Точность (черный фон): {black_bg_accuracy:.2f}%")
    print(f"Точность (белый фон): {white_bg_accuracy:.2f}%")
    
    print("\nСогласованность предсказаний:")
    print(f"Оригинал vs Черный фон: {original_black_agreement:.2f}%")
    print(f"Оригинал vs Белый фон: {original_white_agreement:.2f}%")
    print(f"Черный фон vs Белый фон: {black_white_agreement:.2f}%")
    
    print("\nМатрица ошибок (оригинал):")
    print(original_cm)
    print("\nМатрица ошибок (черный фон):")
    print(black_bg_cm)
    print("\nМатрица ошибок (белый фон):")
    print(white_bg_cm)
    
    # Сохранение результатов в файл
    results_file = os.path.join(results_dir, f"background_influence_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(results_file, 'w') as f:
        f.write("Результаты тестирования влияния фона на классификацию\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Точность классификации:\n")
        f.write(f"Оригинальные изображения: {original_accuracy:.2f}%\n")
        f.write(f"Изображения с черным фоном: {black_bg_accuracy:.2f}%\n")
        f.write(f"Изображения с белым фоном: {white_bg_accuracy:.2f}%\n\n")
        
        f.write("Согласованность предсказаний:\n")
        f.write(f"Оригинал vs Черный фон: {original_black_agreement:.2f}%\n")
        f.write(f"Оригинал vs Белый фон: {original_white_agreement:.2f}%\n")
        f.write(f"Черный фон vs Белый фон: {black_white_agreement:.2f}%\n\n")
        
        f.write("Матрица ошибок (оригинал):\n")
        f.write(str(original_cm) + "\n\n")
        f.write("Матрица ошибок (черный фон):\n")
        f.write(str(black_bg_cm) + "\n\n")
        f.write("Матрица ошибок (белый фон):\n")
        f.write(str(white_bg_cm) + "\n")
    
    print(f"\nРезультаты сохранены в {results_file}")

def main():
    parser = argparse.ArgumentParser(description='Тестирование влияния черного фона на классификацию')
    parser.add_argument('--test_dir', type=str, default=None, 
                        help='Директория с тестовыми изображениями')
    parser.add_argument('--limit', type=int, default=10,
                        help='Ограничение количества тестируемых изображений')
    args = parser.parse_args()
    
    # Определение устройства
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")
    
    # Загрузка модели
    checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    model = load_best_model(checkpoints_dir, device)
    
    # Тестирование
    if args.test_dir:
        # Тестирование на указанной директории
        test_black_background_influence(model, args.test_dir, device)
    else:
        # По умолчанию используем тестовую директорию
        test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'test')
        if os.path.exists(test_dir):
            test_black_background_influence(model, test_dir, device)
        else:
            print("Тестовая директория не найдена. Используем директорию с изображениями больных растений.")
            test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'disease_plants')
            test_black_background_influence(model, test_dir, device)

if __name__ == '__main__':
    main()
