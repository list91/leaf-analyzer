import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms

from cnn_model import load_model
from cnn_dataset import create_leaf_mask

def preprocess_image(image_path, size=(224, 224)):
    """
    Предобработка изображения для предсказания
    
    Args:
        image_path: путь к изображению
        size: размер для ресайза
        
    Returns:
        tensor: предобработанный тензор изображения
    """
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')
    
    # Создание маски листа
    leaf_mask = create_leaf_mask(image)
    
    # Применение маски к изображению
    image_np = np.array(image)
    masked_image = cv2.bitwise_and(image_np, image_np, mask=leaf_mask)
    
    # Конвертация обратно в PIL
    masked_image_pil = Image.fromarray(masked_image)
    
    # Трансформации
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Применение трансформаций
    tensor = transform(masked_image_pil)
    
    return tensor, masked_image_pil

def predict_image(model, image_path, device='cpu'):
    """
    Предсказание класса для изображения
    
    Args:
        model: модель
        image_path: путь к изображению
        device: устройство для вычислений
        
    Returns:
        pred_class: предсказанный класс (0 - здоровый, 1 - больной)
        confidence: уверенность предсказания
        preprocessed_image: предобработанное изображение
    """
    # Предобработка изображения
    tensor, preprocessed_image = preprocess_image(image_path)
    
    # Добавляем размерность батча
    tensor = tensor.unsqueeze(0).to(device)
    
    # Предсказание
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probabilities, 1)
    
    return pred_class.item(), confidence.item(), preprocessed_image

def visualize_prediction(image_path, model_path, output_path=None, device='cpu'):
    """
    Визуализация предсказания модели
    
    Args:
        image_path: путь к изображению
        model_path: путь к модели
        output_path: путь для сохранения результата
        device: устройство для вычислений
    """
    # Загрузка модели
    model, _ = load_model(model_path, device)
    
    # Предсказание
    pred_class, confidence, preprocessed_image = predict_image(model, image_path, device)
    
    # Определение класса
    class_name = "Больной" if pred_class == 1 else "Здоровый"
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Оригинальное изображение
    original = Image.open(image_path).convert('RGB')
    ax1.imshow(np.array(original))
    ax1.set_title("Оригинальное изображение")
    ax1.axis('off')
    
    # Предобработанное изображение с предсказанием
    ax2.imshow(np.array(preprocessed_image))
    ax2.set_title(f"Предсказание: {class_name} (уверенность: {confidence:.2f})")
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Сохранение результата
    if output_path:
        plt.savefig(output_path)
        print(f"Результат сохранен в {output_path}")
    
    plt.show()
    
    return class_name, confidence

def batch_predict(image_dir, model_path, output_dir=None, device='cpu'):
    """
    Пакетное предсказание для всех изображений в директории
    
    Args:
        image_dir: директория с изображениями
        model_path: путь к модели
        output_dir: директория для сохранения результатов
        device: устройство для вычислений
    """
    # Загрузка модели
    model, _ = load_model(model_path, device)
    
    # Создание директории для результатов
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Получение списка изображений
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Статистика
    results = {
        'healthy': 0,
        'diseased': 0,
        'predictions': []
    }
    
    # Обработка каждого изображения
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        
        # Предсказание
        pred_class, confidence, _ = predict_image(model, image_path, device)
        
        # Обновление статистики
        if pred_class == 0:
            results['healthy'] += 1
        else:
            results['diseased'] += 1
        
        # Сохранение информации о предсказании
        results['predictions'].append({
            'file': image_file,
            'class': 'diseased' if pred_class == 1 else 'healthy',
            'confidence': confidence
        })
        
        # Вывод прогресса
        print(f"\rОбработано {i+1}/{len(image_files)} изображений", end="")
        
        # Визуализация и сохранение результата
        if output_dir:
            output_path = os.path.join(output_dir, f"pred_{image_file}")
            visualize_prediction(image_path, model_path, output_path, device)
    
    print("\nОбработка завершена!")
    print(f"Здоровых растений: {results['healthy']}")
    print(f"Больных растений: {results['diseased']}")
    
    return results

if __name__ == '__main__':
    # Пример использования
    image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'test_image.jpg')
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'best_model.pth')
    
    # Визуализация предсказания для одного изображения
    if os.path.exists(image_path) and os.path.exists(model_path):
        visualize_prediction(image_path, model_path)
    else:
        print("Укажите корректные пути к изображению и модели")
