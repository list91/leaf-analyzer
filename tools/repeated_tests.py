import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

from cnn_model import load_model
from cnn_visualize import preprocess_image, predict_image

def test_batch(data_dir, model_path, batch_size=100, device='cpu', seed=None):
    """
    Тестирование модели на батче изображений с заданным seed
    
    Args:
        data_dir: директория с данными
        model_path: путь к модели
        batch_size: размер батча для тестирования
        device: устройство для вычислений
        seed: значение для инициализации генератора случайных чисел
        
    Returns:
        metrics: словарь с метриками тестирования
    """
    # Загрузка модели
    model, _ = load_model(model_path, device)
    model.eval()
    
    # Получение списка изображений
    disease_dir = os.path.join(data_dir, 'disease_plants')
    normal_dir = os.path.join(data_dir, 'normal_plants')
    
    disease_files = [os.path.join(disease_dir, f) for f in os.listdir(disease_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Выбор случайных изображений с заданным seed
    if seed is not None:
        random.seed(seed)
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
    
    print(f"Тестирование с seed={seed} на {len(all_images)} изображениях...")
    
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
    
    # True Negatives, False Positives, False Negatives, True Positives
    tn, fp, fn, tp = cm.ravel()
    
    # Метрики
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'seed': seed,
        'confusion_matrix': cm,
        'true_labels': true_labels,
        'predictions': predictions,
        'confidences': confidences,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }
    
    return metrics

def run_multiple_tests(data_dir, model_path, n_tests=4, batch_size=100, device='cpu'):
    """
    Запуск нескольких тестов с разными seed
    
    Args:
        data_dir: директория с данными
        model_path: путь к модели
        n_tests: количество тестов
        batch_size: размер батча для каждого теста
        device: устройство для вычислений
        
    Returns:
        all_metrics: список словарей с метриками для каждого теста
    """
    # Начальные значения для генератора случайных чисел
    seeds = [42, 123, 456, 789][:n_tests]
    
    # Запуск тестов
    all_metrics = []
    
    for i, seed in enumerate(seeds):
        print(f"Запуск теста {i+1}/{n_tests} с seed={seed}")
        metrics = test_batch(data_dir, model_path, batch_size, device, seed=seed)
        all_metrics.append(metrics)
        
        # Вывод результатов
        cm = metrics['confusion_matrix']
        print(f"Тест {i+1}, Матрица ошибок:")
        print(cm)
        print(f"Точность: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
        print("-" * 40)
    
    return all_metrics

def visualize_test_results(all_metrics, save_path=None):
    """
    Визуализация результатов тестов
    
    Args:
        all_metrics: список словарей с метриками для каждого теста
        save_path: путь для сохранения визуализации
    """
    n_tests = len(all_metrics)
    
    # Создаем фигуру
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Заголовок
    fig.suptitle('Результаты тестирования модели на разных наборах данных', fontsize=16)
    
    # Цвета для визуализации
    colors = ['#4CAF50', '#F44336', '#2196F3', '#FFC107']
    
    # Подготовка данных для визуализации
    test_indices = list(range(1, n_tests + 1))
    accs = [m['accuracy'] * 100 for m in all_metrics]
    precisions = [m['precision'] * 100 for m in all_metrics]
    recalls = [m['recall'] * 100 for m in all_metrics]
    f1s = [m['f1'] * 100 for m in all_metrics]
    
    # 1. Матрицы ошибок для каждого теста
    for i, metrics in enumerate(all_metrics):
        ax = axes[i]
        cm = metrics['confusion_matrix']
        
        # Нормализация матрицы
        cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
        
        # Визуализация матрицы ошибок
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Метки классов
        classes = ['Здоровый', 'Больной']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=f'Тест {i+1} (seed={metrics["seed"]})\nТочность: {metrics["accuracy"]:.2%}',
               ylabel='Истинный класс',
               xlabel='Предсказанный класс')
        
        # Поворот меток по оси x
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Аннотация значений в ячейках
        fmt = '.2f'
        thresh = cm_norm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black")
    
    # 2. Сравнение метрик на одном графике
    if n_tests == 4:  # Для четвертой оси строим сводный график
        ax = axes[3]
        
        width = 0.2
        x = np.arange(4)  # 4 метрики
        
        for i in range(n_tests):
            ax.bar(x + i*width - (n_tests-1)*width/2, 
                [accs[i], precisions[i], recalls[i], f1s[i]], 
                width, label=f'Тест {i+1}', color=colors[i])
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-score'])
        ax.set_ylabel('Процент (%)')
        ax.set_ylim(0, 105)
        ax.set_title('Сравнение метрик')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Добавление значений над столбцами
        for i in range(n_tests):
            metrics = [accs[i], precisions[i], recalls[i], f1s[i]]
            for j, v in enumerate(metrics):
                ax.text(j + i*width - (n_tests-1)*width/2, v + 1, f"{v:.1f}%", 
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Сохранение результата
    if save_path:
        plt.savefig(save_path)
        print(f"Результаты визуализации сохранены в {save_path}")
    
    return fig

if __name__ == '__main__':
    # Путь к данным
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    
    # Получение последней модели
    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(checkpoint_dir, latest_model)
    
    print(f"Используется модель: {latest_model}")
    
    # Запуск нескольких тестов
    all_metrics = run_multiple_tests(
        data_dir, model_path, n_tests=4, batch_size=100, device='cpu'
    )
    
    # Визуализация результатов
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'multi_test_results.png')
    
    visualize_test_results(all_metrics, save_path=save_path)
