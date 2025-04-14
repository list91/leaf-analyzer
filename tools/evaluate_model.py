import os
import torch
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import glob
from datetime import datetime

from preprocessing import LeafPreprocessor
from model import create_model, load_checkpoint
from dataset import prepare_data_loaders

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

def evaluate_model(model, data_loader, device='cpu'):
    """
    Оценка модели на наборе данных
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Расчет метрик
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    precision = precision_score(all_targets, all_predictions, average='weighted') * 100
    recall = recall_score(all_targets, all_predictions, average='weighted') * 100
    f1 = f1_score(all_targets, all_predictions, average='weighted') * 100
    
    # Матрица ошибок
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def main():
    parser = argparse.ArgumentParser(description='Оценка модели на размеченных данных')
    parser.add_argument('--data_dir', type=str, default=None, 
                        help='Директория с данными (должна содержать подпапки disease_plants и normal_plants)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Размер батча для оценки')
    args = parser.parse_args()
    
    # Определение устройства
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")
    
    # Определение директории с данными
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    
    print(f"Используется директория с данными: {data_dir}")
    
    # Загрузка модели
    checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    model = load_best_model(checkpoints_dir, device)
    
    # Подготовка данных
    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_dir, batch_size=args.batch_size
    )
    
    # Оценка на обучающем наборе
    print("\nОценка на обучающем наборе:")
    train_metrics = evaluate_model(model, train_loader, device)
    print(f"Точность: {train_metrics['accuracy']:.2f}%")
    print(f"Precision: {train_metrics['precision']:.2f}%")
    print(f"Recall: {train_metrics['recall']:.2f}%")
    print(f"F1-мера: {train_metrics['f1']:.2f}%")
    print("Матрица ошибок:")
    print(train_metrics['confusion_matrix'])
    
    # Оценка на валидационном наборе
    print("\nОценка на валидационном наборе:")
    val_metrics = evaluate_model(model, val_loader, device)
    print(f"Точность: {val_metrics['accuracy']:.2f}%")
    print(f"Precision: {val_metrics['precision']:.2f}%")
    print(f"Recall: {val_metrics['recall']:.2f}%")
    print(f"F1-мера: {val_metrics['f1']:.2f}%")
    print("Матрица ошибок:")
    print(val_metrics['confusion_matrix'])
    
    # Оценка на тестовом наборе
    print("\nОценка на тестовом наборе:")
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Точность: {test_metrics['accuracy']:.2f}%")
    print(f"Precision: {test_metrics['precision']:.2f}%")
    print(f"Recall: {test_metrics['recall']:.2f}%")
    print(f"F1-мера: {test_metrics['f1']:.2f}%")
    print("Матрица ошибок:")
    print(test_metrics['confusion_matrix'])
    
    # Сохранение результатов в файл
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(results_file, 'w') as f:
        f.write("Результаты оценки модели\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Оценка на обучающем наборе:\n")
        f.write(f"Точность: {train_metrics['accuracy']:.2f}%\n")
        f.write(f"Precision: {train_metrics['precision']:.2f}%\n")
        f.write(f"Recall: {train_metrics['recall']:.2f}%\n")
        f.write(f"F1-мера: {train_metrics['f1']:.2f}%\n")
        f.write("Матрица ошибок:\n")
        f.write(str(train_metrics['confusion_matrix']) + "\n\n")
        
        f.write("Оценка на валидационном наборе:\n")
        f.write(f"Точность: {val_metrics['accuracy']:.2f}%\n")
        f.write(f"Precision: {val_metrics['precision']:.2f}%\n")
        f.write(f"Recall: {val_metrics['recall']:.2f}%\n")
        f.write(f"F1-мера: {val_metrics['f1']:.2f}%\n")
        f.write("Матрица ошибок:\n")
        f.write(str(val_metrics['confusion_matrix']) + "\n\n")
        
        f.write("Оценка на тестовом наборе:\n")
        f.write(f"Точность: {test_metrics['accuracy']:.2f}%\n")
        f.write(f"Precision: {test_metrics['precision']:.2f}%\n")
        f.write(f"Recall: {test_metrics['recall']:.2f}%\n")
        f.write(f"F1-мера: {test_metrics['f1']:.2f}%\n")
        f.write("Матрица ошибок:\n")
        f.write(str(test_metrics['confusion_matrix']) + "\n")
    
    print(f"\nРезультаты сохранены в {results_file}")

if __name__ == '__main__':
    main()
