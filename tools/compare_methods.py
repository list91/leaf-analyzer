import os
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

def create_comparison_image(image_path, methods=['grabcut', 'watershed', 'color']):
    """
    Создает изображение для сравнения различных методов сегментации
    
    :param image_path: Путь к исходному изображению
    :param methods: Список методов для сравнения
    :return: Путь к сохраненному изображению сравнения
    """
    # Определяем пути к результатам
    results_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'images', 
        'results'
    ))
    
    # Получаем имя файла
    filename = os.path.basename(image_path)
    
    # Загружаем исходное изображение
    original = cv2.imread(image_path)
    if original is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None
    
    # Конвертируем в RGB для matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Создаем фигуру для сравнения
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(5 * (len(methods) + 1), 5))
    
    # Отображаем исходное изображение
    axes[0].imshow(original_rgb)
    axes[0].set_title('Исходное изображение')
    axes[0].axis('off')
    
    # Отображаем результаты каждого метода
    for i, method in enumerate(methods):
        result_path = os.path.join(results_dir, f'leaf_contour_{method}_{filename}')
        
        if os.path.exists(result_path):
            result = cv2.imread(result_path)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            axes[i + 1].imshow(result_rgb)
            axes[i + 1].set_title(f'Метод: {method}')
            axes[i + 1].axis('off')
        else:
            axes[i + 1].text(0.5, 0.5, f'Результат для метода {method} не найден', 
                           horizontalalignment='center', verticalalignment='center')
            axes[i + 1].axis('off')
    
    # Сохраняем сравнение
    comparison_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'images', 
        'comparisons'
    ))
    os.makedirs(comparison_dir, exist_ok=True)
    
    comparison_path = os.path.join(comparison_dir, f'comparison_{filename}')
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    
    print(f"Сравнение сохранено: {comparison_path}")
    return comparison_path

def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Сравнение методов сегментации листьев')
    parser.add_argument('--dir', type=str, help='Путь к директории с изображениями')
    parser.add_argument('--image', type=str, help='Путь к отдельному изображению')
    parser.add_argument('--methods', type=str, default='grabcut,watershed,color', 
                        help='Методы для сравнения (через запятую)')
    
    args = parser.parse_args()
    
    # Разбираем методы
    methods = args.methods.split(',')
    
    # Определяем изображения для обработки
    if args.image:
        # Обрабатываем одно изображение
        create_comparison_image(args.image, methods)
    else:
        # Определяем директорию с изображениями
        images_dir = args.dir if args.dir else os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'images', 
            'normal_plants_target_point'
        ))
        
        # Проверяем существование директории
        if not os.path.exists(images_dir):
            print(f"Директория не найдена: {images_dir}")
            return
        
        # Поддерживаемые расширения изображений
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        # Получаем список файлов
        image_files = [
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f)) and 
            os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        if not image_files:
            print(f"В директории {images_dir} не найдено изображений")
            return
        
        print(f"Найдено {len(image_files)} изображений для сравнения")
        
        # Обработка каждого изображения
        for image_path in image_files:
            create_comparison_image(image_path, methods)
        
        print("Сравнение завершено.")

if __name__ == "__main__":
    main()
