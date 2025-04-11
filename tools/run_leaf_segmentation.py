import os
import sys
from leaf_segmentation import process_directory

def main():
    """
    Запускает обработку изображений с красными метками
    """
    # Путь к директории с изображениями
    images_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'images', 
        'normal_plants_target_point'
    ))
    
    # Проверяем существование директории
    if not os.path.exists(images_dir):
        print(f"Директория не найдена: {images_dir}")
        print("Создаем директорию...")
        os.makedirs(images_dir, exist_ok=True)
        print(f"Пожалуйста, поместите изображения с красными метками в директорию: {images_dir}")
        return
    
    # Обработка всех изображений
    print(f"Начинаем обработку изображений из директории: {images_dir}")
    process_directory(images_dir)
    
    print("Обработка завершена.")
    
    # Путь к директории с результатами
    results_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'images', 
        'results'
    ))
    print(f"Результаты сохранены в директории: {results_dir}")

if __name__ == "__main__":
    main()
