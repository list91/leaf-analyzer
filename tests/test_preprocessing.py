import os
from utils.preprocessing import visualize_preprocessing

def test_preprocessor():
    # Путь к тестовому изображению из normal_plants
    image_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'normal_plants')
    test_image = os.path.join(image_dir, os.listdir(image_dir)[0])  # Берем первое изображение
    
    # Создаем директорию для результатов
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Путь для сохранения результата
    save_path = os.path.join(results_dir, 'preprocessed_test.png')
    
    print(f"Обработка изображения: {test_image}")
    print(f"Сохранение результата в: {save_path}")
    
    # Запускаем препроцессинг
    result = visualize_preprocessing(test_image, save_path)
    print("Препроцессинг завершен успешно!")

if __name__ == '__main__':
    test_preprocessor()
