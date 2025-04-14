import os
import torch
import torch.nn.functional as F
from preprocessing import LeafPreprocessor
from model import create_model, load_checkpoint
from logger import prediction_logger as logger

class LeafPredictor:
    def __init__(self, model_path, device='cpu'):
        logger.info(f"Инициализация LeafPredictor для устройства: {device}")
        self.device = device
        self.model = create_model(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())  # Dummy optimizer for loading checkpoint
        
        # Загрузка весов модели
        logger.info(f"Загрузка модели из: {model_path}")
        load_checkpoint(self.model, self.optimizer, model_path)
        self.model.eval()
        
        # Инициализация препроцессора
        self.preprocessor = LeafPreprocessor()
        logger.info("Инициализация завершена")
    
    def predict(self, image_path):
        """
        Предсказание для одного изображения
        
        Args:
            image_path (str): Путь к изображению
            
        Returns:
            tuple: (предсказанный_класс, вероятность)
        """
        logger.info(f"Предсказание для изображения: {image_path}")
        
        # Предобработка изображения
        logger.debug("Предобработка изображения")
        image_tensor = self.preprocessor.preprocess_for_training(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Получение предсказания
        logger.debug("Выполнение предсказания")
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        logger.info(f"Результат: класс={predicted_class}, уверенность={confidence:.2%}")
        return predicted_class, confidence
    
    def predict_batch(self, image_paths):
        """
        Предсказания для batch изображений
        
        Args:
            image_paths (list): Список путей к изображениям
            
        Returns:
            list: Список кортежей (предсказанный_класс, вероятность)
        """
        logger.info(f"Пакетное предсказание для {len(image_paths)} изображений")
        
        # Предобработка батча изображений
        logger.debug("Предобработка батча изображений")
        batch_tensor = self.preprocessor.preprocess_batch(image_paths)
        batch_tensor = batch_tensor.to(self.device)
        
        results = []
        logger.debug("Выполнение предсказаний")
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidences = torch.gather(probabilities, 1, predicted_classes.unsqueeze(1))
            
            for i, (pred_class, confidence) in enumerate(zip(predicted_classes, confidences)):
                result = (pred_class.item(), confidence.item())
                results.append(result)
                logger.debug(f"Изображение {i+1}: класс={result[0]}, уверенность={result[1]:.2%}")
        
        logger.info("Пакетное предсказание завершено")
        return results

def main():
    # Пример использования
    logger.info("Запуск примера использования LeafPredictor")
    
    model_path = 'path/to/best_model.pth'  # Укажите путь к вашей обученной модели
    predictor = LeafPredictor(model_path)
    
    # Пример предсказания для одного изображения
    image_path = 'path/to/image.jpg'
    pred_class, confidence = predictor.predict(image_path)
    
    class_names = ['Здоровый', 'Больной']
    logger.info(f'Предсказание: {class_names[pred_class]}')
    logger.info(f'Уверенность: {confidence:.2%}')

if __name__ == '__main__':
    main()
