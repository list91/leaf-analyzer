import logging
import sys
import os
from datetime import datetime

def setup_logger(name):
    """
    Настройка логгера с выводом в консоль и файл
    """
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Устанавливаем самый низкий уровень для логгера
    
    # Если логгер уже имеет обработчики, очищаем их
    if logger.handlers:
        logger.handlers.clear()
    
    # Форматтер для сообщений
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Обработчик для консоли (INFO и выше)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Создаем директорию для логов если её нет
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Обработчик для файла (DEBUG и выше)
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name):
    """
    Получение существующего логгера или создание нового
    """
    return logging.getLogger(name)

# Создаем логгеры для разных модулей
preprocessing_logger = setup_logger('preprocessing')
dataset_logger = setup_logger('dataset')
training_logger = setup_logger('training')
model_logger = setup_logger('model')
prediction_logger = setup_logger('prediction')

# Добавляем логгер для основного скрипта
main_logger = setup_logger('main')
