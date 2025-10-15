"""
Система ротации логов с ограничением размера файла
Автоматически перезаписывает лог-файлы при превышении 10MB
"""

import os
import logging
import logging.handlers
from datetime import datetime
import threading
from typing import Optional


class RotatingFileHandlerWithSizeLimit(logging.handlers.RotatingFileHandler):
    """
    Обработчик логов с автоматической ротацией при превышении размера файла
    Максимальный размер: 10MB
    """
    
    def __init__(self, filename: str, max_bytes: int = 10 * 1024 * 1024, 
                 backup_count: int = 0, encoding: str = 'utf-8'):
        """
        Args:
            filename: Путь к файлу лога
            max_bytes: Максимальный размер файла в байтах (по умолчанию 10MB)
            backup_count: Количество резервных файлов (0 = перезаписывать)
            encoding: Кодировка файла
        """
        # Создаем директорию если её нет
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        super().__init__(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding
        )
        
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
    
    def doRollover(self):
        """
        Переопределяем метод ротации для перезаписи вместо создания backup файлов
        """
        with self._lock:
            if self.stream:
                self.stream.close()
                self.stream = None
            
            # Просто перезаписываем файл
            if os.path.exists(self.baseFilename):
                try:
                    os.remove(self.baseFilename)
                except OSError:
                    pass
            
            # Открываем новый файл
            if not self.delay:
                self.stream = self._open()


def setup_logger_with_rotation(
    name: str, 
    log_file: str, 
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Настраивает логгер с автоматической ротацией файлов
    
    Args:
        name: Имя логгера
        log_file: Путь к файлу лога
        level: Уровень логирования
        max_bytes: Максимальный размер файла в байтах
        format_string: Формат сообщений лога
    
    Returns:
        Настроенный логгер
    """
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Удаляем существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Создаем обработчик с ротацией
    file_handler = RotatingFileHandlerWithSizeLimit(
        filename=log_file,
        max_bytes=max_bytes,
        backup_count=0,  # Перезаписываем файл
        encoding='utf-8'
    )
    
    # Настраиваем форматтер
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    file_handler.setFormatter(formatter)
    
    # Добавляем обработчик к логгеру
    logger.addHandler(file_handler)
    
    # Отключаем распространение на родительские логгеры
    logger.propagate = False
    
    return logger


def get_log_file_size(log_file: str) -> int:
    """
    Возвращает размер файла лога в байтах
    
    Args:
        log_file: Путь к файлу лога
    
    Returns:
        Размер файла в байтах, 0 если файл не существует
    """
    try:
        return os.path.getsize(log_file)
    except (OSError, FileNotFoundError):
        return 0


def cleanup_old_logs(logs_dir: str = 'logs', max_age_days: int = 7):
    """
    Удаляет старые лог-файлы старше указанного количества дней
    
    Args:
        logs_dir: Директория с логами
        max_age_days: Максимальный возраст файлов в днях
    """
    if not os.path.exists(logs_dir):
        return
    
    current_time = datetime.now().timestamp()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    for filename in os.listdir(logs_dir):
        if filename.endswith('.log'):
            file_path = os.path.join(logs_dir, filename)
            try:
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    print(f"[LOG_CLEANUP] Удален старый лог: {filename}")
            except OSError as e:
                print(f"[LOG_CLEANUP] Ошибка удаления {filename}: {e}")


# Глобальные логгеры для разных компонентов системы
def get_optimal_ema_logger() -> logging.Logger:
    """Логгер для optimal_ema скрипта"""
    return setup_logger_with_rotation(
        name='OptimalEMA',
        log_file='logs/optimal_ema.log',
        level=logging.INFO,
        max_bytes=10 * 1024 * 1024,  # 10MB
        format_string='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_protector_logger() -> logging.Logger:
    """Логгер для protector скрипта"""
    return setup_logger_with_rotation(
        name='Protector',
        log_file='logs/protector.log',
        level=logging.INFO,
        max_bytes=10 * 1024 * 1024,  # 10MB
        format_string='%(asctime)s - %(levelname)s - %(message)s'
    )


def get_telegram_logger() -> logging.Logger:
    """Логгер для Telegram уведомлений"""
    return setup_logger_with_rotation(
        name='TelegramNotifier',
        log_file='logs/telegram.log',
        level=logging.INFO,
        max_bytes=10 * 1024 * 1024,  # 10MB
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_bots_logger() -> logging.Logger:
    """Логгер для основной системы ботов"""
    return setup_logger_with_rotation(
        name='Bots',
        log_file='logs/bots.log',
        level=logging.INFO,
        max_bytes=10 * 1024 * 1024,  # 10MB
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_app_logger() -> logging.Logger:
    """Логгер для Flask приложения"""
    return setup_logger_with_rotation(
        name='FlaskApp',
        log_file='logs/app.log',
        level=logging.INFO,
        max_bytes=10 * 1024 * 1024,  # 10MB
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


if __name__ == "__main__":
    # Тестирование системы ротации
    logger = get_optimal_ema_logger()
    
    # Записываем тестовые сообщения
    for i in range(1000):
        logger.info(f"Тестовое сообщение {i}: " + "A" * 1000)  # Каждое сообщение ~1KB
    
    print(f"Размер файла лога: {get_log_file_size('logs/optimal_ema.log')} байт")
