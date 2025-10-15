#!/usr/bin/env python3
"""
Цветная система логирования для InfoBot
"""
import logging
import sys
from datetime import datetime

class Colors:
    """ANSI цветовые коды"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Основные цвета
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Яркие цвета
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Фоновые цвета
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'

class ColorFormatter(logging.Formatter):
    """Форматтер с цветами для разных уровней логирования"""
    
    # Цвета для разных уровней
    COLORS = {
        'DEBUG': Colors.DIM + Colors.WHITE,
        'INFO': Colors.BRIGHT_CYAN,
        'WARNING': Colors.BRIGHT_YELLOW,
        'ERROR': Colors.BRIGHT_RED,
        'CRITICAL': Colors.BG_RED + Colors.BRIGHT_WHITE,
    }
    
    # Эмодзи для разных категорий
    EMOJIS = {
        'INIT': '🚀',
        'CONFIG': '⚙️',
        'AUTO': '🤖',
        'SYNC': '🔄',
        'CLEANUP': '🧹',
        'STOP': '🛑',
        'ERROR': '❌',
        'SUCCESS': '✅',
        'WARNING': '⚠️',
        'INFO': 'ℹ️',
        'DEBUG': '🔍',
        'RSI': '📈',
        'BOT': '🤖',
        'EXCHANGE': '🏦',
        'API': '🌐',
        'CACHE': '💾',
        'POSITION': '📊',
        'SIGNAL': '🎯',
        'FILTER': '🔍',
        'SAVE': '💾',
        'LOAD': '📂',
        'BATCH': '📦',
        'STOP_LOSS': '🛡️',
        'INACTIVE': '🗑️',
        'STARTUP': '🎬',
        'MATURITY': '🌱',
        'OPTIMAL': '⚡',
        'PROCESS': '⚙️',
        'DEFAULT': '📋',
        'SYSTEM': '🔧',
        'SMART_RSI': '🧠',
        'AUTO_BOT': '🤖',
        'AUTO_SAVE': '💾',
        'EXCHANGE_POSITIONS': '📊',
        'BOTS_CACHE': '💾',
        'POSITION_UPDATE': '🔄',
        'POSITION_SYNC': '🔄',
        'INACTIVE_CLEANUP': '🧹',
        'STOP_LOSS_SETUP': '🛡️',
        'AUTO_BOT_FILTER': '🔍',
        'BOT_INIT': '🤖',
        'BOT_ACTIVE': '✅',
        'BOT_BCH': '🤖',
        'BOT_ES': '🤖',
        'BOT_GPS': '🤖',
        'BOT_HFT': '🤖',
        'BOT_M': '🤖',
        'BOT_RHEA': '🤖',
        'BOT_SLF': '🤖',
        'BOT_TUT': '🤖',
        'LOAD_STATE': '📂',
        'SAVE_STATE': '💾',
        'SIGNAL': '🎯',
    }
    
    def format(self, record):
        # Получаем цвет для уровня логирования
        level_color = self.COLORS.get(record.levelname, Colors.WHITE)
        
        # Извлекаем категорию из сообщения (например, [INIT], [AUTO], etc.)
        category = 'DEFAULT'
        emoji = '📝'
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # Ищем категорию в формате [CATEGORY]
            import re
            match = re.search(r'\[([A-Z_]+)\]', record.msg)
            if match:
                category = match.group(1)
                emoji = self.EMOJIS.get(category, '📝')
        
        # Форматируем время
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Создаем цветное сообщение
        colored_level = f"{level_color}{record.levelname:<8}{Colors.RESET}"
        colored_category = f"{Colors.BRIGHT_MAGENTA}[{category}]{Colors.RESET}"
        colored_emoji = f"{Colors.BRIGHT_YELLOW}{emoji}{Colors.RESET}"
        colored_timestamp = f"{Colors.DIM}{timestamp}{Colors.RESET}"
        
        # Форматируем сообщение
        message = record.getMessage()
        
        # Применяем цвета к разным частям сообщения
        if record.levelname == 'ERROR':
            message = f"{Colors.BRIGHT_RED}{message}{Colors.RESET}"
        elif record.levelname == 'WARNING':
            message = f"{Colors.BRIGHT_YELLOW}{message}{Colors.RESET}"
        elif record.levelname == 'INFO':
            # Выделяем важные части сообщения
            message = self._highlight_important_parts(message)
        
        # Собираем финальное сообщение
        formatted = f"{colored_timestamp} {colored_emoji} {colored_category} {colored_level} {message}"
        
        return formatted
    
    def _highlight_important_parts(self, message):
        """Выделяет важные части сообщения цветом"""
        # Выделяем числа
        import re
        message = re.sub(r'(\d+\.?\d*)', f'{Colors.BRIGHT_CYAN}\\1{Colors.RESET}', message)
        
        # Выделяем статусы
        statuses = ['running', 'idle', 'in_position_long', 'in_position_short', 'paused']
        for status in statuses:
            message = message.replace(status, f'{Colors.BRIGHT_GREEN}{status}{Colors.RESET}')
        
        # Выделяем символы монет
        message = re.sub(r'\b([A-Z]{2,10})\b', f'{Colors.BRIGHT_BLUE}\\1{Colors.RESET}', message)
        
        # Выделяем проценты
        message = re.sub(r'(\d+\.?\d*%)', f'{Colors.BRIGHT_YELLOW}\\1{Colors.RESET}', message)
        
        return message

def setup_color_logging():
    """Настройка цветного логирования"""
    # Создаем логгер
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Удаляем существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Создаем консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Устанавливаем цветной форматтер
    formatter = ColorFormatter()
    console_handler.setFormatter(formatter)
    
    # Добавляем обработчик к логгеру
    logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    # Тест цветного логирования
    setup_color_logging()
    logger = logging.getLogger("test")
    
    logger.info("[INIT] 🚀 Инициализация системы...")
    logger.info("[AUTO] 🤖 Auto Bot включен: True")
    logger.info("[SYNC] 🔄 Синхронизация позиций с биржей")
    logger.warning("[WARNING] ⚠️ Обнаружено 6 расхождений между ботом и биржей")
    logger.error("[ERROR] ❌ Ошибка подключения к бирже")
    logger.info("[BOT] 🤖 Создан бот для BTC (RSI: 25.3, сигнал: ENTER_LONG)")
    logger.info("[POSITION] 📊 Найдено 97 активных позиций с биржи")
    logger.info("[CACHE] 💾 Кэш обновлен: 17 ботов")
