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
        'FILTER_PROCESSING': '🔍',
        'NEW_AUTO_FILTER': '🔍',
        'NEW_BOT_SIGNALS': '🎯',
        'AUTOBOT_FILTER': '🔍',
    }
    
    def format(self, record):
        # Получаем цвет для уровня логирования
        level_color = self.COLORS.get(record.levelname, Colors.WHITE)
        
        # Получаем исходное сообщение (до форматирования)
        # Важно: работаем с record.msg напрямую, чтобы удалить префикс ДО форматирования
        if hasattr(record, 'msg'):
            if isinstance(record.msg, str):
                message = record.msg
            else:
                # Если record.msg - это не строка (например, объект форматирования),
                # получаем отформатированное сообщение
                message = record.getMessage()
        else:
            message = record.getMessage()
        
        # Определяем имя логгера заранее (используется и ниже)
        logger_name = record.name if hasattr(record, 'name') else 'ROOT'
        
        # Извлекаем категорию из сообщения (например, [INIT], [AUTO], etc.)
        category = 'DEFAULT'
        emoji = '📝'
        
        if isinstance(message, str):
            # Ищем категорию в формате [CATEGORY] в начале сообщения
            import re
            # Ищем категорию в начале сообщения (может быть с пробелами или без)
            # Используем более точное регулярное выражение
            match = re.search(r'^\[([A-Z_]+)\]\s*', message)
            if match:
                category = match.group(1)
                emoji = self.EMOJIS.get(category, '📝')
                # Удаляем префикс категории из сообщения, чтобы избежать дубликата
                # Удаляем [CATEGORY] и возможные пробелы после него
                # Важно: удаляем ТОЛЬКО из начала сообщения
                message_cleaned = re.sub(r'^\[([A-Z_]+)\]\s*', '', message, count=1).strip()
                # Убеждаемся, что удалили именно этот префикс
                if message_cleaned != message:
                    message = message_cleaned
                    # Обновляем record.msg, чтобы удалить префикс из финального сообщения
                    if hasattr(record, 'msg') and isinstance(record.msg, str):
                        record.msg = message
                    # Переопределяем getMessage() чтобы вернуть очищенное сообщение
                    try:
                        # Сохраняем оригинальный getMessage
                        original_getMessage = record.getMessage
                        # Переопределяем его
                        def getMessage_override():
                            return message
                        record.getMessage = getMessage_override
                    except:
                        pass
        
        # ВАЖНО: Удаляем любые оставшиеся префиксы [CATEGORY] из сообщения
        # Это нужно для случаев, когда префиксы добавляются динамически
        if isinstance(message, str):
            import re
            # Удаляем все префиксы [CATEGORY] из начала сообщения
            # (на случай, если они добавились после первоначальной обработки)
            message = re.sub(r'^\[([A-Z_]+)\]\s*', '', message, count=1)
            # Также удаляем префиксы после ANSI-кодов
            message = re.sub(r'(\033\[[0-9;]*m)*\[([A-Z_]+)\]\s*', r'\1', message, count=1)
            
            # Специальная обработка для werkzeug логов - упрощаем формат
            if logger_name == 'werkzeug' or 'werkzeug' in logger_name.lower():
                # Убираем дублирование даты/времени и упрощаем формат
                # Было: 192.168.1.2 - - [14/Nov/2025 05:37:36] "%s" %s %s
                # Станет: GET /api/positions 200
                message = re.sub(r'^[\d\.\s-]+\[.*?\]\s*', '', message)  # Убираем IP и дату
                message = re.sub(r'["%s"]+\s*%s\s*%s', '', message)  # Убираем плейсхолдеры
                message = message.strip()
                
                # Если сообщение пустое или содержит только плейсхолдеры, пропускаем
                if not message or message == '%s' or len(message) < 3:
                    return ''  # Пропускаем пустые сообщения
        
        # Определяем префикс на основе имени логгера (как в ai.py)
        if logger_name.startswith('AI.') or logger_name == 'AI.Main':
            prefix = '[AI]'
        elif logger_name == 'werkzeug' or 'werkzeug' in logger_name.lower():
            prefix = '[APP]'
        elif logger_name.startswith('BotsService') or logger_name == 'BotsService' or 'bot' in logger_name.lower():
            prefix = '[BOTS]'
        else:
            # Для остальных логгеров определяем префикс по имени
            if 'ai' in logger_name.lower():
                prefix = '[AI]'
            elif 'app' in logger_name.lower() or 'flask' in logger_name.lower():
                prefix = '[APP]'
            else:
                prefix = '[BOTS]'  # По умолчанию для bots.py
        
        # Форматируем время без даты и миллисекунд (компактный формат)
        try:
            dt = datetime.fromtimestamp(record.created)
            timestamp = dt.strftime('%H:%M:%S')
        except:
            # Если не удалось получить время, используем текущее время
            dt = datetime.now()
            timestamp = dt.strftime('%H:%M:%S')
        
        # Применяем цвета к разным частям сообщения
        if record.levelname == 'ERROR':
            colored_message = f"{Colors.BRIGHT_RED}{message}{Colors.RESET}"
        elif record.levelname == 'WARNING':
            colored_message = f"{Colors.BRIGHT_YELLOW}{message}{Colors.RESET}"
        elif record.levelname == 'INFO':
            # Выделяем важные части сообщения
            colored_message = self._highlight_important_parts(message)
        else:
            colored_message = message
        
        # Создаем цветные части (компактный формат)
        colored_prefix = f"{Colors.BRIGHT_MAGENTA}{prefix}{Colors.RESET}"
        colored_timestamp = f"{Colors.DIM}{timestamp}{Colors.RESET}"
        colored_level = f"{level_color}{record.levelname}{Colors.RESET}"
        
        # Компактный формат: [PREFIX] HH:MM:SS - LEVEL - message
        formatted = f"{colored_prefix} {colored_timestamp} - {colored_level} - {colored_message}"
        
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
