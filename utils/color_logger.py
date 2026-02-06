#!/usr/bin/env python3
"""
Цветная система логирования для InfoBot
"""
import logging
import sys
from datetime import datetime

class LogLevelFilter(logging.Filter):
    """
    Фильтр для управления уровнями логирования в консоли.
    Поддерживает синтаксис: +INFO, -WARNING, +ERROR, -DEBUG и т.д.
    Также поддерживает строку с запятыми: "+INFO, -WARNING, +ERROR, -DEBUG"
    
    Автоматически скрывает DEBUG логи от внешних библиотек (urllib3, pybit и т.д.)
    если DEBUG уровень не включен явно.
    """
    
    # Маппинг строковых уровней на числовые
    LEVEL_MAP = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    
    # Логгеры внешних библиотек, которые обычно шумят в DEBUG
    EXTERNAL_LOGGERS = {
        'urllib3',
        'urllib3.connectionpool',
        'pybit',
        'pybit._http_manager',
        'requests',
        'requests.packages.urllib3',
        'httpcore',
        'httpx',
        'tensorflow',
        'tensorflow.python',
        'tensorflow.core',
        'matplotlib',
        'matplotlib.font_manager',
        'matplotlib.backends',
        'PIL',
        'PIL.PngImagePlugin',
        'pandas',
        'pandas.io',
        'pandas.core',
        'flask_cors',
        'flask_cors.core',
        'werkzeug',
        'flask',
        'flask.app',
    }
    
    def __init__(self, level_settings=None):
        """
        Инициализация фильтра
        
        Args:
            level_settings: Может быть:
                - Список строк: ['+INFO', '-WARNING', '+ERROR', '-DEBUG']
                - Одна строка с запятыми: "+INFO, -WARNING, +ERROR, -DEBUG"
                - None или пустой список [] - все уровни разрешены
        """
        super().__init__()
        self.enabled_levels = set()
        # По умолчанию DEBUG не включен (скрываем шумные логи от библиотек)
        self.debug_enabled = False
        
        # Проверяем, что настройки не None и не пустые
        if level_settings is not None and level_settings != []:
            # Если это пустая строка после обработки, тоже считаем как "все разрешено"
            if isinstance(level_settings, str) and not level_settings.strip():
                # Пустая строка - разрешаем все
                all_levels = set(self.LEVEL_MAP.keys())
                self.enabled_levels = all_levels
                self.debug_enabled = True
            else:
                self._parse_settings(level_settings)
                # Если после парсинга enabled_levels пустой, значит нужно разрешить все
                if not self.enabled_levels:
                    all_levels = set(self.LEVEL_MAP.keys())
                    self.enabled_levels = all_levels
                    self.debug_enabled = True
        else:
            # Если настройки не указаны (None) или пустой список [], включаем все уровни
            all_levels = set(self.LEVEL_MAP.keys())
            self.enabled_levels = all_levels
            # Когда все уровни разрешены, включаем DEBUG для всех (включая внешние библиотеки)
            self.debug_enabled = True
    
    def _parse_settings(self, settings):
        """Парсит настройки уровней логирования"""
        # Если список пустой, разрешаем все уровни
        if not settings:
            return
        
        # Если переданная строка (не список), разбиваем по запятым
        if isinstance(settings, str):
            settings = [s.strip() for s in settings.split(',') if s.strip()]
        
        # Сначала собираем все включенные и выключенные уровни
        enabled = set()
        disabled = set()
        
        for setting in settings:
            # Если это уже строка, используем как есть, иначе преобразуем
            if not isinstance(setting, str):
                setting = str(setting)
            setting = setting.strip().upper()
            if not setting:
                continue
            
            # Парсим формат: +LEVEL или -LEVEL
            if setting.startswith('+'):
                level_name = setting[1:]
                if level_name in self.LEVEL_MAP:
                    enabled.add(level_name)
                    if level_name == 'DEBUG':
                        # Если явно включен DEBUG, разрешаем его для всех (включая внешние библиотеки)
                        self.debug_enabled = True
            elif setting.startswith('-'):
                level_name = setting[1:]
                if level_name in self.LEVEL_MAP:
                    disabled.add(level_name)
                    if level_name == 'DEBUG':
                        self.debug_enabled = False
        
        # Если есть явно включенные уровни, используем только их
        # Иначе используем все уровни кроме явно выключенных
        if enabled:
            self.enabled_levels = enabled
        else:
            # Разрешаем все уровни кроме выключенных
            all_levels = set(self.LEVEL_MAP.keys())
            self.enabled_levels = all_levels - disabled
        
        # КРИТИЧНО: Если enabled_levels пустой после парсинга, значит все уровни выключены
        # В этом случае включаем все уровни (это не должно происходить, но на всякий случай)
        if not self.enabled_levels:
            all_levels = set(self.LEVEL_MAP.keys())
            self.enabled_levels = all_levels
            self.debug_enabled = True
    
    def filter(self, record):
        """
        Фильтрует записи логов на основе настроек уровней
        
        Returns:
            True если запись должна быть показана, False если нужно скрыть
        """
        level_name = record.levelname
        logger_name = record.name if hasattr(record, 'name') else ''
        
        # Скрываем неформатированные сообщения из внешних библиотек (urllib3, pybit, flask-cors)
        # Это проблема библиотек, а не нашего кода - они используют старый стиль форматирования
        try:
            # КРИТИЧНО: Проверяем как отформатированное сообщение, так и исходное
            # Некоторые библиотеки передают неформатированные строки в record.msg
            message = None
            if hasattr(record, 'msg') and isinstance(record.msg, str):
                message = record.msg
            if not message or '%s' not in message:
                # Если в исходном сообщении нет %s, проверяем отформатированное
                try:
                    message = record.getMessage() if hasattr(record, 'getMessage') else str(record.msg)
                except:
                    message = str(record.msg) if hasattr(record, 'msg') else ''
            
            if isinstance(message, str) and message:
                # Агрессивная проверка: скрываем ВСЕ сообщения с множественными неформатированными %s
                # Это типичная проблема библиотек, которые логируют до форматирования
                import re
                
                # Подсчитываем количество неформатированных %s (не %d, %f и т.д.)
                unformatted_count = len(re.findall(r'%s(?!\w)', message))
                
                # Скрываем если:
                # 1. Есть типичные паттерны неформатированных сообщений
                # 2. Или 3+ неформатированных %s (явно неформатированное сообщение)
                # 3. Или сообщение от внешних библиотек с 2+ неформатированными %s
                # 4. Или сообщение содержит паттерн "%s %s %s" (типичный для urllib3)
                has_unformatted = (
                    '%s://%s:%s' in message or 
                    '"%s %s %s"' in message or  # urllib3 паттерн: "%s %s %s"
                    '%s %s %s' in message or
                    'Starting new HTTPS connection' in message or 
                    'Starting new HTTP connection' in message or
                    'Creating converter from' in message or
                    ('Configuring CORS' in message and '%s' in message) or
                    'Settings CORS headers' in message or  # CORS логи с неформатированными %s
                    'CORS request received' in message or  # CORS логи с неформатированными %s
                    'Origin header matches' in message or  # CORS логи с неформатированными %s
                    'CORS have been already evaluated, skipping' in message or  # flask-cors: повторяется на каждый запрос
                    unformatted_count >= 3 or  # Любое сообщение с 3+ неформатированными %s
                    (unformatted_count >= 2 and logger_name.startswith(('urllib3', 'pybit', 'flask_cors', 'requests', 'werkzeug', 'flask', 'app')))
                )
                if has_unformatted:
                    # Это неформатированное сообщение из внешней библиотеки - скрываем его
                    return False
        except:
            pass  # Если не удалось проверить, пропускаем
        
        # Скрываем несущественные SSL ошибки при получении сетевого времени (DEBUG уровень)
        # Это не критичные ошибки, которые не должны засорять логи
        try:
            message = record.getMessage() if hasattr(record, 'getMessage') else str(record.msg)
            if isinstance(message, str) and level_name == 'DEBUG':
                message_lower = message.lower()
                # Проверяем, является ли это SSL ошибкой при получении сетевого времени
                if ('worldtimeapi' in message_lower or 'сетевое время' in message_lower or 'network time' in message_lower) and \
                   ('ssl' in message_lower or 'sslerror' in message_lower or 'unexpected_eof' in message_lower or 'ssl: unexpected_eof' in message_lower):
                    # Это несущественная SSL ошибка - скрываем её
                    return False
        except Exception:
            pass  # Если не удалось проверить, пропускаем

        # Скрываем шум PyTorch: FakeTensor cache stats, cache_hits/cache_misses (неформатированные %s)
        try:
            msg = (record.msg if hasattr(record, 'msg') and isinstance(record.msg, str) else None) or (record.getMessage() if hasattr(record, 'getMessage') else str(record.msg))
            if isinstance(msg, str) and (
                'FakeTensor cache stats' in msg or msg.strip() in ('cache_hits: %s', 'cache_misses: %s')
            ):
                return False
        except Exception:
            pass

        # Всегда скрываем DEBUG от внешних библиотек, если DEBUG не включен явно
        if level_name == 'DEBUG' and not self.debug_enabled:
            for external_logger in self.EXTERNAL_LOGGERS:
                if logger_name.startswith(external_logger):
                    return False
        
        # Если уровни не настроены, разрешаем все (кроме уже отфильтрованных выше)
        if not self.enabled_levels:
            return True
        
        # КРИТИЧНО: Проверяем уровень записи
        # Если уровень не включен, скрываем
        if level_name not in self.enabled_levels:
            return False
        
        return True


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

def _get_timeframe_for_bots_logger(logger_name):
    """Возвращает текущий таймфрейм для префикса TF:X в логах BOTS, иначе пустую строку."""
    if not logger_name or ('BotsService' not in logger_name and 'bot' not in logger_name.lower()):
        return ''
    try:
        from bot_engine.config_loader import get_current_timeframe
        tf = get_current_timeframe()
        return f" TF:{tf}" if tf else ''
    except Exception:
        return ''


def _get_timeframe_for_ai_logger(logger_name):
    """Возвращает текущий таймфрейм для префикса TF:X в логах [AI], иначе пустую строку."""
    if not logger_name:
        return ''
    is_ai = (
        logger_name.startswith('AI.') or
        logger_name == 'AI.Main' or
        (logger_name.lower().startswith('ai') or 'ai.' in logger_name.lower())
    )
    if not is_ai:
        return ''
    try:
        from bot_engine.config_loader import get_current_timeframe
        tf = get_current_timeframe()
        return f" TF:{tf}" if tf else ''
    except Exception:
        return ''


def _get_timeframe_for_app_logger(logger_name):
    """Возвращает текущий таймфрейм для префикса TF:X в логах [APP], иначе пустую строку."""
    if not logger_name:
        return ''
    if logger_name.lower() != 'app' and 'app' not in logger_name.lower():
        return ''
    try:
        from bot_engine.config_loader import get_current_timeframe
        tf = get_current_timeframe()
        return f" TF:{tf}" if tf else ''
    except Exception:
        return ''


class FileFormatterWithTF(logging.Formatter):
    """Форматтер для файла: для логгеров BOTS, AI и APP добавляет префикс TF:X."""
    
    def format(self, record):
        tf_prefix = (
            _get_timeframe_for_bots_logger(record.name)
            or _get_timeframe_for_ai_logger(record.name)
            or _get_timeframe_for_app_logger(record.name)
        )
        if tf_prefix:
            # Вставляем TF после levelname: ... - LEVEL - TF:X - message
            s = super().format(record)
            # Стандартный формат: asctime - name - levelname - message
            if ' - ' in s:
                parts = s.rsplit(' - ', 1)
                if len(parts) == 2:
                    s = f"{parts[0]} -{tf_prefix} - {parts[1]}"
            return s
        return super().format(record)


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
        
        # Для [BOTS], [AI] и [APP] добавляем текущий таймфрейм в префикс (TF:X)
        if prefix == '[BOTS]':
            tf_prefix = _get_timeframe_for_bots_logger(logger_name)
        elif prefix == '[AI]':
            tf_prefix = _get_timeframe_for_ai_logger(logger_name)
        elif prefix == '[APP]':
            tf_prefix = _get_timeframe_for_app_logger(logger_name)
        else:
            tf_prefix = ''
        
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
        
        # Компактный формат: [PREFIX] HH:MM:SS - LEVEL - [TF:X -] message
        if tf_prefix:
            formatted = f"{colored_prefix} {colored_timestamp} - {colored_level} -{tf_prefix} - {colored_message}"
        else:
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

def setup_color_logging(console_log_levels=None, enable_file_logging=True, log_file=None):
    """
    Настройка цветного логирования
    
    Args:
        console_log_levels: Список настроек уровней логирования для консоли, например:
            ['+INFO', '-WARNING', '+ERROR', '-DEBUG']
            Если None - все уровни разрешены
        enable_file_logging: Включить ли файловое логирование с ротацией (по умолчанию True)
        log_file: Путь к файлу лога (по умолчанию определяется автоматически)
    """
    # Явно используем глобальный sys
    import sys as _sys
    sys = _sys
    
    # Создаем логгер
    logger = logging.getLogger()
    # Устанавливаем минимальный уровень, чтобы все сообщения доходили до фильтра
    logger.setLevel(logging.DEBUG)
    
    # КРИТИЧНО: Добавляем файловый обработчик с ротацией (10MB)
    if enable_file_logging:
        # Определяем файл лога автоматически на основе имени скрипта
        if log_file is None:
            script_name = sys.argv[0] if sys.argv else 'app'
            if 'ai.py' in script_name or 'ai' in script_name.lower():
                log_file = 'logs/ai.log'
            elif 'bots.py' in script_name or 'bots' in script_name.lower():
                log_file = 'logs/bots.log'
            else:
                log_file = 'logs/app.log'
        
        # Проверяем, нет ли уже файлового обработчика для этого файла
        has_file_handler = False
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler_file = getattr(handler, 'baseFilename', '')
                if handler_file and (handler_file.endswith(log_file) or log_file in handler_file):
                    has_file_handler = True
                    break
        
        if not has_file_handler:
            try:
                from utils.log_rotation import RotatingFileHandlerWithSizeLimit
                import os
                # Создаем директорию logs если её нет
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = RotatingFileHandlerWithSizeLimit(
                    filename=log_file,
                    max_bytes=10 * 1024 * 1024,  # 10MB
                    backup_count=0,  # Перезаписываем файл
                    encoding='utf-8'
                )
                file_handler.setLevel(logging.DEBUG)
                # Форматтер для файла (без цветов; для BOTS добавляется префикс TF:X)
                file_formatter = FileFormatterWithTF('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                # Если не удалось добавить файловый обработчик, продолжаем без него
                sys.stderr.write(f"[COLOR_LOGGER] ⚠️ Не удалось добавить файловый обработчик: {e}\n")
    
    # Проверяем, есть ли уже консольный обработчик с нашим фильтром
    # Если есть, обновляем фильтр, но не пересоздаём обработчик
    has_our_handler = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            # Проверяем, есть ли наш фильтр
            for filter_obj in handler.filters:
                if isinstance(filter_obj, LogLevelFilter):
                    has_our_handler = True
                    # Обновляем настройки фильтра, если они изменились
                    # Создаём новый фильтр с новыми настройками
                    new_filter = LogLevelFilter(console_log_levels)
                    # Заменяем старый фильтр на новый
                    handler.removeFilter(filter_obj)
                    handler.addFilter(new_filter)
                    break
    
    # КРИТИЧНО: Даже если есть наш обработчик, нужно удалить ВСЕ другие обработчики без фильтра
    # Это гарантирует, что все логи идут только через наш фильтр
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            # Проверяем, есть ли наш фильтр
            has_our_filter = any(isinstance(f, LogLevelFilter) for f in handler.filters)
            if not has_our_filter:
                logger.removeHandler(handler)
    
    # Если обработчик уже есть и фильтр обновлён, не пересоздаём обработчик
    # КРИТИЧНО: Всё равно обновляем monkey patch callHandlers с текущим console_log_levels,
    # иначе патч остаётся от первого вызова (например из imports_and_globals с None) и INFO показывается
    if has_our_handler:
        # Удаляем обработчики без фильтра из других логгеров
        for existing_logger_name in logging.Logger.manager.loggerDict:
            existing_logger = logging.getLogger(existing_logger_name)
            for handler in existing_logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                    has_our_filter = any(isinstance(f, LogLevelFilter) for f in handler.filters)
                    if not has_our_filter:
                        existing_logger.removeHandler(handler)
            existing_logger.propagate = True
            existing_logger.setLevel(logging.DEBUG)
        # Обновляем патч callHandlers, чтобы использовались текущие console_log_levels
        def _patched_callHandlers_existing(self, record):
            level_filter = LogLevelFilter(console_log_levels)
            if not level_filter.filter(record):
                return
            return logging.Logger._original_callHandlers(self, record)
        if hasattr(logging.Logger, '_original_callHandlers'):
            logging.Logger.callHandlers = _patched_callHandlers_existing
        return logger
    
    # КРИТИЧНО: Удаляем ВСЕ консольные обработчики БЕЗ нашего фильтра из ВСЕХ логгеров
    # Это нужно, чтобы гарантировать, что все логи проходят через наш фильтр
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            # Проверяем, есть ли наш фильтр
            has_our_filter = any(isinstance(f, LogLevelFilter) for f in handler.filters)
            if not has_our_filter:
                logger.removeHandler(handler)
    
    # КРИТИЧНО: Удаляем консольные обработчики из ВСЕХ существующих логгеров
    # Это гарантирует, что все логи идут через корневой логгер с нашим фильтром
    for existing_logger_name in logging.Logger.manager.loggerDict:
        existing_logger = logging.getLogger(existing_logger_name)
        # Удаляем все StreamHandler'ы без нашего фильтра
        for handler in existing_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                has_our_filter = any(isinstance(f, LogLevelFilter) for f in handler.filters)
                if not has_our_filter:
                    existing_logger.removeHandler(handler)
        # Убеждаемся, что все логгеры пропагируют в корневой
        existing_logger.propagate = True
        existing_logger.setLevel(logging.DEBUG)
    
    # Создаем консольный обработчик
    # На Windows используем errors='replace' для обработки эмодзи
    console_handler = logging.StreamHandler(sys.stdout)
    # Устанавливаем кодировку для Windows консоли
    if sys.platform == 'win32' and hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass  # Если не удалось, используем стандартную кодировку
    console_handler.setLevel(logging.DEBUG)  # Устанавливаем минимальный уровень для обработчика
    
    # Применяем фильтр уровней
    # Создаем фильтр всегда (даже для пустого списка или None - это означает "все уровни разрешены")
    level_filter = LogLevelFilter(console_log_levels)
    console_handler.addFilter(level_filter)
    
    # Устанавливаем цветной форматтер
    formatter = ColorFormatter()
    console_handler.setFormatter(formatter)
    
    # Добавляем обработчик к логгеру
    logger.addHandler(console_handler)
    
    # ОТЛАДКА: Проверяем, что обработчик добавлен (только для отладки, можно убрать)
    # sys.stderr.write(f"[COLOR_LOGGER] Обработчик добавлен, всего handlers: {len(logger.handlers)}\n")
    # sys.stderr.write(f"[COLOR_LOGGER] enabled_levels: {level_filter.enabled_levels}\n")
    # sys.stderr.write(f"[COLOR_LOGGER] debug_enabled: {level_filter.debug_enabled}\n")
    
    # Настраиваем уровни для внешних логгеров, чтобы они не шумели
    # Определяем, какие уровни разрешены
    allowed_levels = set()
    if level_filter and level_filter.enabled_levels:
        allowed_levels = level_filter.enabled_levels
    else:
        # Если фильтр не настроен, разрешаем все уровни
        allowed_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    
    # Настраиваем уровни для внешних библиотек
    external_loggers = [
        'urllib3',
        'urllib3.connectionpool',
        'urllib3.util',
        'urllib3.poolmanager',
        'pybit',
        'pybit._http_manager',
        'requests',
        'requests.packages.urllib3',
        'httpcore',
        'httpx',
        'tensorflow',
        'tensorflow.python',
        'tensorflow.core',
        'tensorflow._api',
        'pandas',
        'pandas.io',
        'pandas.core',
        'pandas.core.dtypes',
        'pandas.core.dtypes.cast',
        'flask_cors',
        'flask_cors.core',
        'werkzeug',
        'flask',
        'flask.app',
    ]
    
    # Определяем минимальный разрешенный уровень
    level_priority = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50}
    min_level = min([level_priority.get(level, 50) for level in allowed_levels], default=50)
    
    # Устанавливаем уровень для внешних логгеров
    for logger_name in external_loggers:
        external_logger = logging.getLogger(logger_name)
        # НЕ устанавливаем уровень здесь - оставляем DEBUG, чтобы сообщения доходили до фильтра
        # Фильтрация происходит через LogLevelFilter в обработчике
        # Убеждаемся, что они используют корневой логгер (propagate=True)
        external_logger.propagate = True
        # Удаляем все обработчики из внешних логгеров, чтобы они шли только через корневой логгер с фильтром
        for handler in external_logger.handlers[:]:
            external_logger.removeHandler(handler)
    
    # Также настраиваем уровни для наших логгеров
    our_loggers = [
        'exchanges.exchange_factory',
        'exchanges',
        'root',
        'app',
        'BotsService',
        'API.AI',
        'AI.Main',
        'bot_engine.bot_history',
    ]
    
    for logger_name in our_loggers:
        our_logger = logging.getLogger(logger_name)
        # НЕ удаляем обработчики - пусть они остаются, если есть
        # НЕ устанавливаем уровень здесь - оставляем DEBUG, чтобы сообщения доходили до фильтра
        # Фильтрация происходит через LogLevelFilter в обработчике
        # КРИТИЧНО: propagate=True, чтобы сообщения шли в корневой логгер с фильтром
        our_logger.propagate = True
        our_logger.setLevel(logging.DEBUG)
    
    # НЕ устанавливаем уровень корневого логгера на min_level,
    # так как это предотвратит создание сообщений ниже этого уровня,
    # и фильтр не сможет их обработать.
    # Фильтрация происходит через LogLevelFilter в обработчике.
    
    # КРИТИЧНО: Перехватываем создание новых обработчиков через monkey patching
    # Это гарантирует, что все новые обработчики будут проверяться на наличие нашего фильтра
    # Сохраняем оригинальную функцию только один раз (если еще не сохранена)
    if not hasattr(logging.Logger, '_original_add_handler'):
        logging.Logger._original_add_handler = logging.Logger.addHandler
    
    def _patched_add_handler(self, handler):
        """Перехватывает добавление обработчиков и удаляет те, что без нашего фильтра"""
        # Если это StreamHandler для stdout без нашего фильтра - не добавляем его
        if isinstance(handler, logging.StreamHandler):
            # Проверяем, что это stdout или stderr (оба идут в консоль)
            stream = getattr(handler, 'stream', None)
            if stream in (sys.stdout, sys.stderr):
                has_our_filter = any(isinstance(f, LogLevelFilter) for f in handler.filters)
                if not has_our_filter:
                    # Не добавляем обработчик без нашего фильтра
                    return
        # Иначе добавляем как обычно
        return logging.Logger._original_add_handler(self, handler)
    
    # Применяем monkey patch только если еще не применен
    if logging.Logger.addHandler != _patched_add_handler:
        logging.Logger.addHandler = _patched_add_handler
    
    # КРИТИЧНО: Также перехватываем callHandlers для дополнительной защиты
    if not hasattr(logging.Logger, '_original_callHandlers'):
        logging.Logger._original_callHandlers = logging.Logger.callHandlers
    
    def _patched_callHandlers(self, record):
        """Перехватывает вызов обработчиков и фильтрует записи без нашего фильтра"""
        # КРИТИЧНО: Создаем фильтр один раз для всех проверок
        level_filter = LogLevelFilter(console_log_levels)
        
        # Проверяем, есть ли хотя бы один обработчик с нашим фильтром
        has_our_handler = False
        handlers_to_remove = []
        
        for handler in self.handlers:
            if isinstance(handler, logging.StreamHandler):
                stream = getattr(handler, 'stream', None)
                if stream in (sys.stdout, sys.stderr):
                    has_our_filter = any(isinstance(f, LogLevelFilter) for f in handler.filters)
                    if has_our_filter:
                        has_our_handler = True
                    else:
                        # Обработчик без нашего фильтра - помечаем для удаления
                        handlers_to_remove.append(handler)
        
        # Удаляем обработчики без нашего фильтра
        for handler in handlers_to_remove:
            try:
                self.removeHandler(handler)
            except:
                pass
        
        # КРИТИЧНО: Фильтруем запись ПЕРЕД вызовом обработчиков
        if not level_filter.filter(record):
            # Запись отфильтрована - не вызываем обработчики
            return
        
        # Если есть наш обработчик, вызываем оригинальный метод (он применит фильтры)
        if has_our_handler:
            return logging.Logger._original_callHandlers(self, record)
        else:
            # Если нет нашего обработчика, но запись прошла фильтр, вызываем оригинальный метод
            return logging.Logger._original_callHandlers(self, record)
    
    # Применяем monkey patch для callHandlers
    if logging.Logger.callHandlers != _patched_callHandlers:
        logging.Logger.callHandlers = _patched_callHandlers
    
    # Финальная проверка: удаляем все обработчики без фильтра еще раз
    # (на случай если они были созданы между проверками)
    for existing_logger_name in list(logging.Logger.manager.loggerDict.keys()):
        try:
            existing_logger = logging.getLogger(existing_logger_name)
            for handler in existing_logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    stream = getattr(handler, 'stream', None)
                    if stream in (sys.stdout, sys.stderr):
                        has_our_filter = any(isinstance(f, LogLevelFilter) for f in handler.filters)
                        if not has_our_filter:
                            existing_logger.removeHandler(handler)
        except Exception:
            pass  # Игнорируем ошибки при удалении обработчиков
    
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
