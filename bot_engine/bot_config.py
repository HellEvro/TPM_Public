"""
Конфигурация торговых ботов
"""

# RSI параметры
RSI_PERIOD = 14
RSI_OVERSOLD = 29
RSI_OVERBOUGHT = 71

# ✅ RSI зоны для выходов - РАЗДЕЛЬНО для сделок ПО ТРЕНДУ и ПРОТИВ ТРЕНДА
# По тренду - можно ждать большего движения
RSI_EXIT_LONG_WITH_TREND = 65  # Выход из лонга при RSI >= 65 (вход был по UP тренду)
RSI_EXIT_SHORT_WITH_TREND = 35  # Выход из шорта при RSI <= 35 (вход был по DOWN тренду)

# Против тренда - выходим раньше, безопаснее
RSI_EXIT_LONG_AGAINST_TREND = 60  # Выход из лонга при RSI >= 60 (вход был против DOWN тренда)
RSI_EXIT_SHORT_AGAINST_TREND = 40  # Выход из шорта при RSI <= 40 (вход был против UP тренда)

# ❌ СТАРЫЕ RSI_EXIT_LONG и RSI_EXIT_SHORT УДАЛЕНЫ - используйте новые WITH_TREND и AGAINST_TREND

# Расширенные RSI параметры для сильных трендов
RSI_EXTREME_OVERSOLD = 20  # Экстремальная перепроданность
RSI_EXTREME_OVERBOUGHT = 80  # Экстремальная перекупленность
RSI_VOLATILITY_THRESHOLD_HIGH = 1.5  # Высокая волатильность
RSI_VOLATILITY_THRESHOLD_LOW = 0.7   # Низкая волатильность
RSI_DIVERGENCE_LOOKBACK = 10  # Период для поиска дивергенций
RSI_VOLUME_CONFIRMATION_MULTIPLIER = 1.2  # Множитель объема для подтверждения
RSI_STOCH_PERIOD = 14  # Период для стохастического RSI
RSI_EXTREME_ZONE_TIMEOUT = 3  # Максимальное количество свечей в экстремальной зоне без дополнительного подтверждения

# EMA параметры (по умолчанию, если нет оптимальных)
EMA_FAST = 50
EMA_SLOW = 200

# Параметры подтверждения тренда (гибкая настройка)
TREND_CONFIRMATION_BARS = 3  # Количество свечей для подтверждения (по умолчанию)
TREND_MIN_CONFIRMATIONS = 2  # Минимум подтверждений из 3 возможных (мягкие условия)

# Опциональность проверок для определения тренда
TREND_REQUIRE_SLOPE = False  # Требовать наклон EMA_long (False = опциональный критерий)
TREND_REQUIRE_PRICE = True   # Требовать цену выше/ниже EMA_long (True = обязательный)
TREND_REQUIRE_CANDLES = True # Требовать N свечей подряд (True = обязательный)

# Таймфрейм для анализа
TIMEFRAME = '6h'

# Статусы бота
class BotStatus:
    IDLE = 'IDLE'
    IN_POSITION_LONG = 'IN_POSITION_LONG'
    IN_POSITION_SHORT = 'IN_POSITION_SHORT'
    PAUSED = 'PAUSED'

# Направления тренда
class TrendDirection:
    UP = 'UP'
    DOWN = 'DOWN'
    NEUTRAL = 'NEUTRAL'

# Режимы объёма
class VolumeMode:
    FIXED_QTY = 'fixed_qty'
    FIXED_USDT = 'fixed_usdt'
    PERCENT_BALANCE = 'percent_balance'

# Настройки Auto Bot по умолчанию
DEFAULT_AUTO_BOT_CONFIG = {
    'enabled': False,
    'max_concurrent': 10,
    'risk_cap_percent': 10,
    'scope': 'all', # all | whitelist | blacklist
    'whitelist': [],
    'blacklist': [],
    # RSI параметры согласно ТЗ
    'rsi_long_threshold': 29,   # Вход в LONG при RSI <= 29
    'rsi_short_threshold': 71,  # Вход в SHORT при RSI >= 71
    # ✅ Новые параметры RSI выхода с учетом тренда
    'rsi_exit_long_with_trend': 65,      # Выход из LONG при RSI >= 65 (вход по тренду)
    'rsi_exit_long_against_trend': 60,   # Выход из LONG при RSI >= 60 (вход против тренда)
    'rsi_exit_short_with_trend': 35,     # Выход из SHORT при RSI <= 35 (вход по тренду)
    'rsi_exit_short_against_trend': 40,  # Выход из SHORT при RSI <= 40 (вход против тренда)
    'default_position_size': 5,          # Базовый размер позиции (в единицах согласно default_position_mode)
    'default_position_mode': 'percent', # Режим расчета: usdt | percent
    'leverage': 1,                       # ✅ Кредитное плечо (1-125x)
    'check_interval': 180,      # Интервал проверки в секундах (3 мин = 180 сек)
    'monitoring_interval': 10,  # Интервал мониторинга активных ботов в секундах
    # Торговые настройки
    'trading_enabled': True,    # Включить реальную торговлю
    'use_test_server': False,   # Использовать тестовый сервер
    # Защитные механизмы
    'max_loss_percent': 15,   # Максимальный убыток в % от входа (стоп-лосс)
    'take_profit_percent': 70, # Защитный Take Profit в % от входа (рассчитывается как стоп-лосс)
    'trailing_stop_activation': 10, # Процент прибыли, после которого активируется трейлинг
    'trailing_stop_distance': 5,      # Дистанция трейлинга от максимальной цены, %
    'trailing_take_distance': 0.5,    # Резервный trailing-тейк (лимит) в %, когда процесс упадет
    'trailing_update_interval': 3,  # Минимальный интервал обновлений стопов/тейков (сек)
    'max_position_hours': 0,     # Максимальное время удержания позиции в часах (0 = отключено)
    'break_even_protection': True,      # Защита безубыточности
    'break_even_trigger': 20,           # Триггер для break even в % (x1 = 100%, но по умолчанию 20%)
    'break_even_trigger_percent': 5, # Дублирующий ключ для UI (совместимость)
    # Фильтры по тренду
    'trend_detection_enabled': True,    # Включить определение тренда (выключить = пропускает анализ трендов)
    'avoid_down_trend': False,          # Не входить в LONG при нисходящем тренде (КРИТИЧЕСКИ ВАЖНО!)
    'avoid_up_trend': False,            # Не входить в SHORT при восходящем тренде (КРИТИЧЕСКИ ВАЖНО!)
    # Параметры анализа тренда (простой ценовой анализ)
    'trend_analysis_period': 30,       # Количество свечей для анализа тренда (20-50, по умолчанию 30 = 7.5 дней на 6h)
    'trend_price_change_threshold': 7, # Процент изменения цены для определения тренда (3-15%, чем меньше - тем чувствительнее)
    'trend_candles_threshold': 70,     # Сколько % свечей должны расти/падать для тренда (50-80%, 70% = 21 из 30 свечей в одну сторону)
    # Настройки зрелости монет
    'enable_maturity_check': True,      # Включить проверку зрелости монет
    'min_candles_for_maturity': 400,    # Минимум свечей для зрелой монеты (100 дней на 6H)
    'min_rsi_low': 35,                  # Минимальный достигнутый RSI (должен быть <= 35)
    'max_rsi_high': 65,                 # Максимальный достигнутый RSI (должен быть >= 65)
    # RSI временной фильтр
    'rsi_time_filter_enabled': True, # Включить временной фильтр для RSI сигналов
    'rsi_time_filter_candles': 7,       # Минимум свечей с последнего экстремума (4 = 1 день на 6H)
    'rsi_time_filter_upper': 65,        # Верхняя граница спокойной зоны для SHORT
    'rsi_time_filter_lower': 35,        # Нижняя граница спокойной зоны для LONG
    # Набор позиций лимитными ордерами
    'limit_orders_entry_enabled': False, # Включить набор позиций лимитными ордерами (False = рыночный вход)
    'limit_orders_percent_steps': [0, 0.5, 1, 1.5, 2], # Шаги в % от цены входа (0 = рыночный, далее лимитные с увеличивающимся отступом)
    'limit_orders_margin_amounts': [5, 5, 5, 5, 5], # Объем маржи в USDT для каждого ордера (минимум 5 USDT на бирже Bybit, иначе ордер будет отклонен)
    # ExitScam фильтр (защита от резких движений цены)
    'exit_scam_enabled': True,          # Включить проверку на ExitScam
    'exit_scam_candles': 8,            # Количество свечей для проверки (10 = 60 часов на 6H)
    'exit_scam_single_candle_percent': 15,  # Максимальный % изменения одной свечи (15% = блокировка)
    'exit_scam_multi_candle_count': 4,        # Количество свечей для суммарного анализа
    'exit_scam_multi_candle_percent': 50,   # Максимальный суммарный % за N свечей (50% = блокировка)
    # 🤖 ИИ настройки (премиум функции)
    'ai_optimal_entry_enabled': False,  # ИИ определение оптимальной точки входа (выкл. по умолчанию)
    'ai_enabled': True, # Включить подтверждение сигналов AI
    'ai_min_confidence': 0.7,          # Минимальная уверенность AI (0.0-1.0)
    'ai_override_original': True,      # AI может блокировать решения скрипта,
    'anomaly_block_threshold': 0.7,
    'anomaly_detection_enabled': True,
    'anomaly_log_enabled': True,
    'auto_refresh_ui': True,
    'auto_retrain': True,
    'auto_save_interval': 30,
    'auto_train_enabled': True,
    'auto_update_data': True,
    'data_update_interval': 86400,
    'debug_mode': False,
    'enhanced_rsi_enabled': True,
    'enhanced_rsi_require_divergence_confirmation': True,
    'enhanced_rsi_require_volume_confirmation': True,
    'enhanced_rsi_use_stoch_rsi': True,
    'inactive_bot_cleanup_interval': 600,
    'inactive_bot_timeout': 600,
    'log_anomalies': True,
    'log_patterns': True,
    'log_predictions': True,
    'lstm_enabled': True,
    'lstm_min_confidence': 0.6,
    'lstm_weight': 1.5,
    'min_volatility_threshold': 0.05,
    'pattern_enabled': True,
    'pattern_min_confidence': 0.6,
    'pattern_weight': 1.2,
    'position_sync_interval': 30,
    'refresh_interval': 3,
    'retrain_hour': 3,
    'retrain_interval': 604800,
    'risk_management_enabled': True,
    'risk_update_interval': 300,
    'rsi_divergence_lookback': 10,
    'rsi_extreme_overbought': 80,
    'rsi_extreme_oversold': 20,
    'rsi_extreme_zone_timeout': 3,
    'rsi_update_interval': 300,
    'rsi_volume_confirmation_multiplier': 1.2,
    'stop_loss_setup_interval': 300,
}

# Настройки по умолчанию для отдельного бота
DEFAULT_BOT_CONFIG = {
    'volume_mode': VolumeMode.FIXED_USDT,
    'volume_value': 5.0,
    'status': BotStatus.IDLE,
    'auto_managed': False,
    'max_loss_percent': 2.0,  # Максимальная потеря в процентах для стоп-лосса
    'leverage': 10  # ✅ Кредитное плечо по умолчанию
}

# Системные настройки
class SystemConfig:
    # Интервалы обновления (в секундах)
    RSI_UPDATE_INTERVAL = 300 # 30 минут (рекомендуется для 6H RSI)
    ACCOUNT_UPDATE_INTERVAL = 5  # 5 секунд
    UI_REFRESH_INTERVAL = 3  # 3 секунды
    AUTO_SAVE_INTERVAL = 30  # 30 секунд
    BOT_STATUS_UPDATE_INTERVAL = 1  # 1 секунда - интервал обновления детальной информации о состоянии ботов (цена входа, SL, TP, ликвидация, PnL)
    INACTIVE_BOT_CLEANUP_INTERVAL = 600  # 10 минут - интервал проверки и удаления неактивных ботов
    INACTIVE_BOT_TIMEOUT = 600  # 10 минут - время ожидания перед удалением бота без реальных позиций на бирже
    STOP_LOSS_SETUP_INTERVAL = 300  # 5 минут - интервал установки недостающих стоп-лоссов
    POSITION_SYNC_INTERVAL = 30  # 30 секунд - интервал синхронизации позиций с биржей
    
    # Умное обновление RSI
    SMART_RSI_UPDATE = True  # Учитывать время до закрытия свечи
    RSI_CANDLE_CHECK_INTERVAL = 300  # 5 минут для проверки изменений текущей свечи
    
    # Улучшенная система RSI
    ENHANCED_RSI_ENABLED = True # Включить улучшенную систему RSI для сильных трендов (ОТКЛЮЧЕНО для тестирования)
    ENHANCED_RSI_REQUIRE_VOLUME_CONFIRMATION = True  # Требовать подтверждение объемом
    ENHANCED_RSI_REQUIRE_DIVERGENCE_CONFIRMATION = True  # Требовать подтверждение дивергенцией (строгий режим)
    ENHANCED_RSI_USE_STOCH_RSI = True  # Использовать Stochastic RSI для дополнительного подтверждения
    
    # Настройки Enhanced RSI
    RSI_EXTREME_ZONE_TIMEOUT = 3  # Таймаут экстремальной зоны (свечей)
    RSI_EXTREME_OVERSOLD = 20  # Экстремальная перепроданность
    RSI_EXTREME_OVERBOUGHT = 80  # Экстремальная перекупленность
    RSI_VOLUME_CONFIRMATION_MULTIPLIER = 1.2  # Множитель объёма
    RSI_DIVERGENCE_LOOKBACK = 10  # Период поиска дивергенций
    
    # Торговые параметры RSI согласно техзаданию
    RSI_OVERSOLD = 29  # Зона покупки (LONG при RSI <= 29)
    RSI_OVERBOUGHT = 71  # Зона продажи (SHORT при RSI >= 71)
    
    # ✅ RSI зоны для выходов - РАЗДЕЛЬНО для сделок ПО ТРЕНДУ и ПРОТИВ ТРЕНДА
    # По тренду - можно ждать большего движения
    RSI_EXIT_LONG_WITH_TREND = 65  # Выход из лонга при RSI >= 65 (вход был по UP тренду)
    RSI_EXIT_SHORT_WITH_TREND = 35  # Выход из шорта при RSI <= 35 (вход был по DOWN тренду)
    
    # Против тренда - выходим раньше, безопаснее
    RSI_EXIT_LONG_AGAINST_TREND = 60  # Выход из лонга при RSI >= 60 (вход был против DOWN тренда)
    RSI_EXIT_SHORT_AGAINST_TREND = 40  # Выход из шорта при RSI <= 40 (вход был против UP тренда)
    
    # ❌ СТАРЫЕ RSI_EXIT_LONG и RSI_EXIT_SHORT УДАЛЕНЫ - используйте новые WITH_TREND и AGAINST_TREND
    
    # ❌ EMA и старые TREND параметры больше не используются (тренд определяется простым анализом цены)
    TREND_CONFIRMATION_BARS = 3  # Количество свечей для подтверждения (по умолчанию)
    TREND_MIN_CONFIRMATIONS = 2  # Минимум подтверждений из 3 возможных (мягкие условия)
    
    # Опциональность проверок для определения тренда
    TREND_REQUIRE_SLOPE = False  # Требовать наклон EMA_long (False = опциональный критерий, дает +1 балл)
    TREND_REQUIRE_PRICE = True   # Требовать цену выше/ниже EMA_long (True = обязательный критерий)
    TREND_REQUIRE_CANDLES = True # Требовать N свечей подряд (True = обязательный критерий)
    
    # Константы для фильтрации зрелости монет
    MIN_CANDLES_FOR_MATURITY = 400  # Минимум свечей для зрелой монеты (100 дней на 6H)
    MIN_RSI_LOW = 35   # Минимальный достигнутый RSI
    MAX_RSI_HIGH = 65  # Максимальный достигнутый RSI
    MIN_VOLATILITY_THRESHOLD = 0.05  # Минимальная волатильность (5%)
    
    # Настройки API
    BOTS_SERVICE_PORT = 5001
    BOTS_SERVICE_HOST = '0.0.0.0'  # Доступ из сети
    MAIN_APP_PORT = 5000
    MAIN_APP_HOST = '127.0.0.1'
    REQUEST_TIMEOUT = 30
    
    # Настройки UI
    AUTO_REFRESH_UI = True
    PRESERVE_FILTERS = True
    TOAST_DURATION = 3000  # миллисекунды
    
    # Отладка
    DEBUG_MODE = False
    
    # ========================================================================
    # НАСТРОЙКИ УРОВНЕЙ ЛОГИРОВАНИЯ В КОНСОЛИ
    # ========================================================================
    # Позволяет фильтровать логи, выводимые в консоль (stdout).
    # Файловые логи (logs/*.log) НЕ затрагиваются - туда пишется всё.
    #
    # ДОСТУПНЫЕ УРОВНИ:
    #   DEBUG    - отладочная информация (самый подробный)
    #   INFO     - информационные сообщения
    #   WARNING  - предупреждения
    #   ERROR    - ошибки
    #   CRITICAL - критические ошибки (самый важный)
    #
    # ФОРМАТ НАСТРОЙКИ:
    #   Можно использовать два варианта:
    #   1. Список строк: ['+INFO', '-WARNING', '+ERROR']
    #   2. Одна строка с запятыми: "+INFO, -WARNING, +ERROR"
    #
    # СИНТАКСИС:
    #   +LEVEL  - включить уровень (показывать в консоли)
    #   -LEVEL  - выключить уровень (скрыть из консоли)
    #
    # ЛОГИКА РАБОТЫ:
    #   1. Если указаны уровни с префиксом + (включить):
    #      - Показываются ТОЛЬКО указанные уровни
    #      - Все остальные автоматически скрываются
    #      Пример: ['+ERROR'] → показывается только ERROR
    #
    #   2. Если указаны ТОЛЬКО уровни с префиксом - (выключить):
    #      - Показываются ВСЕ уровни КРОМЕ указанных
    #      Пример: ['-DEBUG', '-INFO'] → показываются WARNING, ERROR, CRITICAL
    #
    #   3. Если список пустой, строка пустая или None:
    #      - Показываются ВСЕ уровни (фильтрация отключена)
    #
    # ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:
    #
    #   # Пример 1: Показывать только ошибки
    #   CONSOLE_LOG_LEVELS = ['+ERROR']
    #   # или
    #   CONSOLE_LOG_LEVELS = "+ERROR"
    #
    #   # Пример 2: Показывать только ERROR и WARNING
    #   CONSOLE_LOG_LEVELS = ['+ERROR', '+WARNING']
    #   # или
    #   CONSOLE_LOG_LEVELS = "+ERROR, +WARNING"
    #
    #   # Пример 3: Показывать всё кроме DEBUG
    #   CONSOLE_LOG_LEVELS = ['-DEBUG']
    #   # или
    #   CONSOLE_LOG_LEVELS = "-DEBUG"
    #
    #   # Пример 4: Показывать всё кроме DEBUG и INFO
    #   CONSOLE_LOG_LEVELS = ['-DEBUG', '-INFO']
    #   # или
    #   CONSOLE_LOG_LEVELS = "-DEBUG, -INFO"
    #
    #   # Пример 5: Явно указать все нужные уровни
    #   CONSOLE_LOG_LEVELS = ['+INFO', '+WARNING', '+ERROR', '+CRITICAL']
    #   # или
    #   CONSOLE_LOG_LEVELS = "+INFO, +WARNING, +ERROR, +CRITICAL"
    #
    #   # Пример 6: Смешанный синтаксис (если есть +, то - игнорируются)
    #   CONSOLE_LOG_LEVELS = ['+ERROR', '-DEBUG', '-INFO']
    #   # Результат: показывается только ERROR (т.к. есть +ERROR)
    #
    #   # Пример 7: Отключить фильтрацию (показывать всё)
    #   CONSOLE_LOG_LEVELS = []
    #   # или
    #   CONSOLE_LOG_LEVELS = ""
    #
    # ВАЖНО:
    #   - Настройка применяется ТОЛЬКО к консольному выводу
    #   - В файлы logs/*.log пишется всё независимо от этой настройки
    #   - Регистр не важен: '+error' = '+ERROR' = '+Error'
    #   - Пробелы вокруг запятых игнорируются
    # ========================================================================
    CONSOLE_LOG_LEVELS = []  # По умолчанию все уровни разрешены
    
    # ⚡ ТРЕЙСИНГ: Включить детальное логирование КАЖДОЙ строки кода (для отладки зависаний)
    ENABLE_CODE_TRACING = False  # ⚠️ ВНИМАНИЕ: Сильно замедляет работу! Включать только для отладки!
    TRACE_INCLUDE_KEYWORDS = [
        'bots_modules',
        'bot_engine',
        'InfoBot',
        'exchanges',
        'scripts',
        'launcher',
    ]
    TRACE_SKIP_KEYWORDS = [
        'logging',
        'threading',
        'queue',
        'socket',
        'ssl',
        'http',
        'urllib',
        'json',
        'datetime',
        'traceback',
        'site-packages',
        'AppData',
        'Python3',
    ]
    TRACE_WRITE_TO_FILE = True
    TRACE_LOG_FILE = 'logs/trace.log'
    TRACE_MAX_LINE_LENGTH = 200

# Настройки риск-менеджмента
class RiskConfig:
    # Стоп-лосс и защитные механизмы
    STOP_LOSS_PERCENT = 15.0
    TRAILING_STOP_ACTIVATION = 20.0  # Процент прибыли для активации трейлинга
    TRAILING_STOP_DISTANCE = 150.0    # x1.5 от входа
    MAX_POSITION_TIME_HOURS = 48
    SECURITY_PROTECTION = True
    SECURITY_TRIGGER_PERCENT = 100.0
    
    # Автоматические корректировки
    PRICE_SLIPPAGE_BUFFER = 5.0  # Процент проскальзывания для страховки (5% по умолчанию)
    
    # Auto Bot ограничения
    MAX_CONCURRENT_POSITIONS = 10
    MAX_RISK_PERCENT = 20.0
    
    # Премиум функции (требуют лицензии)
    STOP_ANALYSIS_ENABLED = True       # Анализ стопов для обучения ИИ
    BACKTEST_ENABLED = True            # Бэктестинг перед входом
    SMART_RISK_MANAGEMENT = True       # Умный риск-менеджмент с бэктестом
    AI_OPTIMAL_ENTRY_ENABLED = True

# Настройки фильтров
class FilterConfig:
    # Списки монет
    WHITELIST = []
    BLACKLIST = []
    
    # RSI фильтры
    MIN_RSI = 0
    MAX_RSI = 100
    
    # Тренды
    ALLOWED_TRENDS = ['UP', 'DOWN', 'NEUTRAL']

# Настройки биржи
class ExchangeConfig:
    DEFAULT_EXCHANGE = 'bybit'
    SUPPORTED_EXCHANGES = ['bybit', 'binance', 'okx']
    FUTURES_ONLY = True

# ==========================================
# ИИ МОДУЛИ (ПРЕМИУМ ФУНКЦИИ)
# ==========================================

class AIConfig:
    """
    Настройки ИИ модулей
    
    ИИ функции являются премиум дополнением и требуют лицензии.
    Для активации лицензии: python scripts/activate_premium.py
    """
    
    # Общие настройки
    AI_ENABLED = True
    AI_CONFIDENCE_THRESHOLD = 0.65  # Минимальная уверенность для применения рекомендации ИИ (0.0-1.0)
    
    # Anomaly Detection - обнаружение аномалий (pump/dump)
    AI_ANOMALY_DETECTION_ENABLED = True
    AI_ANOMALY_MODEL_PATH = 'data/ai/models/anomaly_detector.pkl'
    AI_ANOMALY_SCALER_PATH = 'data/ai/models/anomaly_scaler.pkl'
    AI_ANOMALY_BLOCK_THRESHOLD = 0.7
    
    # LSTM Predictor - предсказание движения цены
    AI_LSTM_ENABLED = True
    AI_LSTM_MODEL_PATH = 'data/ai/models/lstm_predictor.keras'  # ✅ Keras 3 формат
    AI_LSTM_SCALER_PATH = 'data/ai/models/lstm_scaler.pkl'
    AI_LSTM_WEIGHT = 1.5
    AI_LSTM_MIN_CONFIDENCE = 0.6
    
    # Pattern Recognition - распознавание графических паттернов
    AI_PATTERN_ENABLED = True
    AI_PATTERN_MODEL_PATH = 'data/ai/models/pattern_detector.pkl'
    AI_PATTERN_SCALER_PATH = 'data/ai/models/pattern_scaler.pkl'
    AI_PATTERN_WEIGHT = 1.2
    AI_PATTERN_MIN_CONFIDENCE = 0.6
    
    # Dynamic Risk Management - умный SL/TP
    AI_RISK_MANAGEMENT_ENABLED = True
    AI_RISK_MODEL_PATH = 'data/ai/models/risk_manager.h5'
    AI_RISK_UPDATE_INTERVAL = 300
    
    # Кэширование предсказаний
    AI_CACHE_PREDICTIONS = True
    AI_CACHE_TTL = 300  # Время жизни кэша в секундах (5 минут)
    
    # Логирование ИИ решений
    AI_LOG_PREDICTIONS = True
    AI_LOG_ANOMALIES = True
    AI_LOG_PATTERNS = True
    
    # ==========================================
    # АВТОМАТИЧЕСКОЕ ОБУЧЕНИЕ
    # ==========================================
    
    # Включить автоматическое обучение
    AI_AUTO_TRAIN_ENABLED = True
    
    # Первичная настройка при старте (если модель не найдена)
    AI_AUTO_TRAIN_ON_STARTUP = True  # True = автоматически собрать данные и обучить модель
    
    # Частота обновления данных
    AI_AUTO_UPDATE_DATA = True
    AI_DATA_UPDATE_INTERVAL = 86400
    AI_UPDATE_COINS_COUNT = 0  # 0 = ВСЕ монеты с биржи (инкрементальное обновление)
    AI_INITIAL_COINS_COUNT = 0  # 0 = ВСЕ монеты при первичной настройке (полная история)
    
    # Частота переобучения модели
    AI_AUTO_RETRAIN = True
    AI_RETRAIN_INTERVAL = 604800
    
    # Время запуска обучения (по умолчанию - ночью)
    AI_RETRAIN_HOUR = 3

