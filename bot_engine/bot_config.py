"""
Конфигурация торговых ботов
"""

# RSI параметры
RSI_PERIOD = 14
RSI_OVERSOLD = 29
RSI_OVERBOUGHT = 71
RSI_EXIT_LONG = 65
RSI_EXIT_SHORT = 35

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
    'enabled': True,
    'max_concurrent': 10,
    'risk_cap_percent': 10,
    'scope': 'all',  # all | whitelist | blacklist
    'whitelist': [],
    'blacklist': [],
    # RSI параметры согласно ТЗ
    'rsi_long_threshold': 29,   # Вход в LONG при RSI <= 29
    'rsi_short_threshold': 71,  # Вход в SHORT при RSI >= 71
    'rsi_exit_long': 60,        # Выход из LONG при RSI >= 65
    'rsi_exit_short': 40,       # Выход из SHORT при RSI <= 35
    'default_position_size': 5,  # Размер позиции в USDT
    'check_interval': 180,      # Интервал проверки в секундах (3 мин = 180 сек)
    'monitoring_interval': 10,  # Интервал мониторинга активных ботов в секундах
    # Торговые настройки
    'trading_enabled': True,    # Включить реальную торговлю
    'use_test_server': False,   # Использовать тестовый сервер
    'max_risk_per_trade': 2,  # Максимальный риск на сделку в %
    # Защитные механизмы
    'max_loss_percent': 15,   # Максимальный убыток в % от входа (стоп-лосс)
    'trailing_stop_activation': 50,  # Активация trailing stop при прибыли в % (x3 = 300%)
    'trailing_stop_distance': 15,    # Расстояние trailing stop в % (x1.5 = 150%)
    'max_position_hours': 0,     # Максимальное время удержания позиции в часах (0 = отключено)
    'break_even_protection': True,      # Защита безубыточности
    'break_even_trigger': 100,        # Триггер для break even в % (x1 = 100%)
    # Фильтры по тренду
    'trend_detection_enabled': True,    # Включить определение тренда (выключить = пропускает анализ трендов)
    'avoid_down_trend': True,          # Не входить в LONG при нисходящем тренде (КРИТИЧЕСКИ ВАЖНО!)
    'avoid_up_trend': True,            # Не входить в SHORT при восходящем тренде (КРИТИЧЕСКИ ВАЖНО!)
    # Настройки зрелости монет
    'enable_maturity_check': True,      # Включить проверку зрелости монет
    'min_candles_for_maturity': 400,    # Минимум свечей для зрелой монеты (100 дней на 6H)
    'min_rsi_low': 35,                  # Минимальный достигнутый RSI (должен быть <= 35)
    'max_rsi_high': 65,                 # Максимальный достигнутый RSI (должен быть >= 65)
    # RSI временной фильтр
    'rsi_time_filter_enabled': True,    # Включить временной фильтр для RSI сигналов
    'rsi_time_filter_candles': 4,       # Минимум свечей с последнего экстремума (4 = 1 день на 6H)
    'rsi_time_filter_upper': 65,        # Верхняя граница спокойной зоны для SHORT
    'rsi_time_filter_lower': 35,        # Нижняя граница спокойной зоны для LONG
    # ExitScam фильтр (защита от резких движений цены)
    'exit_scam_enabled': True,          # Включить проверку на ExitScam
    'exit_scam_candles': 8,            # Количество свечей для проверки (10 = 60 часов на 6H)
    'exit_scam_single_candle_percent': 15,  # Максимальный % изменения одной свечи (15% = блокировка)
    'exit_scam_multi_candle_count': 4,        # Количество свечей для суммарного анализа
    'exit_scam_multi_candle_percent': 50,   # Максимальный суммарный % за N свечей (50% = блокировка)
    # 🤖 ИИ настройки (премиум функции)
    'ai_optimal_entry_enabled': False,  # ИИ определение оптимальной точки входа (выкл. по умолчанию)
}

# Настройки по умолчанию для отдельного бота
DEFAULT_BOT_CONFIG = {
    'volume_mode': VolumeMode.FIXED_USDT,
    'volume_value': 5.0,
    'status': BotStatus.IDLE,
    'auto_managed': False,
    'max_loss_percent': 2.0  # Максимальная потеря в процентах для стоп-лосса
}

# Системные настройки
class SystemConfig:
    # Интервалы обновления (в секундах)
    RSI_UPDATE_INTERVAL = 300  # 30 минут (рекомендуется для 6H RSI)
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
    ENHANCED_RSI_ENABLED = False  # Включить улучшенную систему RSI для сильных трендов (ОТКЛЮЧЕНО для тестирования)
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
    RSI_EXIT_LONG = 65  # Выход из лонга (при RSI >= 65)
    RSI_EXIT_SHORT = 35  # Выход из шорта (при RSI <= 35)
    
    # EMA параметры для анализа тренда (дефолтные, если нет оптимальных)
    EMA_FAST = 50  # Быстрая EMA
    EMA_SLOW = 200  # Медленная EMA
    
    # Параметры подтверждения тренда (гибкая настройка)
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
    
    # ⚡ ТРЕЙСИНГ: Включить детальное логирование КАЖДОЙ строки кода (для отладки зависаний)
    ENABLE_CODE_TRACING = False  # ⚠️ ВНИМАНИЕ: Сильно замедляет работу! Включать только для отладки!

# Настройки риск-менеджмента
class RiskConfig:
    # Стоп-лосс и защитные механизмы
    STOP_LOSS_PERCENT = 15.0
    TRAILING_STOP_ACTIVATION = 300.0  # x3 от входа
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
