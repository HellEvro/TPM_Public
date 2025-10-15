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

# EMA параметры
EMA_FAST = 50
EMA_SLOW = 200

# Подтверждение тренда (количество баров)
TREND_CONFIRMATION_BARS = 3

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
    'max_concurrent': 5,
    'risk_cap_percent': 10.0,
    'scope': 'all',  # all | whitelist | blacklist
    'whitelist': [],
    'blacklist': [],
    # RSI параметры согласно ТЗ
    'rsi_long_threshold': 29,   # Вход в LONG при RSI <= 29
    'rsi_short_threshold': 71,  # Вход в SHORT при RSI >= 71
    'rsi_exit_long': 65,        # Выход из LONG при RSI >= 65
    'rsi_exit_short': 35,       # Выход из SHORT при RSI <= 35
    'default_position_size': 10.0,  # Размер позиции в USDT
    'check_interval': 180,      # Интервал проверки в секундах (3 мин = 180 сек)
    'monitoring_interval': 10,  # Интервал мониторинга активных ботов в секундах
    # Торговые настройки
    'trading_enabled': True,    # Включить реальную торговлю
    'use_test_server': False,   # Использовать тестовый сервер
    'max_risk_per_trade': 2.0,  # Максимальный риск на сделку в %
    # Защитные механизмы
    'max_loss_percent': 15.0,   # Максимальный убыток в % от входа (стоп-лосс)
    'trailing_stop_activation': 300.0,  # Активация trailing stop при прибыли в % (x3 = 300%)
    'trailing_stop_distance': 150.0,    # Расстояние trailing stop в % (x1.5 = 150%)
    'max_position_hours': 0,     # Максимальное время удержания позиции в часах (0 = отключено)
    'break_even_protection': True,      # Защита безубыточности
    'break_even_trigger': 100.0,        # Триггер для break even в % (x1 = 100%)
    # Фильтры по тренду
    'avoid_down_trend': True,           # Не входить в LONG при нисходящем тренде
    'avoid_up_trend': True,             # Не входить в SHORT при восходящем тренде
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
    'exit_scam_candles': 10,            # Количество свечей для проверки (10 = 60 часов на 6H)
    'exit_scam_single_candle_percent': 15.0,  # Максимальный % изменения одной свечи (15% = блокировка)
    'exit_scam_multi_candle_count': 4,        # Количество свечей для суммарного анализа
    'exit_scam_multi_candle_percent': 50.0,   # Максимальный суммарный % за N свечей (50% = блокировка)
}

# Настройки по умолчанию для отдельного бота
DEFAULT_BOT_CONFIG = {
    'volume_mode': VolumeMode.FIXED_USDT,
    'volume_value': 10.0,
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
    BOT_STATUS_UPDATE_INTERVAL = 5  # 5 секунд - интервал обновления детальной информации о состоянии ботов (цена входа, SL, TP, ликвидация, PnL)
    INACTIVE_BOT_CLEANUP_INTERVAL = 600  # 10 минут - интервал проверки и удаления неактивных ботов
    INACTIVE_BOT_TIMEOUT = 600  # 10 минут - время ожидания перед удалением бота без реальных позиций на бирже
    
    # Умное обновление RSI
    SMART_RSI_UPDATE = True  # Учитывать время до закрытия свечи
    RSI_CANDLE_CHECK_INTERVAL = 300  # 5 минут для проверки изменений текущей свечи
    
    # Улучшенная система RSI
    ENHANCED_RSI_ENABLED = True  # Включить улучшенную систему RSI для сильных трендов
    ENHANCED_RSI_REQUIRE_VOLUME_CONFIRMATION = True  # Требовать подтверждение объемом
    ENHANCED_RSI_REQUIRE_DIVERGENCE_CONFIRMATION = False  # Требовать подтверждение дивергенцией (строгий режим)
    ENHANCED_RSI_USE_STOCH_RSI = True  # Использовать Stochastic RSI для дополнительного подтверждения
    
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

# Настройки риск-менеджмента
class RiskConfig:
    # Стоп-лосс и защитные механизмы
    STOP_LOSS_PERCENT = 15.0
    TRAILING_STOP_ACTIVATION = 300.0  # x3 от входа
    TRAILING_STOP_DISTANCE = 150.0    # x1.5 от входа
    MAX_POSITION_TIME_HOURS = 48
    SECURITY_PROTECTION = True
    SECURITY_TRIGGER_PERCENT = 100.0
    
    # Auto Bot ограничения
    MAX_CONCURRENT_POSITIONS = 10
    MAX_RISK_PERCENT = 20.0

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