"""
ШАБЛОН (только дефолтные значения). НЕ подставляйте сюда свои ключи!

  Копируйте в configs/app_config.py только при первом запуске (если app_config.py нет).
  Рабочий конфиг — configs/app_config.py; не перезаписывайте его из этого файла.
  1. configs/keys.example.py -> configs/keys.py и заполните ключи.
  2. Этот файл -> configs/app_config.py только если app_config.py ещё не создан.
"""
from configs.keys import EXCHANGES, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# ==================== БЛОК 1: СЕРВЕР И ПОРТ ====================
# ВНИМАНИЕ: APP_HOST '0.0.0.0' открывает доступ из интернета. Используйте в доверенной сети.
APP_HOST = '0.0.0.0'                       # Хост приложения (0.0.0.0 = все интерфейсы)
APP_PORT = 5000                            # Порт основного приложения
APP_DEBUG = True                           # Режим отладки (Flask debug)

# ==================== БЛОК 2: БИРЖА ====================
ACTIVE_EXCHANGE = 'BYBIT'                  # Активная биржа: BYBIT | BINANCE | OKX

# ==================== БЛОК 3: ПОЗИЦИИ И PnL ====================
GROWTH_MULTIPLIER = 3.0                    # Множитель для определения быстрого роста позиций
DEFAULT_PNL_THRESHOLD = 1000               # Порог PnL по умолчанию для уведомлений, USDT
MIN_PNL_THRESHOLD = 5                      # Минимальный порог PnL, USDT
HIGH_ROI_THRESHOLD = 100                   # Порог высокого ROI для уведомлений, %
HIGH_LOSS_THRESHOLD = -40                  # Порог большого убытка для уведомлений, USDT

# ==================== БЛОК 4: ОБНОВЛЕНИЕ ДАННЫХ И ГРАФИКИ ====================
UPDATE_INTERVAL = 2000                     # Интервал обновления данных UI, мс
CHART_UPDATE_INTERVAL = 60000              # Интервал обновления графиков, мс
CLOSED_PNL_UPDATE_INTERVAL = 10000        # Интервал обновления закрытого PnL, мс
CHART_MAX_POINTS = 30                      # Макс. точек на графике
CHART_COLORS = {                           # Цвета для графиков (прибыль/убыток)
    'POSITIVE': {
        'BORDER': '#4CAF50',
        'BACKGROUND': 'rgba(74, 175, 80, 0.2)'
    },
    'NEGATIVE': {
        'BORDER': '#f44336',
        'BACKGROUND': 'rgba(244, 67, 54, 0.2)'
    }
}

# Настройки графика статистики
STATISTICS_CHART = {                       # Параметры графика статистики (Chart.js)
    'type': 'line',                        # Тип графика
    'tension': 0.4,                        # Сглаживание линий
    'fill': True,                          # Заливка под линией
    'responsive': True,                    # Адаптивность
    'maintain_aspect_ratio': False,        # Не сохранять пропорции
    'animation': False,                    # Отключить анимацию
    'max_points': CHART_MAX_POINTS,         # Макс. точек
    'height': 600,                         # Высота графика, px
    'scales': {
        'y': {
            'begin_at_zero': False,
            'grid': {
                'color': 'rgba(255, 255, 255, 0.1)'
            }
        },
        'x': {
            'grid': {
                'display': False
            }
        }
    },
    'legend': {
        'display': False
    },
    'colors': CHART_COLORS                  # Цвета из CHART_COLORS
}

DEFAULT_PAGE_SIZE = 10                     # Размер страницы по умолчанию (позиции/таблицы)
AVAILABLE_PAGE_SIZES = [10, 50, 100]       # Доступные размеры страницы
CHART_CACHE_TIME = 5 * 60 * 1000           # Время кэша графиков, мс (5 мин)
SMA200_CACHE_TIME = 7 * 60 * 1000          # Время кэша SMA200, мс (7 мин)
DEFAULT_SORT_ORDER = 'pnl_desc'            # Сортировка по умолчанию
SORT_OPTIONS = {                           # Варианты сортировки для UI
    'pnl_desc': 'PNL (макс-мин)',
    'pnl_asc': 'PNL (мин-макс)',
    'alphabet_asc': 'A-Z',
    'alphabet_desc': 'Z-A'
}

# ==================== БЛОК 5: ТЕМА И UI ====================
DEFAULT_THEME = 'dark'                     # Тема по умолчанию: dark | light
THEME_COLORS = {                           # Цвета тем (фон, текст, секции, границы)
    'dark': {
        'bg_color': '#1a1a1a',
        'text_color': '#fff',
        'section_bg': '#2d2d2d',
        'border_color': '#404040'
    },
    'light': {
        'bg_color': '#f0f0f0',
        'text_color': '#000',
        'section_bg': '#fff',
        'border_color': '#ddd'
    }
}

# ==================== БЛОК 6: ЛОГИРОВАНИЕ ====================
LOG_DIR = 'logs'                           # Директория логов
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'   # Формат строки лога
LOG_LEVEL = 'INFO'                         # Уровень логирования: DEBUG | INFO | WARNING | ERROR

# ==================== БЛОК 7: TELEGRAM И УВЕДОМЛЕНИЯ ====================
TELEGRAM_NOTIFICATIONS_ENABLED = True      # Включить уведомления в Telegram
TELEGRAM_NOTIFY = {                        # Какие события отправлять в Telegram
    'ERRORS': True,              # Уведомления об ошибках
    'RAPID_GROWTH': True,        # Быстрый рост позиций
    'HIGH_ROI': True,            # Высокий ROI (>100%)
    'HIGH_LOSS': True,           # Большие убытки (<-40 USDT)
    'HIGH_PNL': True,            # Высокий PnL (>1000 USDT)
    'DAILY_REPORT': True,        # Ежедневный отчет
    'DAILY_REPORT_TIME': '00:00', # Время отправки дневного отчета
    'STATISTICS': True,          # Отправка статистики
    'STATISTICS_INTERVAL': 300,    # Интервал отправки в секундах (если включено)
    'STATISTICS_INTERVAL_ENABLED': True,  # Включить/выключить отправку по интервалу
    'STATISTICS_TIME_ENABLED': False,     # Включить/выключить отправку в определенное время
    'STATISTICS_TIME': ['09:00', '21:00']
}

# ==================== БЛОК 8: БЭКАПЫ И СИНХРОНИЗАЦИЯ ВРЕМЕНИ ====================
DATABASE_BACKUP = {                        # Настройки фонового бэкапа БД (каждый процесс бэкапит только свою БД)
    'ENABLED': True,
    'INTERVAL_MINUTES': 180,
    'RUN_ON_START': True,
    'APP_ENABLED': True,         # app.py → app_data.db
    'AI_ENABLED': True,          # ai.py → ai_data.db
    'BOTS_ENABLED': True,        # bots.py → bots_data.db
    'BACKUP_DIR': None,
    'MAX_RETRIES': 3,
    'KEEP_LAST_N': 5,
}
TIME_SYNC = {                              # Синхронизация времени с NTP (только Windows)
    'ENABLED': False,            # Включить автоматическую синхронизацию времени
    'INTERVAL_MINUTES': 30,      # Интервал синхронизации в минутах (раз в полчаса)
    'SERVER': 'time.windows.com', # Сервер времени для синхронизации
    'RUN_ON_START': True,        # Синхронизировать сразу при запуске
    'REQUIRE_ADMIN': True         # True = планировщик только при запуске от админа; False = цикл всегда, синхронизация сработает при запуске от админа
}