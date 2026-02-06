"""
Настройки для standalone AI лаунчера (ai.py).

По аналогии с SystemConfig из bot_config.py, но с фокусом на режимы
обучения/оркестрации и отладочные флаги, которые не нужны основному bots.py.
"""

import os


def _memory_limit_mb_from_env():
    """Лимит ОЗУ в MB из переменной окружения AI_MEMORY_LIMIT_MB (0 = не задан)."""
    try:
        v = os.environ.get('AI_MEMORY_LIMIT_MB', '').strip()
        return int(v) if v else 0
    except ValueError:
        return 0


class AILauncherConfig:
    """Базовые параметры поведения ai.py."""

    # Общие флаги
    DEBUG_MODE = True  # Переводит логгеры в DEBUG и включает дополнительные сообщения
    VERBOSE_PROCESS_LIFECYCLE = True  # Логировать запуск/остановку дочерних процессов подробнее
    LOG_SIGNAL_EVENTS = True  # Печатать стек/детали при получении сигналов остановки
    
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
    CONSOLE_LOG_LEVELS = ['-DEBUG']  # По умолчанию все уровни разрешены

    # Управление трейсингом кода (каждая строка)
    ENABLE_CODE_TRACING = False  # ⚠️ Будет сильно замедлять работу, как и в bots.py
    TRACE_INCLUDE_KEYWORDS = [
        'ai_launcher_source',
        'bot_engine/ai',
        'bot_engine\\ai',
        'bot_engine/ai_backtester_new',
        'bot_engine\\ai_backtester_new',
        'bot_engine/ai_strategy_optimizer',
        'bot_engine\\ai_strategy_optimizer',
        'bot_engine/ai_backtester',
        'bot_engine\\ai_backtester',
        'bots_modules',
        'license_generator',
        'trace_debug',
    ]
    TRACE_SKIP_KEYWORDS = [
        'logging',
        'threading',
        'concurrent',
        'asyncio',
        'json',
        'http',
        'requests',
        'site-packages',
    ]
    TRACE_WRITE_TO_FILE = True
    TRACE_LOG_FILE = 'logs/ai_trace.log'
    TRACE_MAX_LINE_LENGTH = 200

    # Доп. опции можно расширять позднее (например, управление интервалами, автотренером и т.д.)

    # Интервалы циклов (по умолчанию делаем минимальные, чтобы не «засыпать»)
    BACKTEST_LOOP_DELAY_SECONDS = 1      # Пауза между бэктестами (0 = сразу следующий цикл)
    OPTIMIZER_LOOP_DELAY_SECONDS = 1     # Пауза между оптимизациями
    TRAINING_LOOP_DELAY_SECONDS = 1      # Доп. пауза в training worker (если нужно)
    DATA_COLLECTION_INTERVAL_SECONDS = 60  # Уже используется, но можно перенастроить здесь

    # Ожидание готовности данных (data_service_status.ready). Используется лаунчером, если поддерживается.
    # При таймауте лаунчер пишет «Не удалось дождаться готовности данных» и продолжает обучение.
    DATA_READY_WAIT_TIMEOUT_SECONDS = 120

    # ========================================================================
    # ОГРАНИЧЕНИЕ ОЗУ (снижение потребления памяти ai.py)
    # ========================================================================
    # Лимит задаётся в общем конфиге bot_config.SystemConfig или переменными окружения.
    # В конфиге: SystemConfig.AI_MEMORY_PCT = 30  (30% ОЗУ) или SystemConfig.AI_MEMORY_LIMIT_MB = 2048.
    # В env: AI_MEMORY_PCT=35, AI_MEMORY_LIMIT_MB=2048. Приоритет у процента.
    # MEMORY_LIMIT_MB: сюда подставляется значение (ai.py при старте пишет его в env из конфига/env).
    # На Linux/macOS: при старте вызывается resource.setrlimit(RLIMIT_AS).
    # На Windows: setrlimit недоступен; только снижение нагрузки (потоки, символы, свечи).
    MEMORY_LIMIT_MB = _memory_limit_mb_from_env()

    # При заданном MEMORY_LIMIT_MB — жёсткие лимиты нагрузки (на Windows нет setrlimit,
    # поэтому только так можно приблизить потребление к целевому; учитываем несколько процессов).
    CANDLES_LOADER_MAX_WORKERS = 2 if MEMORY_LIMIT_MB else 10   # потоков при загрузке свечей
    MAX_SYMBOLS_FOR_CANDLES = 50 if MEMORY_LIMIT_MB else 100   # макс. символов в get_all_candles_dict
    MAX_CANDLES_PER_SYMBOL = 300 if MEMORY_LIMIT_MB else 1000   # макс. свечей на символ
    # Размер батча при обучении моделей (LSTM, pattern_detector и т.д.) — меньше батч = меньше пик ОЗУ
    TRAINING_BATCH_SIZE = 16 if MEMORY_LIMIT_MB else 32


class AITrainingStrategyConfig:
    """
    Параметры тренировочного режима AI.

    Эти оверрайды применяются только в ai.py / симуляциях, чтобы не трогать боевые
    настройки bots.py. По умолчанию почти все фильтры выключены, а AITrainer может
    самостоятельно включать/выключать их в процессе поиска лучших стратегий.
    """

    ENABLED = True

    # Значения, которыми перекрываем DEFAULT_AUTO_BOT_CONFIG при обучении
    PARAM_OVERRIDES = {
        # Фильтры
        'trend_detection_enabled': False,
        'avoid_down_trend': False,
        'avoid_up_trend': False,
        'exit_scam_enabled': False,
        'enable_maturity_check': False,
        'rsi_time_filter_enabled': False,
        'enhanced_rsi_enabled': False,
        'enhanced_rsi_require_volume_confirmation': False,
        'enhanced_rsi_require_divergence_confirmation': False,
        'enhanced_rsi_use_stoch_rsi': False,
        'limit_orders_entry_enabled': False,
        # Риск/позиции
        'max_position_hours': 0,
        'break_even_protection': False,
        'ai_optimal_entry_enabled': False,
    }

    # Какие флаги можно мутировать/переключать в процессе обучения
    MUTABLE_FILTERS = {
        'trend_detection_enabled': True,
        'avoid_down_trend': True,
        'avoid_up_trend': True,
        'exit_scam_enabled': True,
        'enable_maturity_check': True,
        'rsi_time_filter_enabled': True,
        'enhanced_rsi_enabled': True,
    }