"""
Обработка торговых сигналов
"""

import logging

logger = logging.getLogger('SignalProcessor')

# Импорт функции логирования сигналов (с fallback если недоступна)
try:
    from bot_engine.bot_history import log_bot_signal
except ImportError:
    def log_bot_signal(*args, **kwargs): pass

def get_effective_signal(coin, config):
    """
    Определяет эффективный сигнал монеты с учетом фильтров

    Args:
        coin: Данные монеты
        config: Конфигурация автобота

    Returns:
        str: Эффективный сигнал (ENTER_LONG, ENTER_SHORT, WAIT)
    """
    symbol = coin.get('symbol', 'UNKNOWN')

    # Получаем настройки
    # ✅ ИСПРАВЛЕНО: Используем False по умолчанию (как в bot_config.py), а не True
    avoid_down_trend = config.get('avoid_down_trend', False)
    avoid_up_trend = config.get('avoid_up_trend', False)
    rsi_long_threshold = config.get('rsi_long_threshold', 29)
    rsi_short_threshold = config.get('rsi_short_threshold', 71)

    # Получаем данные монеты
    from bot_engine.config_loader import get_rsi_from_coin_data, get_trend_from_coin_data
    rsi = get_rsi_from_coin_data(coin)
    trend = get_trend_from_coin_data(coin)

    # Если базовый сигнал WAIT (из-за незрелости) - возвращаем сразу
    base_signal = coin.get('signal', 'WAIT')
    if base_signal == 'WAIT':
        return 'WAIT'

    # Проверяем Enhanced RSI сигнал
    enhanced_rsi = coin.get('enhanced_rsi', {})
    if enhanced_rsi.get('enabled') and enhanced_rsi.get('enhanced_signal'):
        signal = enhanced_rsi.get('enhanced_signal')
    else:
        signal = base_signal

    if signal == 'WAIT':
        return signal

    # Проверка трендов
    if signal == 'ENTER_SHORT' and avoid_up_trend and rsi >= rsi_short_threshold and trend == 'UP':

        return 'WAIT'

    if signal == 'ENTER_LONG' and avoid_down_trend and rsi <= rsi_long_threshold and trend == 'DOWN':

        return 'WAIT'

    # ✅ УБРАНО избыточное логирование - оно дублируется в других местах
    # 
    return signal

def check_autobot_filters(symbol, signal, coin_data, config, maturity_check_func, exit_scam_check_func, position_check_func):
    """
    Проверяет все фильтры для автобота

    Args:
        symbol: Символ монеты
        signal: Торговый сигнал
        coin_data: Данные монеты
        config: Конфигурация
        maturity_check_func: Функция проверки зрелости
        exit_scam_check_func: Функция проверки ExitScam
        position_check_func: Функция проверки существующих позиций
    """
    try:
        # 1. Проверка зрелости
        if not maturity_check_func(symbol):

            return False

        # 2. Проверка ExitScam
        if not exit_scam_check_func(symbol, coin_data):
            logger.warning(f"[AUTOBOT_FILTER] {symbol}: ❌ БЛОКИРОВКА: ExitScam фильтр")
            return False

        # 3. Проверка тренда
        from bot_engine.config_loader import get_trend_from_coin_data
        trend = get_trend_from_coin_data(coin_data)
        # ✅ ИСПРАВЛЕНО: Используем False по умолчанию (как в bot_config.py), а не True
        avoid_down_trend = config.get('avoid_down_trend', False)
        avoid_up_trend = config.get('avoid_up_trend', False)

        if signal == 'ENTER_LONG' and avoid_down_trend and trend == 'DOWN':

            return False

        if signal == 'ENTER_SHORT' and avoid_up_trend and trend == 'UP':

            return False

        # 4. Проверка существующих позиций
        if not position_check_func(symbol, signal):

            return False

        return True

    except Exception as e:
        logger.error(f"[AUTOBOT_FILTER] {symbol}: Ошибка проверки фильтров: {e}")
        return False

def process_auto_bot_signals(coins_rsi_data, bots_data, config, filter_check_func, create_bot_func):
    """
    Обрабатывает сигналы автобота для создания новых ботов

    Args:
        coins_rsi_data: Данные монет с RSI
        bots_data: Данные существующих ботов
        config: Конфигурация автобота
        filter_check_func: Функция проверки фильтров
        create_bot_func: Функция создания бота
    """
    try:
        # Проверяем, включен ли автобот
        auto_bot_enabled = config['enabled']

        if not auto_bot_enabled:

            return

        max_concurrent = config['max_concurrent']
        current_active = sum(1 for bot in bots_data.values() 
                           if bot['status'] not in ['idle', 'paused'])

        if current_active >= max_concurrent:

            return

        logger.info(" 🔍 Проверка сигналов для создания новых ботов...")

        # Получаем монеты с сигналами
        potential_coins = []
        for symbol, coin_data in coins_rsi_data.items():
            from bot_engine.config_loader import get_rsi_from_coin_data, get_trend_from_coin_data
            rsi = get_rsi_from_coin_data(coin_data)
            trend = get_trend_from_coin_data(coin_data)

            if rsi is None:
                continue

            # Проверяем сигналы
            rsi_long_threshold = config.get('rsi_long_threshold', 29)
            rsi_short_threshold = config.get('rsi_short_threshold', 71)

            signal = None
            if rsi <= rsi_long_threshold:
                signal = 'ENTER_LONG'
            elif rsi >= rsi_short_threshold:
                signal = 'ENTER_SHORT'

            if signal:
                # Логируем обнаруженный сигнал
                price = coin_data.get('price', 0)
                log_bot_signal(symbol, signal, rsi, price, {
                    'trend': trend,
                    'source': 'autobot_scanner'
                })

                # Проверяем фильтры
                if filter_check_func(symbol, signal, coin_data):
                    potential_coins.append({
                        'symbol': symbol,
                        'rsi': rsi,
                        'trend': trend,
                        'signal': signal,
                        'coin_data': coin_data
                    })

        logger.info(f" 🎯 Найдено {len(potential_coins)} потенциальных сигналов")

        # Создаем ботов
        created_bots = 0
        for coin in potential_coins[:max_concurrent - current_active]:
            symbol = coin['symbol']

            # Проверяем, нет ли уже бота
            if symbol in bots_data:

                continue

            # Создаем бота
            try:
                logger.info(f" 🚀 Создаем бота для {symbol} ({coin['signal']}, RSI: {coin['rsi']:.1f})")
                create_bot_func(symbol)
                created_bots += 1

            except Exception as e:
                # Блокировка фильтрами - это нормальная работа системы, логируем как WARNING
                error_str = str(e)
                if 'заблокирован фильтрами' in error_str or 'filters_blocked' in error_str:
                    logger.warning(f" ⚠️ Ошибка создания бота для {symbol}: {e}")
                else:
                    logger.error(f" ❌ Ошибка создания бота для {symbol}: {e}")

        if created_bots > 0:
            logger.info(f" ✅ Создано {created_bots} новых ботов")

    except Exception as e:
        logger.error(f" ❌ Ошибка обработки сигналов: {e}")
