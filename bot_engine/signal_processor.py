"""
Обработка торговых сигналов
"""

import logging

logger = logging.getLogger('SignalProcessor')


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
    avoid_down_trend = config.get('avoid_down_trend', True)
    avoid_up_trend = config.get('avoid_up_trend', True)
    rsi_long_threshold = config.get('rsi_long_threshold', 29)
    rsi_short_threshold = config.get('rsi_short_threshold', 71)
    
    # Получаем данные монеты
    rsi = coin.get('rsi6h', 50)
    trend = coin.get('trend', coin.get('trend6h', 'NEUTRAL'))
    
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
        logger.debug(f"[SIGNAL] {symbol}: ❌ SHORT заблокирован (RSI={rsi:.1f} >= {rsi_short_threshold} + UP тренд)")
        return 'WAIT'
    
    if signal == 'ENTER_LONG' and avoid_down_trend and rsi <= rsi_long_threshold and trend == 'DOWN':
        logger.debug(f"[SIGNAL] {symbol}: ❌ LONG заблокирован (RSI={rsi:.1f} <= {rsi_long_threshold} + DOWN тренд)")
        return 'WAIT'
    
    logger.debug(f"[SIGNAL] {symbol}: ✅ {signal} разрешен (RSI={rsi:.1f}, Trend={trend})")
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
            logger.debug(f"[AUTOBOT_FILTER] {symbol}: Монета незрелая")
            return False
        
        # 2. Проверка ExitScam
        if not exit_scam_check_func(symbol, coin_data):
            logger.warning(f"[AUTOBOT_FILTER] {symbol}: ❌ БЛОКИРОВКА: ExitScam фильтр")
            return False
        else:
            logger.info(f"[AUTOBOT_FILTER] {symbol}: ✅ ExitScam фильтр пройден")
        
        # 3. Проверка тренда
        trend = coin_data.get('trend6h', 'NEUTRAL')
        avoid_down_trend = config.get('avoid_down_trend', True)
        avoid_up_trend = config.get('avoid_up_trend', True)
        
        if signal == 'ENTER_LONG' and avoid_down_trend and trend == 'DOWN':
            logger.debug(f"[AUTOBOT_FILTER] {symbol}: DOWN тренд - не открываем LONG")
            return False
            
        if signal == 'ENTER_SHORT' and avoid_up_trend and trend == 'UP':
            logger.debug(f"[AUTOBOT_FILTER] {symbol}: UP тренд - не открываем SHORT")
            return False
        
        # 4. Проверка существующих позиций
        if not position_check_func(symbol, signal):
            logger.debug(f"[AUTOBOT_FILTER] {symbol}: Уже есть позиция на бирже")
            return False
        
        logger.debug(f"[AUTOBOT_FILTER] {symbol}: ✅ Все фильтры пройдены")
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
            logger.debug("[AUTO_BOT] ⏹️ Автобот выключен")
            return
        
        max_concurrent = config['max_concurrent']
        current_active = sum(1 for bot in bots_data.values() 
                           if bot['status'] not in ['idle', 'paused'])
        
        if current_active >= max_concurrent:
            logger.debug(f"[AUTO_BOT] 🚫 Достигнут лимит активных ботов ({current_active}/{max_concurrent})")
            return
        
        logger.info("[AUTO_BOT] 🔍 Проверка сигналов для создания новых ботов...")
        
        # Получаем монеты с сигналами
        potential_coins = []
        for symbol, coin_data in coins_rsi_data.items():
            rsi = coin_data.get('rsi6h')
            trend = coin_data.get('trend6h', 'NEUTRAL')
            
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
                # Проверяем фильтры
                if filter_check_func(symbol, signal, coin_data):
                    potential_coins.append({
                        'symbol': symbol,
                        'rsi': rsi,
                        'trend': trend,
                        'signal': signal,
                        'coin_data': coin_data
                    })
        
        logger.info(f"[AUTO_BOT] 🎯 Найдено {len(potential_coins)} потенциальных сигналов")
        
        # Создаем ботов
        created_bots = 0
        for coin in potential_coins[:max_concurrent - current_active]:
            symbol = coin['symbol']
            
            # Проверяем, нет ли уже бота
            if symbol in bots_data:
                logger.debug(f"[AUTO_BOT] ⚠️ Бот для {symbol} уже существует")
                continue
            
            # Создаем бота
            try:
                logger.info(f"[AUTO_BOT] 🚀 Создаем бота для {symbol} ({coin['signal']}, RSI: {coin['rsi']:.1f})")
                create_bot_func(symbol)
                created_bots += 1
                
            except Exception as e:
                logger.error(f"[AUTO_BOT] ❌ Ошибка создания бота для {symbol}: {e}")
        
        if created_bots > 0:
            logger.info(f"[AUTO_BOT] ✅ Создано {created_bots} новых ботов")
        
    except Exception as e:
        logger.error(f"[AUTO_BOT] ❌ Ошибка обработки сигналов: {e}")

