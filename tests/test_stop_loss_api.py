"""
Тестовый скрипт для проверки работы стоп-лоссов через API биржи

Проверяет:
1. Инициализацию exchange
2. Получение позиций
3. Установку стоп-лоссов
4. Установку трейлинг стопов
5. Синхронизацию данных
"""

import sys
import os
import json
from pathlib import Path

# Добавляем корневую директорию в путь
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from exchanges.exchange_factory import ExchangeFactory
from app.config import EXCHANGES
from bots_modules.imports_and_globals import get_exchange, set_exchange
from bots_modules.init_functions import ensure_exchange_initialized
from bots_modules.sync_and_cache import check_missing_stop_losses
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StopLossTest')

def test_exchange_initialization():
    """Тест 1: Проверка инициализации exchange"""
    logger.info("=" * 80)
    logger.info("ТЕСТ 1: Инициализация Exchange")
    logger.info("=" * 80)
    
    try:
        # Инициализация через Factory
        exchange = ExchangeFactory.create_exchange(
            'BYBIT',
            EXCHANGES['BYBIT']['api_key'],
            EXCHANGES['BYBIT']['api_secret']
        )
        
        if not exchange:
            logger.error("❌ Exchange не создан!")
            return False
        
        logger.info(f"✅ Exchange создан: {type(exchange)}")
        
        # Устанавливаем в GlobalState
        set_exchange(exchange)
        logger.info("✅ Exchange установлен в GlobalState")
        
        # Проверяем через get_exchange()
        current_exchange = get_exchange()
        if current_exchange:
            logger.info(f"✅ Exchange получен через get_exchange(): {type(current_exchange)}")
        else:
            logger.error("❌ get_exchange() вернул None!")
            return False
        
        # Проверяем через ensure_exchange_initialized()
        if ensure_exchange_initialized():
            logger.info("✅ ensure_exchange_initialized() вернул True")
        else:
            logger.error("❌ ensure_exchange_initialized() вернул False!")
            return False
        
        # Тест подключения
        try:
            account_info = exchange.get_unified_account_info()
            logger.info(f"✅ Тест подключения успешен. Баланс: {account_info.get('totalWalletBalance', 'N/A')} USDT")
        except Exception as e:
            logger.warning(f"⚠️ Тест подключения не удался: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_get_positions():
    """Тест 2: Получение позиций с биржи"""
    logger.info("=" * 80)
    logger.info("ТЕСТ 2: Получение позиций с биржи")
    logger.info("=" * 80)
    
    try:
        exchange = get_exchange()
        if not exchange:
            logger.error("❌ Exchange недоступен!")
            return False
        
        # Получаем позиции через API
        positions_response = exchange.client.get_positions(
            category="linear",
            settleCoin="USDT"
        )
        
        logger.info(f"📊 Ответ API: retCode={positions_response.get('retCode')}")
        logger.info(f"📊 Сообщение: {positions_response.get('retMsg')}")
        
        if positions_response.get('retCode') != 0:
            logger.error(f"❌ Ошибка получения позиций: {positions_response.get('retMsg')}")
            return False
        
        exchange_positions = positions_response.get('result', {}).get('list', [])
        logger.info(f"✅ Получено позиций: {len(exchange_positions)}")
        
        # Показываем активные позиции
        active_positions = []
        for pos in exchange_positions:
            size = float(pos.get('size', 0))
            if abs(size) > 0:
                symbol = pos.get('symbol', '')
                side = pos.get('side', '')
                entry_price = float(pos.get('avgPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                stop_loss = pos.get('stopLoss', '')
                trailing_stop = pos.get('trailingStop', '')
                
                active_positions.append({
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'entry_price': entry_price,
                    'mark_price': mark_price,
                    'stop_loss': stop_loss,
                    'trailing_stop': trailing_stop
                })
                
                logger.info(f"  📈 {symbol} {side}: размер={size}, вход={entry_price}, цена={mark_price}")
                logger.info(f"     SL: {stop_loss if stop_loss else '❌ НЕТ'}, Trailing: {trailing_stop if trailing_stop else '❌ НЕТ'}")
        
        if not active_positions:
            logger.warning("⚠️ Нет активных позиций для тестирования")
            return None
        
        logger.info(f"✅ Найдено активных позиций: {len(active_positions)}")
        return active_positions
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения позиций: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_set_stop_loss(position_data):
    """Тест 3: Установка стоп-лосса"""
    logger.info("=" * 80)
    logger.info("ТЕСТ 3: Установка стоп-лосса")
    logger.info("=" * 80)
    
    try:
        exchange = get_exchange()
        if not exchange:
            logger.error("❌ Exchange недоступен!")
            return False
        
        symbol = position_data['symbol']
        side = position_data['side']
        entry_price = position_data['entry_price']
        
        # Определяем positionIdx
        position_idx = 1 if side == 'Buy' else 2
        
        # Рассчитываем стоп-лосс (5%)
        if side == 'Buy':  # LONG
            stop_price = entry_price * 0.95
        else:  # SHORT
            stop_price = entry_price * 1.05
        
        logger.info(f"📊 Позиция: {symbol} {side}")
        logger.info(f"📊 Цена входа: {entry_price}")
        logger.info(f"📊 Расчетный стоп-лосс: {stop_price} ({'LONG: -5%' if side == 'Buy' else 'SHORT: +5%'})")
        logger.info(f"📊 positionIdx: {position_idx}")
        
        # Устанавливаем стоп-лосс
        stop_result = exchange.client.set_trading_stop(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            stopLoss=str(stop_price)
        )
        
        logger.info(f"📊 Ответ API: {json.dumps(stop_result, indent=2)}")
        
        if stop_result.get('retCode') == 0:
            logger.info(f"✅ Стоп-лосс успешно установлен: {stop_price}")
            
            # Проверяем установку - получаем позицию снова
            positions_response = exchange.client.get_positions(
                category="linear",
                symbol=symbol,
                settleCoin="USDT"
            )
            
            if positions_response.get('retCode') == 0:
                positions = positions_response.get('result', {}).get('list', [])
                for pos in positions:
                    if pos.get('symbol') == symbol and abs(float(pos.get('size', 0))) > 0:
                        current_stop_loss = pos.get('stopLoss', '')
                        if current_stop_loss:
                            logger.info(f"✅ Проверка: Стоп-лосс на бирже = {current_stop_loss}")
                            return True
                        else:
                            logger.warning(f"⚠️ Стоп-лосс не найден на бирже после установки")
                            return False
            
            return True
        else:
            error_msg = stop_result.get('retMsg', 'Unknown error')
            ret_code = stop_result.get('retCode', 'Unknown')
            logger.error(f"❌ Ошибка установки стоп-лосса: {error_msg} (retCode={ret_code})")
            
            # Проверяем код ошибки 34040 (not modified)
            if ret_code == 34040 or "not modified" in error_msg.lower():
                logger.info("ℹ️ Стоп-лосс уже установлен на эту цену (это нормально)")
                return True
            
            return False
        
    except Exception as e:
        logger.error(f"❌ Ошибка установки стоп-лосса: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_set_trailing_stop(position_data):
    """Тест 4: Установка трейлинг стопа"""
    logger.info("=" * 80)
    logger.info("ТЕСТ 4: Установка трейлинг стопа")
    logger.info("=" * 80)
    
    try:
        exchange = get_exchange()
        if not exchange:
            logger.error("❌ Exchange недоступен!")
            return False
        
        symbol = position_data['symbol']
        side = position_data['side']
        entry_price = position_data['entry_price']
        mark_price = position_data['mark_price']
        
        # Определяем positionIdx
        position_idx = 1 if side == 'Buy' else 2
        
        # Рассчитываем процент прибыли
        if side == 'Buy':  # LONG
            profit_percent = ((mark_price - entry_price) / entry_price) * 100
        else:  # SHORT
            profit_percent = ((entry_price - mark_price) / entry_price) * 100
        
        logger.info(f"📊 Позиция: {symbol} {side}")
        logger.info(f"📊 Цена входа: {entry_price}, Текущая: {mark_price}")
        logger.info(f"📊 Прибыль: {profit_percent:.2f}%")
        
        # Трейлинг стоп устанавливается только при прибыли >= 3%
        if profit_percent < 3.0:
            logger.warning(f"⚠️ Прибыль {profit_percent:.2f}% < 3%, трейлинг стоп не будет установлен")
            logger.info("ℹ️ Для теста требуется позиция в прибыли >= 3%")
            return None
        
        # Устанавливаем трейлинг стоп (1.5%)
        trailing_distance = 0.015  # 1.5% в десятичной форме
        
        logger.info(f"📊 Устанавливаем трейлинг стоп: {trailing_distance * 100}%")
        logger.info(f"📊 positionIdx: {position_idx}")
        
        # Устанавливаем трейлинг стоп
        trailing_result = exchange.client.set_trading_stop(
            category="linear",
            symbol=symbol,
            positionIdx=position_idx,
            trailingStop=str(trailing_distance)
        )
        
        logger.info(f"📊 Ответ API: {json.dumps(trailing_result, indent=2)}")
        
        if trailing_result.get('retCode') == 0:
            logger.info(f"✅ Трейлинг стоп успешно установлен: {trailing_distance * 100}%")
            
            # Проверяем установку
            positions_response = exchange.client.get_positions(
                category="linear",
                symbol=symbol,
                settleCoin="USDT"
            )
            
            if positions_response.get('retCode') == 0:
                positions = positions_response.get('result', {}).get('list', [])
                for pos in positions:
                    if pos.get('symbol') == symbol and abs(float(pos.get('size', 0))) > 0:
                        current_trailing = pos.get('trailingStop', '')
                        if current_trailing:
                            logger.info(f"✅ Проверка: Трейлинг стоп на бирже = {current_trailing}")
                            return True
                        else:
                            logger.warning(f"⚠️ Трейлинг стоп не найден на бирже после установки")
                            return False
            
            return True
        else:
            error_msg = trailing_result.get('retMsg', 'Unknown error')
            ret_code = trailing_result.get('retCode', 'Unknown')
            logger.error(f"❌ Ошибка установки трейлинг стопа: {error_msg} (retCode={ret_code})")
            
            if ret_code == 34040 or "not modified" in error_msg.lower():
                logger.info("ℹ️ Трейлинг стоп уже установлен (это нормально)")
                return True
            
            return False
        
    except Exception as e:
        logger.error(f"❌ Ошибка установки трейлинг стопа: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_check_missing_stop_losses_function():
    """Тест 5: Проверка функции check_missing_stop_losses()"""
    logger.info("=" * 80)
    logger.info("ТЕСТ 5: Вызов функции check_missing_stop_losses()")
    logger.info("=" * 80)
    
    try:
        result = check_missing_stop_losses()
        
        if result:
            logger.info("✅ Функция check_missing_stop_losses() выполнилась успешно")
        else:
            logger.error("❌ Функция check_missing_stop_losses() вернула False")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Ошибка выполнения check_missing_stop_losses(): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Основная функция тестирования"""
    logger.info("🚀 ЗАПУСК ТЕСТОВ СТОП-ЛОССОВ API")
    logger.info("=" * 80)
    
    results = {
        'exchange_init': False,
        'get_positions': None,
        'set_stop_loss': None,
        'set_trailing_stop': None,
        'check_function': False
    }
    
    # Тест 1: Инициализация
    results['exchange_init'] = test_exchange_initialization()
    if not results['exchange_init']:
        logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: Exchange не инициализирован. Остановка тестов.")
        return
    
    # Тест 2: Получение позиций
    positions = test_get_positions()
    results['get_positions'] = positions
    
    if positions is False:
        logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось получить позиции. Остановка тестов.")
        return
    
    if not positions:
        logger.warning("⚠️ Нет активных позиций. Тесты установки стоп-лоссов пропущены.")
    else:
        # Используем первую активную позицию для тестов
        test_position = positions[0]
        logger.info(f"📊 Используем позицию для тестов: {test_position['symbol']} {test_position['side']}")
        
        # Тест 3: Установка стоп-лосса
        results['set_stop_loss'] = test_set_stop_loss(test_position)
        
        # Тест 4: Установка трейлинг стопа (если есть прибыль)
        results['set_trailing_stop'] = test_set_trailing_stop(test_position)
    
    # Тест 5: Проверка основной функции
    results['check_function'] = test_check_missing_stop_losses_function()
    
    # Итоговый отчет
    logger.info("=" * 80)
    logger.info("📊 ИТОГОВЫЙ ОТЧЕТ")
    logger.info("=" * 80)
    logger.info(f"✅ Инициализация Exchange: {'✅ PASS' if results['exchange_init'] else '❌ FAIL'}")
    logger.info(f"✅ Получение позиций: {'✅ PASS' if results['get_positions'] is not False else '❌ FAIL'}")
    if results['set_stop_loss'] is not None:
        logger.info(f"✅ Установка стоп-лосса: {'✅ PASS' if results['set_stop_loss'] else '❌ FAIL'}")
    if results['set_trailing_stop'] is not None:
        logger.info(f"✅ Установка трейлинг стопа: {'✅ PASS' if results['set_trailing_stop'] else '❌ FAIL'}")
    logger.info(f"✅ Функция check_missing_stop_losses(): {'✅ PASS' if results['check_function'] else '❌ FAIL'}")
    logger.info("=" * 80)
    
    # Общий результат
    all_passed = (
        results['exchange_init'] and
        results['get_positions'] is not False and
        results['check_function']
    )
    
    if all_passed:
        logger.info("🎉 ВСЕ КРИТИЧЕСКИЕ ТЕСТЫ ПРОЙДЕНЫ!")
    else:
        logger.error("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ!")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Тестирование прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

