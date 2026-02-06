"""
Безопасная версия тестового скрипта для проверки работы стоп-лоссов через API

ТОЛЬКО ЧТЕНИЕ - не устанавливает реальные стоп-лоссы!
Проверяет только возможность работы с API.
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
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StopLossTestSafe')

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
    """Тест 2: Получение позиций с биржи и анализ стоп-лоссов"""
    logger.info("=" * 80)
    logger.info("ТЕСТ 2: Получение позиций и анализ стоп-лоссов (ТОЛЬКО ЧТЕНИЕ)")
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
        
        # Анализируем позиции
        active_positions = []
        positions_without_sl = []
        
        for pos in exchange_positions:
            size = float(pos.get('size', 0))
            if abs(size) > 0:
                symbol = pos.get('symbol', '')
                side = pos.get('side', '')
                entry_price = float(pos.get('avgPrice', 0))
                mark_price = float(pos.get('markPrice', 0))
                stop_loss = pos.get('stopLoss', '')
                trailing_stop = pos.get('trailingStop', '')
                
                position_info = {
                    'symbol': symbol,
                    'side': side,
                    'size': size,
                    'entry_price': entry_price,
                    'mark_price': mark_price,
                    'stop_loss': stop_loss,
                    'trailing_stop': trailing_stop,
                    'positionIdx': pos.get('positionIdx', 0)
                }
                
                active_positions.append(position_info)
                
                # Рассчитываем процент прибыли/убытка
                if side == 'Buy':  # LONG
                    profit_percent = ((mark_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    profit_percent = ((entry_price - mark_price) / entry_price) * 100
                
                logger.info(f"  📈 {symbol} {side}: размер={size}, вход={entry_price}, цена={mark_price}")
                logger.info(f"     PnL: {profit_percent:.2f}%")
                logger.info(f"     SL: {stop_loss if stop_loss else '❌ ОТСУТСТВУЕТ'}, Trailing: {trailing_stop if trailing_stop else '❌ ОТСУТСТВУЕТ'}")
                
                if not stop_loss:
                    positions_without_sl.append(position_info)
                    logger.warning(f"     ⚠️ ВНИМАНИЕ: У этой позиции НЕТ стоп-лосса!")
                
                # Проверяем возможность установки стоп-лосса (расчет без установки)
                if side == 'Buy':  # LONG
                    calculated_sl = entry_price * 0.95
                else:  # SHORT
                    calculated_sl = entry_price * 1.05
                
                logger.info(f"     💡 Расчетный SL (5%): {calculated_sl:.6f}")
        
        logger.info(f"✅ Найдено активных позиций: {len(active_positions)}")
        
        if positions_without_sl:
            logger.warning(f"⚠️ Позиций БЕЗ стоп-лосса: {len(positions_without_sl)}")
            for pos in positions_without_sl:
                logger.warning(f"   - {pos['symbol']} {pos['side']}")
        else:
            logger.info("✅ У всех позиций есть стоп-лоссы")
        
        return {
            'total': len(active_positions),
            'without_sl': len(positions_without_sl),
            'positions': active_positions
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения позиций: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_api_method_availability():
    """Тест 3: Проверка доступности методов API"""
    logger.info("=" * 80)
    logger.info("ТЕСТ 3: Проверка доступности методов API")
    logger.info("=" * 80)
    
    try:
        exchange = get_exchange()
        if not exchange:
            logger.error("❌ Exchange недоступен!")
            return False
        
        # Проверяем наличие метода set_trading_stop
        if hasattr(exchange.client, 'set_trading_stop'):
            logger.info("✅ Метод set_trading_stop доступен")
        else:
            logger.error("❌ Метод set_trading_stop НЕ доступен!")
            return False
        
        # Проверяем наличие метода get_positions
        if hasattr(exchange.client, 'get_positions'):
            logger.info("✅ Метод get_positions доступен")
        else:
            logger.error("❌ Метод get_positions НЕ доступен!")
            return False
        
        # Проверяем метод update_stop_loss в exchange
        if hasattr(exchange, 'update_stop_loss'):
            logger.info("✅ Метод exchange.update_stop_loss доступен")
        else:
            logger.warning("⚠️ Метод exchange.update_stop_loss НЕ доступен")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка проверки методов API: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_stop_loss_calculation(position_data):
    """Тест 4: Проверка расчета стоп-лоссов (без установки)"""
    logger.info("=" * 80)
    logger.info("ТЕСТ 4: Расчет стоп-лоссов (БЕЗ УСТАНОВКИ)")
    logger.info("=" * 80)
    
    try:
        symbol = position_data['symbol']
        side = position_data['side']
        entry_price = position_data['entry_price']
        mark_price = position_data['mark_price']
        position_idx = position_data.get('positionIdx', 0)
        
        logger.info(f"📊 Позиция: {symbol} {side}")
        logger.info(f"📊 Цена входа: {entry_price}")
        logger.info(f"📊 Текущая цена: {mark_price}")
        logger.info(f"📊 positionIdx: {position_idx}")
        
        # Рассчитываем стоп-лосс (5%)
        if side == 'Buy':  # LONG
            stop_price = entry_price * 0.95
            stop_percent = -5.0
        else:  # SHORT
            stop_price = entry_price * 1.05
            stop_percent = 5.0
        
        logger.info(f"📊 Расчетный стоп-лосс: {stop_price:.6f} ({stop_percent}% от входа)")
        
        # Рассчитываем расстояние до стоп-лосса
        if side == 'Buy':  # LONG
            distance_to_sl = ((mark_price - stop_price) / mark_price) * 100
        else:  # SHORT
            distance_to_sl = ((stop_price - mark_price) / mark_price) * 100
        
        logger.info(f"📊 Расстояние до SL от текущей цены: {distance_to_sl:.2f}%")
        
        # Проверяем параметры для установки (без установки)
        sl_params = {
            "category": "linear",
            "symbol": symbol,
            "stopLoss": str(round(stop_price, 6)),
            "positionIdx": position_idx
        }
        
        logger.info(f"📊 Параметры для установки SL:")
        logger.info(f"   {json.dumps(sl_params, indent=2)}")
        
        logger.info("✅ Расчет завершен (стоп-лосс НЕ установлен - это безопасный режим)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка расчета стоп-лосса: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Основная функция безопасного тестирования"""
    logger.info("🚀 ЗАПУСК БЕЗОПАСНЫХ ТЕСТОВ СТОП-ЛОССОВ API (ТОЛЬКО ЧТЕНИЕ)")
    logger.info("=" * 80)
    logger.info("⚠️  ВНИМАНИЕ: Этот скрипт НЕ изменяет реальные стоп-лоссы!")
    logger.info("=" * 80)
    
    results = {
        'exchange_init': False,
        'get_positions': None,
        'api_methods': False,
        'calculations': []
    }
    
    # Тест 1: Инициализация
    results['exchange_init'] = test_exchange_initialization()
    if not results['exchange_init']:
        logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: Exchange не инициализирован. Остановка тестов.")
        return False
    
    # Тест 2: Получение позиций
    positions_data = test_get_positions()
    results['get_positions'] = positions_data
    
    if positions_data is False:
        logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось получить позиции. Остановка тестов.")
        return False
    
    # Тест 3: Проверка методов API
    results['api_methods'] = test_api_method_availability()
    
    # Тест 4: Расчет стоп-лоссов для каждой позиции
    if positions_data and positions_data.get('positions'):
        for pos in positions_data['positions']:
            calc_result = test_stop_loss_calculation(pos)
            results['calculations'].append(calc_result)
    
    # Итоговый отчет
    logger.info("=" * 80)
    logger.info("📊 ИТОГОВЫЙ ОТЧЕТ")
    logger.info("=" * 80)
    logger.info(f"✅ Инициализация Exchange: {'✅ PASS' if results['exchange_init'] else '❌ FAIL'}")
    logger.info(f"✅ Получение позиций: {'✅ PASS' if results['get_positions'] is not False else '❌ FAIL'}")
    if results['get_positions'] and isinstance(results['get_positions'], dict):
        logger.info(f"   📊 Всего позиций: {results['get_positions'].get('total', 0)}")
        logger.info(f"   ⚠️ Без стоп-лосса: {results['get_positions'].get('without_sl', 0)}")
    logger.info(f"✅ Проверка методов API: {'✅ PASS' if results['api_methods'] else '❌ FAIL'}")
    logger.info(f"✅ Расчеты стоп-лоссов: {len([r for r in results['calculations'] if r])}/{len(results['calculations'])}")
    logger.info("=" * 80)
    
    # Общий результат
    all_passed = (
        results['exchange_init'] and
        results['get_positions'] is not False and
        results['api_methods']
    )
    
    if all_passed:
        logger.info("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        logger.info("💡 Для тестирования установки стоп-лоссов используйте: python scripts/test_stop_loss_api.py")
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

