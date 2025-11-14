"""
Сбор исторических данных для обучения ИИ моделей

Скачивает свечи 6H для всех монет с биржи за последние 1-2 года.
Данные сохраняются в CSV формате в data/ai/historical/

Использование:
    python scripts/ai/collect_historical_data.py --limit 20    # Топ 20 монет
    python scripts/ai/collect_historical_data.py --limit 50    # Топ 50 монет
    python scripts/ai/collect_historical_data.py --all         # Все монеты
"""

import sys
sys.path.append('.')

from exchanges.exchange_factory import ExchangeFactory
from app.config import EXCHANGES
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
import logging
import argparse

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_data_for_coin(exchange, symbol, days=730):
    """
    Собирает исторические данные для одной монеты с пагинацией
    
    Args:
        exchange: Объект биржи
        symbol: Символ монеты (например, 'BTC')
        days: Количество дней истории (730 = 2 года)
    
    Returns:
        Количество собранных свечей
    """
    all_candles = []
    
    # Для 6H свечей:
    # - 1 день = 4 свечи
    # - 730 дней = 2920 свечей
    # - API отдает до 1000 свечей за запрос
    # - Нужно ~3 запроса для 2 лет
    
    required_candles = days * 4  # 4 свечи 6H в день
    candles_per_request = 1000
    total_requests = (required_candles // candles_per_request) + 1
    
    # Начинаем с текущего времени и идем в прошлое
    end_time = int(time.time() * 1000)  # Миллисекунды
    
    for request_num in range(total_requests):
        try:
            # Вызываем get_kline напрямую с пагинацией
            response = exchange.client.get_kline(
                category="linear",
                symbol=f"{symbol}USDT",
                interval="360",  # 6H в минутах
                limit=1000,
                end=end_time  # Получаем свечи ДО этого времени
            )
            
            # Проверка rate limiting
            if response.get('retCode') == 10006:
                logger.warning(f"[{symbol}] Rate limit, ждем 5 секунд...")
                time.sleep(5)
                continue
            
            if response.get('retCode') == 0:
                klines = response.get('result', {}).get('list', [])
                
                if not klines:
                    logger.info(f"[{symbol}] Запрос {request_num+1}/{total_requests}: данных больше нет")
                    break
                
                # Преобразуем в формат свечей
                for k in klines:
                    candle = {
                        'timestamp': int(k[0]),
                        'time': int(k[0]),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5])
                    }
                    all_candles.append(candle)
                
                # Выводим прогресс только для первого и последнего запроса
                if request_num == 0 or request_num == total_requests - 1 or len(klines) < 1000:
                    print(f"  Request {request_num+1}/{total_requests}: {len(klines)} candles (total: {len(all_candles)})")
                
                # Обновляем end_time для следующего запроса (берем timestamp самой старой свечи)
                oldest_timestamp = int(klines[-1][0])  # Последняя свеча в списке - самая старая
                end_time = oldest_timestamp - 1  # Минус 1 мс чтобы не получить ту же свечу
                
                # Если получили меньше чем лимит, значит данных больше нет
                if len(klines) < 1000:
                    logger.info(f"[{symbol}] Получено меньше лимита - данных больше нет")
                    break
            else:
                logger.warning(f"[{symbol}] Ошибка API: {response.get('retMsg', 'Unknown')}")
                break
            
            # Rate limiting между запросами
            time.sleep(0.5)
        
        except Exception as e:
            logger.error(f"[{symbol}] Ошибка в запросе {request_num+1}: {e}")
            break
    
    # Сохраняем данные
    if all_candles:
        new_df = pd.DataFrame(all_candles)
        
        # Убираем дубликаты в новых данных
        new_df = new_df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Подготавливаем путь к файлу
        output_dir = Path('data/ai/historical')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{symbol}_6h_historical.csv'
        
        # ИНКРЕМЕНТАЛЬНОЕ ОБНОВЛЕНИЕ: объединяем со старыми данными
        if output_file.exists():
            try:
                # Загружаем существующие данные
                old_df = pd.read_csv(output_file)
                
                # Объединяем старые + новые
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
                
                # Убираем дубликаты (оставляем более новую запись)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                
                # Сортируем от старых к новым
                combined_df = combined_df.sort_values('timestamp')
                
                # Сохраняем объединённые данные
                combined_df.to_csv(output_file, index=False)
                
                new_count = len(combined_df) - len(old_df)
                print(f"  Updated: {len(old_df)} -> {len(combined_df)} (+{new_count} new candles)")
                return new_count
            except Exception as e:
                print(f"  [WARNING] Error loading old data: {e}, overwriting...")
                # Если ошибка при чтении старого файла, перезаписываем
                new_df.to_csv(output_file, index=False)
                print(f"  Saved: {len(new_df)} candles (overwritten)")
                return len(new_df)
        else:
            # Файл не существует, создаём новый
            new_df = new_df.sort_values('timestamp')
            new_df.to_csv(output_file, index=False)
            print(f"  Created: {len(new_df)} candles")
            return len(new_df)
    
    print(f"  [WARNING] No data to save")
    return 0


def get_top_coins(exchange, limit=None):
    """
    Получает список всех монет с биржи
    
    Args:
        exchange: Объект биржи
        limit: Количество монет (None = все)
    
    Returns:
        Список символов
    """
    logger.info(f"Получаем список монет с биржи...")
    
    try:
        # Используем get_all_pairs() из BybitExchange
        all_pairs = exchange.get_all_pairs()
        
        if all_pairs:
            logger.info(f"✅ Получено {len(all_pairs)} монет с биржи")
            
            # Ограничиваем если нужно
            if limit:
                result = all_pairs[:limit]
                logger.info(f"Ограничено до {limit} монет")
                return result
            
            return all_pairs
        else:
            logger.warning("Не удалось получить монеты с биржи, используем дефолтный список")
            return get_default_coins()[:limit] if limit else get_default_coins()
    
    except Exception as e:
        logger.error(f"Ошибка получения списка монет: {e}")
        logger.warning("Используем дефолтный список монет")
        return get_default_coins()[:limit] if limit else get_default_coins()


def get_default_coins():
    """Возвращает дефолтный список монет"""
    return [
        'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOT', 'LINK', 'MATIC',
        'AVAX', 'UNI', 'ATOM', 'XRP', 'DOGE', 'SHIB', 'APE', 'SAND',
        'NEAR', 'FTM', 'AAVE', 'GRT', 'ALGO', 'MANA', 'CRV', 'LDO',
        'ARB', 'OP', 'APT', 'SUI', 'SEI', 'TIA'
    ]


def main():
    """Основная функция"""
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Сбор исторических данных для ИИ')
    parser.add_argument('--limit', type=int, default=20, help='Количество монет (по умолчанию: 20)')
    parser.add_argument('--all', action='store_true', help='Собрать для всех монет')
    parser.add_argument('--days', type=int, default=730, help='Количество дней истории (по умолчанию: 730)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HISTORICAL DATA COLLECTION")
    print("=" * 60)
    print()
    
    # Параметры сбора
    days_history = args.days
    
    logger.info(f"Параметры:")
    logger.info(f"  - История: {days_history} дней ({days_history//365} года)")
    logger.info(f"  - Таймфрейм: 6h")
    print()
    
    # Инициализируем биржу
    logger.info("Инициализация биржи...")
    try:
        exchange = ExchangeFactory.create_exchange(
            'BYBIT',
            EXCHANGES['BYBIT']['api_key'],
            EXCHANGES['BYBIT']['api_secret']
        )
        logger.info("✅ Биржа инициализирована: BYBIT")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации биржи: {e}")
        return
    
    print()
    
    # Получаем список монет с биржи
    logger.info("Получение списка монет с биржи...")
    all_symbols = get_top_coins(exchange, limit=None)
    
    logger.info(f"✅ Получено {len(all_symbols)} монет с биржи")
    print()
    
    # Определяем сколько монет собирать
    if args.all:
        symbols = all_symbols
        logger.info(f"Режим: ВСЕ монеты ({len(symbols)})")
    else:
        symbols = all_symbols[:args.limit]
        logger.info(f"Режим: Топ {args.limit} монет")
    
    print()
    logger.info(f"Будет собрано данных для {len(symbols)} монет")
    
    # Показываем первые 10
    for i, symbol in enumerate(symbols[:10], 1):
        print(f"  {i}. {symbol}")
    if len(symbols) > 10:
        print(f"  ... и еще {len(symbols) - 10} монет")
    print()
    
    # Оценка времени
    estimated_time_minutes = len(symbols) * 0.5  # ~30 секунд на монету
    logger.info(f"Примерное время: {estimated_time_minutes:.0f} минут")
    print()
    
    # Собираем данные
    start_time = time.time()
    successful = 0
    failed = 0
    total_candles = 0
    
    print("=" * 60)
    print("STARTING DATA COLLECTION")
    print("=" * 60)
    print()
    
    for i, symbol in enumerate(symbols, 1):
        # Прогресс-бар
        progress = (i / len(symbols)) * 100
        elapsed = time.time() - start_time
        eta = (elapsed / i) * (len(symbols) - i) if i > 0 else 0
        
        print(f"\n[{i}/{len(symbols)}] {symbol} | Progress: {progress:.1f}% | ETA: {eta/60:.1f}m")
        print("-" * 60)
        
        try:
            candles_count = collect_data_for_coin(exchange, symbol, days=days_history)
            
            if candles_count > 0:
                successful += 1
                total_candles += candles_count
                print(f"[OK] {symbol}: {candles_count} candles collected")
            else:
                failed += 1
                print(f"[WARNING] {symbol}: No data available")
        
        except Exception as e:
            failed += 1
            print(f"[ERROR] {symbol}: {e}")
        
        # Краткая статистика каждые 10 монет
        if i % 10 == 0:
            print()
            print("=" * 60)
            print(f"CHECKPOINT: {i}/{len(symbols)} coins processed")
            print(f"Success: {successful} | Failed: {failed} | Candles: {total_candles:,}")
            print(f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
            print("=" * 60)
    
    # Итоговая статистика
    elapsed_total = time.time() - start_time
    
    # Подсчитываем РЕАЛЬНОЕ количество свечей во всех файлах
    print("\nПодсчет общего количества свечей...")
    actual_total_candles = 0
    historical_dir = Path("data/ai/historical")
    
    if historical_dir.exists():
        for csv_file in historical_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                actual_total_candles += len(df) - 1  # -1 для заголовка
            except:
                pass
    
    print()
    print("=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Успешно: {successful}/{len(symbols)} монет")
    print(f"Ошибок: {failed}/{len(symbols)}")
    print(f"Новых свечей добавлено: {total_candles:,}")
    print(f"Всего свечей в базе: {actual_total_candles:,}")
    print(f"Время выполнения: {elapsed_total:.0f} секунд ({elapsed_total/60:.1f} минут)")
    print()
    print(f"Данные сохранены в: data/ai/historical/")
    print()
    print("Следующий шаг:")
    print("  python scripts/ai/train_anomaly_on_real_data.py")
    print()


if __name__ == '__main__':
    main()

