"""
Скрипт для запуска backtesting торговой стратегии

Использует исторические данные для тестирования производительности
AI модулей и торговой логики.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Добавляем корневую директорию в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot_engine.ai.backtester import BacktestEngine


def load_historical_data(data_dir: str = "data/ai/historical", symbols: list = None) -> dict:
    """
    Загружает исторические данные для бэктеста
    
    Args:
        data_dir: Директория с CSV файлами
        symbols: Список символов для загрузки (None = все)
    
    Returns:
        Словарь {symbol: DataFrame}
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        return {}
    
    csv_files = sorted(data_path.glob("*.csv"))
    
    historical_data = {}
    loaded_count = 0
    
    print(f"\nLoading historical data...")
    print("=" * 60)
    
    for csv_file in csv_files:
        symbol = csv_file.stem.replace('_6h_historical', '')
        
        # Фильтруем по списку символов
        if symbols and symbol not in symbols:
            continue
        
        try:
            df = pd.read_csv(csv_file)
            
            # Проверяем наличие необходимых колонок
            required_cols = ['close', 'high', 'low', 'open']
            if not all(col in df.columns for col in required_cols):
                print(f"  [SKIP] {symbol}: Missing required columns")
                continue
            
            # Проверяем наличие timestamp
            if 'timestamp' not in df.columns and 'time' not in df.columns:
                print(f"  [SKIP] {symbol}: No timestamp column")
                continue
            
            # Стандартизируем название timestamp
            if 'time' in df.columns and 'timestamp' not in df.columns:
                df['timestamp'] = df['time']
            
            historical_data[symbol] = df
            loaded_count += 1
            
            if loaded_count % 50 == 0:
                print(f"  Loaded {loaded_count} symbols...")
        
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
    
    print("=" * 60)
    print(f"[OK] Loaded {len(historical_data)} symbols")
    print()
    
    return historical_data


def main():
    """Основная функция запуска backtesting"""
    parser = argparse.ArgumentParser(description='Запуск backtesting')
    parser.add_argument('--symbols', nargs='+', help='Символы для тестирования (по умолчанию все)')
    parser.add_argument('--days', type=int, default=30, help='Количество дней для бэктеста')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Начальный баланс USDT')
    parser.add_argument('--leverage', type=int, default=10, help='Кредитное плечо')
    parser.add_argument('--max-positions', type=int, default=5, help='Максимум одновременных позиций')
    parser.add_argument('--output', type=str, help='Файл для сохранения результатов')
    args = parser.parse_args()
    
    print("=" * 60)
    print("BACKTEST ENGINE")
    print("=" * 60)
    
    print(f"\nParameters:")
    print(f"  Symbols: {args.symbols if args.symbols else 'all'}")
    print(f"  Period: {args.days} days")
    print(f"  Initial balance: {args.initial_balance} USDT")
    print(f"  Leverage: {args.leverage}x")
    print(f"  Max positions: {args.max_positions}")
    
    # Загружаем исторические данные
    historical_data = load_historical_data(symbols=args.symbols)
    
    if not historical_data:
        print("\n[ERROR] No historical data!")
        print("Run first: python scripts/ai/collect_historical_data.py --all")
        return 1
    
    # Вычисляем период бэктеста
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Создаем backtesting engine
    backtest = BacktestEngine(config={
        'initial_balance': args.initial_balance,
        'leverage': args.leverage,
        'max_positions': args.max_positions
    })
    
    # Запускаем бэктест
    print("\n" + "=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()
    
    results = backtest.run_backtest(
        historical_data=historical_data,
        start_date=start_date,
        end_date=end_date
    )
    
    # Выводим результаты
    if results.get('success'):
        report = backtest.generate_report(results)
        print("\n" + report)
        
        # Сохраняем результаты
        if args.output:
            backtest.save_results(results, args.output)
        else:
            # Автоматическое имя файла
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_{timestamp}.json"
            backtest.save_results(results, filename)
        
        return 0
    else:
        print("\n[ERROR] Backtest failed!")
        print(f"Error: {results.get('error', 'Unknown')}")
        return 1


if __name__ == "__main__":
    exit(main())

