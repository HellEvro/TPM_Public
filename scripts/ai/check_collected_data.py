"""
Проверка собранных исторических данных
"""

import sys
sys.path.append('.')

import pandas as pd
from pathlib import Path
from datetime import datetime

def check_collected_data():
    """Проверяет собранные данные"""
    
    data_dir = Path('data/ai/historical')
    
    if not data_dir.exists():
        print("Directory not found: data/ai/historical")
        return
    
    csv_files = list(data_dir.glob('*_6h_historical.csv'))
    
    if not csv_files:
        print("No CSV files found")
        return
    
    print(f"Found {len(csv_files)} files")
    print()
    
    total_candles = 0
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        symbol = csv_file.stem.replace('_6h_historical', '')
        
        # Проверяем наличие поля timestamp или time
        timestamp_field = 'timestamp' if 'timestamp' in df.columns else 'time'
        
        if timestamp_field not in df.columns:
            print(f"{symbol}: ERROR - no timestamp field")
            continue
        
        # Информация о данных
        first_ts = df.iloc[0][timestamp_field]
        last_ts = df.iloc[-1][timestamp_field]
        
        first_dt = datetime.fromtimestamp(first_ts / 1000)
        last_dt = datetime.fromtimestamp(last_ts / 1000)
        
        days = (last_dt - first_dt).days
        
        print(f"{symbol}:")
        print(f"  Candles: {len(df)}")
        print(f"  Period: {first_dt.strftime('%Y-%m-%d')} - {last_dt.strftime('%Y-%m-%d')}")
        print(f"  Days: {days}")
        print(f"  Columns: {list(df.columns)}")
        print()
        
        total_candles += len(df)
    
    print(f"TOTAL: {total_candles:,} candles from {len(csv_files)} coins")

if __name__ == '__main__':
    check_collected_data()

