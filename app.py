import base64
from io import BytesIO
from flask import Flask, render_template, jsonify, request
import threading
import time
from datetime import datetime
import os
import webbrowser
from threading import Timer

# Проверка наличия конфигурации
if not os.path.exists('app/config.py'):
    print("\n" + "="*80)
    print("❌ ОШИБКА: Файл конфигурации не найден!")
    print("="*80)
    print()
    print("📝 Для первого запуска выполните:")
    print()
    print("   1. Скопируйте файл конфигурации:")
    if os.name == 'nt':  # Windows
        print("      copy app\\config.example.py app\\config.py")
    else:  # Linux/Mac
        print("      cp app/config.example.py app/config.py")
    print()
    print("   2. Отредактируйте app/config.py:")
    print("      - Добавьте свои API ключи бирж")
    print("      - Настройте Telegram (опционально)")
    print()
    print("   3. Запустите снова:")
    print("      python app.py")
    print()
    print("="*80)
    print()
    import sys
    sys.exit(1)

from app.config import *

import sys
from app.telegram_notifier import TelegramNotifier
from exchanges.exchange_factory import ExchangeFactory
import json
import requests
from threading import Lock
from app.language import get_current_language, save_language
import concurrent.futures
from functools import partial
# from bot_engine.bot_manager import BotManager  # Убираем - теперь в отдельном сервисе

# Добавим константы
BOTS_SERVICE_URL = 'http://127.0.0.1:5001'
class DEFAULTS:
    PNL_THRESHOLD = 100

# Глобальные переменные для хранения данных
positions_data = {
    'high_profitable': [],
    'profitable': [],
    'losing': [],
    'last_update': None,
    'closed_pnl': [],
    'total_trades': 0,
    'rapid_growth': [],
    'stats': {
        'total_pnl': 0,
        'total_profit': 0,
        'total_loss': 0,
        'high_profitable_count': 0,
        'profitable_count': 0,
        'losing_count': 0,
        'top_profitable': [],
        'top_losing': []
    }
}

# Глобальные переменные для максимальных значений
max_profit_values = {}
max_loss_values = {}

app = Flask(__name__, static_folder='static')
app.config['DEBUG'] = APP_DEBUG
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

telegram = TelegramNotifier()

# Создаем директорию для логов, если её нет
if not os.path.exists('logs'):
    os.makedirs('logs')

def log_to_file(filename, data):
    """Записывает данные в файл с временной меткой"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(f'logs/{filename}', 'a', encoding='utf-8') as f:
        f.write(f"\n=== {timestamp} ===\n")
        f.write(data)
        f.write("\n")

def format_positions(positions):
    """Форматирует позиции для записи в лог"""
    if not positions:
        return "No positions"
    
    result = []
    for pos in positions:
        result.append(
            f"Symbol: {pos['symbol']}\n"
            f"PnL: {pos['pnl']:.2f} USDT\n"
            f"ROI: {pos['roi']:.2f}%\n"
        )
    return "\n".join(result)

def format_stats(stats):
    """Форматирует стаику для записи в лог"""
    return (
        f"Total PnL: {stats['total_pnl']:.2f} USDT\n"
        f"Total profit: {stats['total_profit']:.2f} USDT\n"
        f"Total loss: {stats['total_loss']:.2f} USDT\n"
        f"Number of high-profitable positions: {stats['high_profitable_count']}\n"
        f"Number of profitable positions: {stats['profitable_count']}\n"
        f"Number of losing positions: {stats['losing_count']}\n"
        f"\nTOP-3 profitable:\n{format_positions(stats['top_profitable'])}\n"
        f"\nTOP-3 losing:\n{format_positions(stats['top_losing'])}"
    )

stats_lock = Lock()

def background_update():
    global positions_data, last_stats_time
    last_log_minute = -1
    last_stats_time = None
    thread_id = threading.get_ident()
    
    while True:
        try:
            current_time = time.time()
            
            with stats_lock:
                time_since_last = current_time - last_stats_time if last_stats_time else None
                should_send_stats = (
                    TELEGRAM_NOTIFY.get('STATISTICS', False) and 
                    TELEGRAM_NOTIFY.get('STATISTICS_INTERVAL_ENABLED', True) and
                    (last_stats_time is None or 
                     current_time - last_stats_time >= TELEGRAM_NOTIFY['STATISTICS_INTERVAL'])
                )
                
                # Логирование для диагностики (только важные моменты)
                if should_send_stats:
                    if last_stats_time is None:
                        print(f"[Thread {thread_id}] Первый запуск - отправляем статистику")
                    else:
                        minutes_passed = time_since_last / 60
                        print(f"[Thread {thread_id}] Прошло {minutes_passed:.1f} минут - отправляем статистику")

            positions, rapid_growth = current_exchange.get_positions()
            if not positions:
                time.sleep(2)
                continue

            # Добавляем проверку каждой позиции для уведомлений
            for position in positions:
                telegram.check_position_notifications(position)

            # Проверяем быстрорастущие позиции
            if rapid_growth:
                telegram.check_rapid_growth(rapid_growth)

            high_profitable = []
            profitable = []
            losing = []
            
            total_profit = 0
            total_loss = 0
            
            # Обновляем общее количество сделок
            positions_data['total_trades'] = len(positions)
            positions_data['rapid_growth'] = rapid_growth
            
            # Распределяем позиции по категориям
            for position in positions:
                pnl = position['pnl']
                if pnl > 0:
                    if pnl >= 100:
                        high_profitable.append(position)
                    else:
                        profitable.append(position)
                    total_profit += pnl
                elif pnl < 0:
                    losing.append(position)
                    total_loss += pnl
            
            # Сортировка позиций
            high_profitable.sort(key=lambda x: x['pnl'], reverse=True)
            profitable.sort(key=lambda x: x['pnl'], reverse=True)
            losing.sort(key=lambda x: x['pnl'])
            
            # Получаем TOP-3
            all_profitable = high_profitable + profitable
            all_profitable.sort(key=lambda x: x['pnl'], reverse=True)
            top_profitable = all_profitable[:3] if all_profitable else []
            top_losing = losing[:3] if losing else []
            
            # Обновляем positions_data
            stats = {
                'total_pnl': total_profit + total_loss,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'high_profitable_count': len(high_profitable),
                'profitable_count': len(high_profitable) + len(profitable),
                'losing_count': len(losing),
                'top_profitable': top_profitable,
                'top_losing': top_losing,
                'total_trades': len(positions)
            }
            
            positions_data.update({
                'high_profitable': high_profitable,
                'profitable': profitable,
                'losing': losing,
                'stats': stats,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            # Отправка статистики в Telegram только если нужно
            if should_send_stats:
                try:
                    with stats_lock:
                        print(f"[Thread {thread_id}] Acquired stats_lock for sending")
                        print(f"[Thread {thread_id}] Sending statistics...")
                        telegram.send_statistics(positions_data['stats'])
                        last_stats_time = current_time
                        print(f"[Thread {thread_id}] Stats sent at {datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')}")
                        print(f"[Thread {thread_id}] Released stats_lock after sending")
                except Exception as e:
                    print(f"[Thread {thread_id}] Error sending statistics: {e}")

            # Логируем только при изменении количества позиций или отправке статистики
            current_positions_count = positions_data['total_trades']
            if should_send_stats or current_positions_count != getattr(background_update, 'last_positions_count', -1):
                profitable_count = len([p for p in positions if p['pnl'] > 0])
                losing_count = len([p for p in positions if p['pnl'] < 0])
                print(f"[Thread {thread_id}] Updated positions: {current_positions_count} (прибыльные: {profitable_count}, убыточные: {losing_count})")
                background_update.last_positions_count = current_positions_count
            time.sleep(2)
            
        except Exception as e:
            print(f"Error in background_update: {str(e)}")
            telegram.send_error(str(e))
            time.sleep(5)

# Флаг для отслеживания первого апуска
FIRST_RUN = True

def open_browser():
    """Открывает браузер только при первом запуске"""
    global FIRST_RUN
    if FIRST_RUN and not os.environ.get('WERKZEUG_RUN_MAIN'):
        webbrowser.open(f'http://localhost:{APP_PORT}')
        FIRST_RUN = False

@app.route('/')
def index():
    return render_template('index.html', get_current_language=get_current_language)

@app.route('/bots')
def bots_page():
    """Страница управления ботами"""
    return render_template('index.html', get_current_language=get_current_language)

def analyze_symbol(symbol, force_update=False):
    """Анализирует отдельный символ"""
    clean_symbol = symbol.replace('USDT', '')
    analysis = determine_trend_and_position(clean_symbol, force_update)
    if analysis:
        return {
            'symbol': symbol,
            'trend_analysis': analysis
        }
    return None

def analyze_positions_parallel(positions, max_workers=10):
    """Параллельный анализ позиций"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        analyzed = list(filter(None, executor.map(
            lambda p: analyze_symbol(p['symbol']),
            positions
        )))
        
        for position, analysis in zip(positions, analyzed):
            if analysis and analysis['symbol'] == position['symbol']:
                position['trend_analysis'] = analysis['trend_analysis']
                
        return [p for p in positions if 'trend_analysis' in p]

def analyze_pairs_parallel(pairs, max_workers=10):
    """Параллельный анализ пар"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(filter(None, executor.map(analyze_symbol, pairs)))

@app.route('/get_positions')
def get_positions():
    pnl_threshold = float(request.args.get('pnl_threshold', 100))
    
    all_available_pairs = []  # Больше не используется
    
    if not positions_data['high_profitable'] and not positions_data['profitable'] and not positions_data['losing']:
        # Получаем данные аккаунта даже если нет позиций
        try:
            wallet_data = current_exchange.get_wallet_balance()
        except Exception as e:
            print(f"[API] Error getting wallet data: {str(e)}")
            wallet_data = {
                'total_balance': 0,
                'available_balance': 0,
                'realized_pnl': 0
            }
        
        return jsonify({
            'high_profitable': [],
            'profitable': [],
            'losing': [],
            'stats': {
                'total_pnl': 0,
                'total_profit': 0,
                'total_loss': 0,
                'high_profitable_count': 0,
                'profitable_count': 0,
                'losing_count': 0,
                'top_profitable': [],
                'top_losing': [],
                'total_trades': 0
            },
            'rapid_growth': [],
            'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
            'growth_multiplier': GROWTH_MULTIPLIER,
            'all_pairs': [],
            'wallet_data': {
                'total_balance': wallet_data['total_balance'],
                'available_balance': wallet_data['available_balance'],
                'realized_pnl': wallet_data['realized_pnl']
            }
        })

    # Получаем все позиции
    all_positions = (positions_data['high_profitable'] + 
                    positions_data['profitable'] + 
                    positions_data['losing'])
    
    # Создаем множество символов из активных позиций
    active_position_symbols = set(position['symbol'] for position in all_positions)
    
    # Фильтруем доступные пары
    available_pairs = [pair for pair in all_available_pairs if pair not in active_position_symbols]
    
    # Распределяем позиции по категориям
    high_profitable = []
    profitable = []
    losing = []
    total_profit = 0
    total_loss = 0
    
    for position in all_positions:
        pnl = position['pnl']
        if pnl > 0:
            if pnl >= pnl_threshold:
                high_profitable.append(position)
            else:
                profitable.append(position)
            total_profit += pnl
        elif pnl < 0:
            losing.append(position)
            total_loss += pnl

    # Сортируем позиции
    high_profitable.sort(key=lambda x: x['pnl'], reverse=True)
    profitable.sort(key=lambda x: x['pnl'], reverse=True)
    losing.sort(key=lambda x: x['pnl'])
    
    # Получаем TOP-3
    all_profitable = high_profitable + profitable
    all_profitable.sort(key=lambda x: x['pnl'], reverse=True)
    top_profitable = all_profitable[:3] if all_profitable else []
    top_losing = losing[:3] if losing else []

    # Формируем статистику
    stats = {
        'total_pnl': total_profit + total_loss,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'high_profitable_count': len(high_profitable),
        'profitable_count': len(profitable),
        'losing_count': len(losing),
        'top_profitable': top_profitable,
        'top_losing': top_losing,
        'total_trades': len(all_positions)
    }
    
    # Получаем данные аккаунта
    try:
        wallet_data = current_exchange.get_wallet_balance()
    except Exception as e:
        print(f"[API] Error getting wallet data: {str(e)}")
        wallet_data = {
            'total_balance': 0,
            'available_balance': 0,
            'realized_pnl': 0
        }
    
    return jsonify({
        'high_profitable': high_profitable,
        'profitable': profitable,
        'losing': losing,
        'stats': stats,
        'rapid_growth': positions_data['rapid_growth'],
        'last_update': positions_data['last_update'],
        'growth_multiplier': GROWTH_MULTIPLIER,
        'all_pairs': available_pairs,
        'wallet_data': {
            'total_balance': wallet_data['total_balance'],
            'available_balance': wallet_data['available_balance'],
            'realized_pnl': wallet_data['realized_pnl']
        }
    })

@app.route('/api/positions')
def api_positions():
    """API endpoint for positions - redirects to get_positions"""
    return get_positions()

@app.route('/api/balance')
def get_balance():
    """Получение баланса"""
    try:
        if not current_exchange:
            return jsonify({'error': 'Exchange not initialized'}), 500
        
        wallet_data = current_exchange.get_wallet_balance()
        return jsonify({
            'success': True,
            'balance': wallet_data['total_balance'],
            'available_balance': wallet_data['available_balance'],
            'realized_pnl': wallet_data['realized_pnl']
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/closed_pnl')
def get_closed_pnl():
    """Получение закрытых позиций"""
    try:
        sort_by = request.args.get('sort', 'time')
        print(f"[API] Getting closed PNL, sort by: {sort_by}")
        
        # Получаем баланс и PNL
        wallet_data = current_exchange.get_wallet_balance()
        
        # Получаем закрытые позиции
        closed_pnl = current_exchange.get_closed_pnl(sort_by)
        print(f"[API] Found {len(closed_pnl)} closed positions")
        
        return jsonify({
            'success': True,
            'closed_pnl': closed_pnl,
            'wallet_data': {
                'total_balance': wallet_data['total_balance'],
                'available_balance': wallet_data['available_balance'],
                'realized_pnl': wallet_data['realized_pnl']
            }
        })
    except Exception as e:
        print(f"[API] Error getting closed PNL: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500





def calculate_statistics(positions):
    """Calculates statistics for positions"""
    total_profit = 0
    total_loss = 0
    high_profitable = []
    profitable = []
    losing = []

    for position in positions:
        pnl = position['pnl']
        if pnl > 0:
            if pnl >= 100:
                high_profitable.append(position)
            else:
                profitable.append(position)
            total_profit += pnl
        else:
            losing.append(position)
            total_loss += pnl

    return {
        'total_pnl': total_profit + total_loss,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'total_trades': len(positions),
        'profitable_count': len(high_profitable) + len(profitable),
        'losing_count': len(losing),
        'top_profitable': sorted(high_profitable + profitable, key=lambda x: x['pnl'], reverse=True)[:3],
        'top_losing': sorted(losing, key=lambda x: x['pnl'])[:3]
    }

def send_daily_report():
    """Отправка ежедневного отчета"""
    while True:
        now = datetime.now()
        if now.strftime('%H:%M') == TELEGRAM_NOTIFY['DAILY_REPORT_TIME']:
            positions, _ = exchange.get_positions()
            if positions:
                stats = calculate_statistics(positions)
                telegram.send_daily_report(stats)
        time.sleep(60)  # Проверяем каждую минуту

# Глобальная переменная для хранения текущей биржи
current_exchange = None

def init_exchange():
    """Инициализация биржи"""
    try:
        print(f"[INIT] Получение конфигурации для {ACTIVE_EXCHANGE}...")
        exchange_config = EXCHANGES[ACTIVE_EXCHANGE]
        print(f"[INIT] Конфигурация получена: {exchange_config}")
        
        print(f"[INIT] Создание экземпляра биржи {ACTIVE_EXCHANGE}...")
        exchange = ExchangeFactory.create_exchange(
            ACTIVE_EXCHANGE,
            exchange_config['api_key'],
            exchange_config['api_secret'],
            exchange_config.get('passphrase')  # Добавляем passphrase для OKX
        )
        
        print(f"[INIT] ✅ Биржа {ACTIVE_EXCHANGE} успешно создана")
        return exchange
    except Exception as e:
        print(f"[INIT] ❌ Ошибка создания биржи {ACTIVE_EXCHANGE}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/api/exchanges', methods=['GET'])
def get_exchanges():
    """Получение списка доступных бирж"""
    exchanges = [{
        'name': name,
        'enabled': config['enabled'],
        'active': name == ACTIVE_EXCHANGE
    } for name, config in EXCHANGES.items()]
    return jsonify({'exchanges': exchanges})

@app.route('/api/exchange', methods=['POST'])
def switch_exchange():
    """Переключение активной биржи"""
    global current_exchange
    try:
        data = request.get_json()
        exchange_name = data.get('exchange')
        
        if exchange_name not in EXCHANGES:
            return jsonify({'error': 'Exchange not found'}), 404
            
        if not EXCHANGES[exchange_name]['enabled']:
            return jsonify({'error': 'Exchange is disabled'}), 400
        
        try:
            # Создаем новый экземпляр биржи для проверки подключения
            exchange_config = EXCHANGES[exchange_name]
            new_exchange = ExchangeFactory.create_exchange(
                exchange_name,
                exchange_config['api_key'],
                exchange_config['api_secret'],
                exchange_config.get('passphrase')  # Добавляем passphrase для OKX
            )
            
            # Пробуем получить позиции для проверки работоспособности
            positions, _ = new_exchange.get_positions()
            
            # Если все хорошо, обновляем конфигурацию
            with open('app/config.py', 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Обновляем активную биржу в конфиге
            new_config = config_content.replace(
                f"ACTIVE_EXCHANGE = '{ACTIVE_EXCHANGE}'",
                f"ACTIVE_EXCHANGE = '{exchange_name}'"
            )
            
            with open('app/config.py', 'w', encoding='utf-8') as f:
                f.write(new_config)
            
            # Обновляем текущую биржу
            current_exchange = new_exchange
            
            return jsonify({
                'success': True,
                'message': f'Switched to {exchange_name}'
            })
            
        except Exception as e:
            print(f"Error testing new exchange connection: {str(e)}")
            return jsonify({
                'error': f'Failed to connect to {exchange_name}: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Error in switch_exchange: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Инициализируем биржу при запуске
print(f"[INIT] Инициализация биржи {ACTIVE_EXCHANGE}...")
current_exchange = init_exchange()
if not current_exchange:
    print("[INIT] ❌ Не удалось инициализировать биржу")
    sys.exit(1)
else:
    print(f"[INIT] ✅ Биржа {ACTIVE_EXCHANGE} успешно инициализирована")

# Убираем инициализацию менеджера ботов - теперь он в отдельном сервисе
# bot_manager = BotManager(exchange)

# Добавляем функцию clean_symbol если она где-то используется
def clean_symbol(symbol):
    """Удаляет 'USDT' из названия символа"""
    return symbol.replace('USDT', '')

@app.route('/api/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    language = data.get('language', 'en')
    print(f"Setting language to: {language}")
    save_language(language)
    telegram.set_language(language)
    return jsonify({'status': 'success', 'language': language})

@app.route('/api/ticker/<symbol>')
def get_ticker(symbol):
    try:
        print(f"[TICKER] Getting ticker for {symbol}...")
        
        # Проверяем инициализацию биржи
        if not exchange:
            print("[TICKER] Exchange not initialized")
            return jsonify({'error': 'Exchange not initialized'}), 500
            
        # Получаем данные тикера
        ticker_data = exchange.get_ticker(symbol)
        print(f"[TICKER] Raw ticker data: {ticker_data}")
        
        if ticker_data:
            print(f"[TICKER] Successfully got ticker for {symbol}: {ticker_data}")
            return jsonify(ticker_data)
            
        print(f"[TICKER] No ticker data available for {symbol}")
        return jsonify({'error': 'No ticker data available'}), 404
        
    except Exception as e:
        print(f"[TICKER] Error getting ticker for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/close_position', methods=['POST'])
def close_position():
    """Закрытие позиции"""
    try:
        data = request.json
        if not data or not all(k in data for k in ['symbol', 'size', 'side']):
            return jsonify({
                'success': False,
                'message': 'Не указаны обязательные параметры (symbol, size, side)'
            }), 400

        print(f"[API] Closing position: {data}")
        result = exchange.close_position(
            symbol=data['symbol'],
            size=float(data['size']),
            side=data['side'],
            order_type=data.get('order_type', 'Limit')  # По умолчанию используем Limit для обратной совместимости
        )
        
        print(f"[API] Close position result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"[API] Error closing position: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Ошибка при закрытии позиции: {str(e)}'
        }), 500

@app.route('/api/get_language')
def get_language():
    """Получение текущего языка"""
    try:
        current_lang = get_current_language()
        return jsonify({
            'success': True,
            'language': current_lang
        })
    except Exception as e:
        print(f"Error getting language: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@app.route('/api/blacklist', methods=['POST'])
def manage_blacklist():
    """Управление черным списком"""
    try:
        data = request.get_json()
        action = data.get('action')
        symbol = data.get('symbol')
        
        if not action or not symbol:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400
            
        blacklist_file = 'data/blacklist.json'
        os.makedirs('data', exist_ok=True)
        
        try:
            with open(blacklist_file, 'r') as f:
                blacklist = json.load(f)
        except:
            blacklist = []
            
        if action == 'add':
            if symbol not in blacklist:
                blacklist.append(symbol)
        elif action == 'remove':
            if symbol in blacklist:
                blacklist.remove(symbol)
                
        with open(blacklist_file, 'w') as f:
            json.dump(blacklist, f)
            
        return jsonify({
            'success': True,
            'blacklist': blacklist
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Добавляем глобальные переменные для кэширования
ticker_analysis_cache = {}
CACHE_TIMEOUT = 300  # 5 минут

def determine_trend_and_position(symbol, force_update=False):
    """Определяет тренд и позицию цены на графике с кэшированием"""
    global ticker_analysis_cache
    current_time = time.time()
    
    # Проверяем кэш, если не требуется принудительное обновление
    if not force_update and symbol in ticker_analysis_cache:
        cached_data = ticker_analysis_cache[symbol]
        if current_time - cached_data['timestamp'] < CACHE_TIMEOUT:
            return cached_data['data']
    
    try:
        # Получаем исторические данные
        data = exchange.get_chart_data(symbol, '1d', '1M')
        if not data.get('success'):
            return None
            
        candles = data.get('data', {}).get('candles', [])
        if not candles:
            return None
            
        # Находим минимальную и максимальную цену
        min_price = min(float(candle['low']) for candle in candles)
        max_price = max(float(candle['high']) for candle in candles)
        current_price = float(candles[-1]['close'])
        
        # Определяем позицию цены в процентах от диапазона
        price_range = max_price - min_price
        if price_range == 0:
            return None
            
        position_percent = ((current_price - min_price) / price_range) * 100
        
        # Определяем тренд
        period = 14  # период для определения тренда
        if len(candles) < period:
            return None
            
        recent_prices = [float(candle['close']) for candle in candles[-period:]]
        first_half = sum(recent_prices[:period//2]) / (period//2)
        second_half = sum(recent_prices[period//2:]) / (period//2)
        
        if second_half > first_half * 1.02:  # 2% разница для определения роста
            trend = 'рост'
        elif first_half > second_half * 1.02:  # 2% разница для определения падения
            trend = 'падение'
        else:
            trend = 'флэт'
            
        # Определяем состояние тикера
        state = None
        if 0 <= position_percent <= 10:
            if trend in ['флэт', 'рост']:
                state = 'дно рынка'
            else:
                state = 'падение'
        elif 10 < position_percent <= 40:
            state = trend
        elif 40 < position_percent <= 60:
            state = trend
        elif 60 < position_percent <= 90:
            if trend == 'флэт':
                state = 'диапазон распродажи'
            elif trend == 'падение':
                state = 'диапазон падения'
            else:
                state = 'рост'
        elif 90 < position_percent <= 100:
            if trend == 'флэт':
                state = 'хай рынка'
            elif trend == 'падение':
                state = 'падение'
            else:
                state = 'диапазон распродажи'
        
        result = {
            'trend': trend,
            'position_percent': position_percent,
            'state': state
        }
        
        # Сохраняем результат в кэш
        ticker_analysis_cache[symbol] = {
            'data': result,
            'timestamp': current_time
        }
        
        return result
        
    except Exception:
        return None

def clear_old_cache():
    """Очищает устаревшие данные из кэша"""
    global ticker_analysis_cache
    current_time = time.time()
    expired_symbols = [
        symbol for symbol, data in ticker_analysis_cache.items()
        if current_time - data['timestamp'] >= CACHE_TIMEOUT
    ]
    for symbol in expired_symbols:
        del ticker_analysis_cache[symbol]

@app.route('/api/ticker_analysis/<symbol>')
def get_ticker_analysis(symbol):
    """Получение анализа тикера (тренд и позиция на графике)"""
    try:
        force_update = request.args.get('force_update', '0') == '1'
        analysis = determine_trend_and_position(symbol, force_update)
        if analysis:
            return jsonify({
                'success': True,
                'data': analysis,
                'cached': not force_update and symbol in ticker_analysis_cache
            })
        return jsonify({
            'success': False,
            'error': 'Could not analyze ticker'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def background_cache_cleanup():
    """Фоновая очистка кэша"""
    while True:
        try:
            clear_old_cache()
        except Exception as e:
            print(f"Error in cache cleanup: {str(e)}")
        time.sleep(60)  # Проверяем каждую минуту

# Кэш для хранения данных свечей
candles_cache = {}
CACHE_TIMEOUT = 300  # 5 минут

@app.route('/api/candles/<symbol>')
def get_candles(symbol):
    """Получение свечей для расчета тренда на клиенте"""
    current_time = time.time()
    
    # Проверяем кэш
    if symbol in candles_cache:
        cached_data = candles_cache[symbol]
        if current_time - cached_data['timestamp'] < CACHE_TIMEOUT:
            return jsonify(cached_data['data'])
    
    try:
        # Получаем данные за последний месяц
        data = exchange.get_chart_data(symbol, '1d', '1M')
        if not data.get('success'):
            return jsonify({'success': False, 'error': 'Не удалось получить данные'})
        
        # Сохраняем в кэш
        candles_cache[symbol] = {
            'data': data,
            'timestamp': current_time
        }
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def clear_old_cache():
    """Очистка устаревших данных из кэша"""
    current_time = time.time()
    expired_symbols = [
        symbol for symbol, data in candles_cache.items()
        if current_time - data['timestamp'] >= CACHE_TIMEOUT
    ]
    for symbol in expired_symbols:
        del candles_cache[symbol]

def background_cache_cleanup():
    """Фоновая очистка кэша"""
    while True:
        try:
            clear_old_cache()
        except Exception:
            pass
        time.sleep(60)  # Проверяем каждую минуту

# Прокси для API endpoints ботов (перенаправляем на внешний сервис)
def call_bots_service(endpoint, method='GET', data=None, timeout=10):
    """Универсальная функция для вызова API сервиса ботов"""
    try:
        url = f"{BOTS_SERVICE_URL}{endpoint}"
        
        if method == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return {'success': False, 'error': f'Unsupported method: {method}'}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                'success': False, 
                'error': f'Bots service returned status {response.status_code}',
                'details': response.text
            }
            
    except requests.exceptions.ConnectionError:
        return {
            'success': False, 
            'error': 'Сервис ботов недоступен. Запустите в отдельном терминале: python bots.py',
            'service_url': BOTS_SERVICE_URL,
            'instructions': 'Откройте новый терминал и выполните: python bots.py'
        }
    except requests.exceptions.Timeout:
        return {'success': False, 'error': f'Timeout calling bots service ({timeout}s)'}
    except Exception as e:
        return {'success': False, 'error': f'Error calling bots service: {str(e)}'}



@app.route('/api/bots/list', methods=['GET'])
def get_bots_list():
    """Получение списка всех ботов (прокси к сервису ботов)"""
    result = call_bots_service('/api/bots/list')
    status_code = 200 if result.get('success') else 500
    return jsonify(result), status_code

@app.route('/api/bots/control', methods=['POST'])
def control_bot():
    """Управление ботом (прокси к сервису ботов)"""
    data = request.get_json()
    result = call_bots_service('/api/bots/control', method='POST', data=data)
    status_code = 200 if result.get('success') else 500
    return jsonify(result), status_code

@app.route('/api/bots/config', methods=['GET', 'POST'])
def bots_config():
    """Получение и обновление конфигурации Auto Bot (прокси к сервису ботов)"""
    if request.method == 'GET':
        result = call_bots_service('/api/bots/config')
    else:
        data = request.get_json()
        result = call_bots_service('/api/bots/config', method='POST', data=data)
    
    status_code = 200 if result.get('success') else 500
    return jsonify(result), status_code

@app.route('/api/bots/status', methods=['GET'])
def get_bots_status():
    """Получение общего статуса ботов (прокси к сервису ботов)"""
    result = call_bots_service('/api/bots/status')
    status_code = 200 if result.get('success') else 500
    return jsonify(result), status_code

@app.route('/api/bots/pairs', methods=['GET'])
def get_bots_pairs():
    """Получение списка доступных торговых пар (прокси к сервису ботов)"""
    result = call_bots_service('/api/bots/pairs')
    status_code = 200 if result.get('success') else 500
    return jsonify(result), status_code

@app.route('/api/bots/ensure/<symbol>', methods=['POST'])
def ensure_bot_exists(symbol):
    """Создание бота для символа (прокси к сервису ботов)"""
    result = call_bots_service(f'/api/bots/ensure/{symbol}', method='POST')
    status_code = 200 if result.get('success') else 500
    return jsonify(result), status_code

@app.route('/api/bots/health', methods=['GET'])
def get_bots_health():
    """Проверка состояния сервиса ботов"""
    result = call_bots_service('/health', timeout=5)
    status_code = 200 if result.get('status') == 'ok' else 503
    return jsonify(result), status_code

@app.route('/api/bots/status/<symbol>', methods=['GET'])
def get_bot_status(symbol):
    """Получить статус конкретного бота (прокси к сервису ботов)"""
    result = call_bots_service(f'/api/bots/status/{symbol}')
    status_code = 200 if result.get('success') else 500
    return jsonify(result), status_code

@app.route('/api/bots/create', methods=['POST'])
def create_bot():
    """Создать бота (прокси к сервису ботов)"""
    data = request.get_json()
    result = call_bots_service('/api/bots/create', method='POST', data=data)
    status_code = 200 if result.get('success') else 500
    return jsonify(result), status_code

@app.route('/api/bots/auto-bot', methods=['POST'])
def toggle_auto_bot():
    """Переключить Auto Bot (прокси к сервису ботов)"""
    data = request.get_json()
    result = call_bots_service('/api/bots/auto-bot', method='POST', data=data)
    status_code = 200 if result.get('success') else 500
    return jsonify(result), status_code

@app.route('/get_symbol_chart/<symbol>')
def get_symbol_chart(symbol):
    """Получение миниграфика для символа"""
    try:
        theme = request.args.get('theme', 'dark')
        print(f"[CHART] Getting chart for {symbol} with theme {theme}")
        
        # Проверяем инициализацию биржи
        if not current_exchange:
            print("[CHART] Exchange not initialized")
            return jsonify({'error': 'Exchange not initialized'}), 500
            
        # Получаем данные свечей для миниграфика
        data = current_exchange.get_chart_data(symbol, '1h', '24h')  # 1 час свечи за 24 часа
        if not data.get('success'):
            print(f"[CHART] Failed to get chart data for {symbol}")
            return jsonify({'error': 'Failed to get chart data'}), 500
            
        candles = data.get('data', {}).get('candles', [])
        if not candles:
            print(f"[CHART] No candles data for {symbol}")
            return jsonify({'error': 'No chart data available'}), 404
            
        # Создаем простой миниграфик
        import matplotlib
        matplotlib.use('Agg')  # Используем неинтерактивный бэкенд
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        import io
        import base64
        
        # Настраиваем стиль в зависимости от темы
        if theme == 'light':
            plt.style.use('default')
            bg_color = 'white'
            line_color = '#1f77b4'
        else:
            plt.style.use('dark_background')
            bg_color = '#2d2d2d'
            line_color = '#00ff00'
            
        # Подготавливаем данные
        times = [datetime.fromtimestamp(int(candle['timestamp']) / 1000) for candle in candles]
        prices = [float(candle['close']) for candle in candles]
        
        # Создаем график
        fig, ax = plt.subplots(figsize=(4, 2), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        # Рисуем линию цены
        ax.plot(times, prices, color=line_color, linewidth=1.5)
        
        # Настраиваем внешний вид
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Убираем отступы
        plt.tight_layout(pad=0)
        
        # Конвертируем в base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                   facecolor=bg_color, edgecolor='none')
        buffer.seek(0)
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        print(f"[CHART] Successfully generated chart for {symbol}")
        return jsonify({
            'success': True,
            'chart': chart_data
        })
        
    except Exception as e:
        print(f"[CHART] Error generating chart for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_sma200_position/<symbol>')
def get_sma200_position(symbol):
    """Получение позиции цены относительно SMA200"""
    try:
        print(f"[SMA200] Getting SMA200 position for {symbol}")
        
        # Проверяем инициализацию биржи
        if not current_exchange:
            print("[SMA200] Exchange not initialized")
            return jsonify({'error': 'Exchange not initialized'}), 500
            
        # Получаем исторические данные для расчета SMA200
        data = current_exchange.get_chart_data(symbol, '1d', '200d')  # Дневные свечи за 200 дней
        if not data.get('success'):
            print(f"[SMA200] Failed to get chart data for {symbol}")
            return jsonify({'error': 'Failed to get chart data'}), 500
            
        candles = data.get('data', {}).get('candles', [])
        if len(candles) < 200:
            print(f"[SMA200] Not enough data for SMA200 calculation for {symbol}")
            return jsonify({'error': 'Not enough data for SMA200'}), 400
            
        # Рассчитываем SMA200
        closes = [float(candle['close']) for candle in candles[-200:]]
        sma200 = sum(closes) / 200
        
        # Получаем текущую цену
        current_price = closes[-1]
        
        # Определяем позицию относительно SMA200
        above_sma200 = current_price > sma200
        
        print(f"[SMA200] {symbol}: Current={current_price:.4f}, SMA200={sma200:.4f}, Above={above_sma200}")
        
        return jsonify({
            'success': True,
            'above_sma200': above_sma200,
            'current_price': current_price,
            'sma200': sma200,
            'difference_percent': ((current_price - sma200) / sma200) * 100
        })
        
    except Exception as e:
        print(f"[SMA200] Error calculating SMA200 for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Создаем директорию для логов
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Очищаем старые логи при запуске
    log_files = ['logs/bots.log', 'logs/app.log', 'logs/error.log']
    for log_file in log_files:
        if os.path.exists(log_file):
            file_size = os.path.getsize(log_file)
            if file_size > 2 * 1024 * 1024:  # 2MB
                print(f"[APP] 🗑️ Очищаем большой лог файл: {log_file} ({file_size / 1024 / 1024:.1f}MB)")
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Лог файл очищен при запуске - {datetime.now().isoformat()}\n")
            else:
                print(f"[APP] 📝 Лог файл в порядке: {log_file} ({file_size / 1024:.1f}KB)")
    
    # Открываем браузер с задержкой
    Timer(1.5, open_browser).start()
    
    # Запускаем фоновые процессы (теперь всегда, так как reloader отключен)
    update_thread = threading.Thread(target=background_update)
    update_thread.daemon = True
    update_thread.start()
    
    # Запускаем поток для отправки дневного отчета
    if TELEGRAM_NOTIFY['DAILY_REPORT']:
        daily_report_thread = threading.Thread(target=send_daily_report)
        daily_report_thread.daemon = True
        daily_report_thread.start()
        
    # Запускаем поток очистки кэша
    cache_cleanup_thread = threading.Thread(target=background_cache_cleanup)
    cache_cleanup_thread.daemon = True
    cache_cleanup_thread.start()
    
    # Запускаем Flask-сервер (отключаем reloader для стабильности Telegram уведомлений)
    app.run(debug=False, host=APP_HOST, port=APP_PORT, use_reloader=False) 