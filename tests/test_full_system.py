"""
Комплексное тестирование всей системы InfoBot
Проверяет все аспекты: фильтры, конфиг, боты, историю
"""

import sys
import io
import json
import time
import requests
from datetime import datetime
from colorama import init, Fore, Style

# Исправление для Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Инициализация colorama
init(autoreset=True)

# URL сервисов
BOTS_SERVICE_URL = 'http://127.0.0.1:5001'
APP_SERVICE_URL = 'http://127.0.0.1:5000'

# Счетчики тестов
tests_passed = 0
tests_failed = 0
tests_total = 0

def print_header(text):
    """Печатает заголовок теста"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}{text}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

def print_test(name):
    """Печатает название теста"""
    global tests_total
    tests_total += 1
    print(f"\n{Fore.YELLOW}[ТЕСТ {tests_total}] {name}{Style.RESET_ALL}")

def print_success(message):
    """Печатает успешный результат"""
    global tests_passed
    tests_passed += 1
    print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")

def print_error(message):
    """Печатает ошибку"""
    global tests_failed
    tests_failed += 1
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")

def print_info(message):
    """Печатает информацию"""
    print(f"{Fore.BLUE}ℹ️  {message}{Style.RESET_ALL}")

def print_warning(message):
    """Печатает предупреждение"""
    print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")

# ==========================================
# ТЕСТЫ СЕРВИСОВ
# ==========================================

def test_services_online():
    """Проверка что оба сервиса запущены"""
    print_header("ПРОВЕРКА СЕРВИСОВ")
    
    # Тест 1: Bots Service
    print_test("Bots Service (порт 5001) онлайн")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/status", timeout=5)
        if response.status_code == 200:
            print_success("Bots Service онлайн")
        else:
            print_error(f"Bots Service вернул код {response.status_code}")
    except Exception as e:
        print_error(f"Bots Service недоступен: {e}")
    
    # Тест 2: App Service
    print_test("App Service (порт 5000) онлайн")
    try:
        response = requests.get(f"{APP_SERVICE_URL}/", timeout=5)
        if response.status_code == 200:
            print_success("App Service онлайн")
        else:
            print_error(f"App Service вернул код {response.status_code}")
    except Exception as e:
        print_error(f"App Service недоступен: {e}")

# ==========================================
# ТЕСТЫ КОНФИГУРАЦИИ
# ==========================================

def test_configuration():
    """Проверка конфигурации Auto Bot"""
    print_header("ПРОВЕРКА КОНФИГУРАЦИИ")
    
    # Тест 3: Получение конфигурации
    print_test("Получение конфигурации Auto Bot")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/auto-bot", timeout=5)
        data = response.json()
        
        if data.get('success'):
            config = data.get('config', {})
            print_success("Конфигурация получена")
            print_info(f"Enabled: {config.get('enabled')}")
            print_info(f"Max concurrent: {config.get('max_concurrent')}")
            print_info(f"RSI LONG threshold: {config.get('rsi_long_threshold')}")
            print_info(f"RSI SHORT threshold: {config.get('rsi_short_threshold')}")
            print_info(f"RSI time filter enabled: {config.get('rsi_time_filter_enabled')}")
            print_info(f"RSI time filter candles: {config.get('rsi_time_filter_candles')}")
            
            # Проверяем критические параметры
            if config.get('enabled') == False:
                print_success("Auto Bot выключен (безопасно)")
            else:
                print_warning("Auto Bot включен!")
            
            if config.get('enable_maturity_check') == True:
                print_success("Проверка зрелости включена")
            else:
                print_error("Проверка зрелости ВЫКЛЮЧЕНА!")
            
            if config.get('rsi_time_filter_enabled') == True:
                print_success(f"RSI временной фильтр включен ({config.get('rsi_time_filter_candles')} свечей)")
            else:
                print_warning("RSI временной фильтр выключен")
                
        else:
            print_error(f"Ошибка получения конфигурации: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при получении конфигурации: {e}")
    
    # Тест 4: Системная конфигурация
    print_test("Получение системной конфигурации")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/system-config", timeout=5)
        data = response.json()
        
        if data.get('success'):
            config = data.get('config', {})
            print_success("Системная конфигурация получена")
            print_info(f"RSI update interval: {config.get('rsi_update_interval')} сек")
            print_info(f"Auto save interval: {config.get('auto_save_interval')} сек")
            print_info(f"Refresh interval: {config.get('refresh_interval')} сек")
        else:
            print_error(f"Ошибка получения системной конфигурации: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при получении системной конфигурации: {e}")

# ==========================================
# ТЕСТЫ RSI ДАННЫХ И ФИЛЬТРОВ
# ==========================================

def test_rsi_data_and_filters():
    """Проверка RSI данных и фильтров монет"""
    print_header("ПРОВЕРКА RSI ДАННЫХ И ФИЛЬТРОВ")
    
    # Тест 5: Получение монет с RSI
    print_test("Получение списка монет с RSI данными")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/coins-with-rsi", timeout=10)
        data = response.json()
        
        if data.get('success'):
            coins = data.get('coins', {})
            manual_positions = data.get('manual_positions', [])
            
            print_success(f"Получено {len(coins)} монет с RSI данными")
            print_info(f"Ручных позиций: {len(manual_positions)}")
            
            # Анализируем сигналы
            enter_long = sum(1 for coin in coins.values() if coin.get('effective_signal') == 'ENTER_LONG')
            enter_short = sum(1 for coin in coins.values() if coin.get('effective_signal') == 'ENTER_SHORT')
            wait = sum(1 for coin in coins.values() if coin.get('effective_signal') == 'WAIT')
            
            print_info(f"Сигналы: ENTER_LONG={enter_long}, ENTER_SHORT={enter_short}, WAIT={wait}")
            
            # Проверяем наличие обязательных полей
            if len(coins) > 0:
                sample_coin = list(coins.values())[0]
                required_fields = ['symbol', 'rsi6h', 'trend6h', 'signal', 'effective_signal']
                missing_fields = [field for field in required_fields if field not in sample_coin]
                
                if not missing_fields:
                    print_success("Все обязательные поля присутствуют")
                else:
                    print_error(f"Отсутствуют поля: {missing_fields}")
            
            # Проверяем ручные позиции
            if manual_positions:
                print_success(f"Ручные позиции определены: {manual_positions[:5]}")
            else:
                print_info("Нет ручных позиций")
                
        else:
            print_error(f"Ошибка получения RSI данных: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при получении RSI данных: {e}")
    
    # Тест 6: Зрелые монеты
    print_test("Проверка зрелых монет")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/mature-coins", timeout=5)
        data = response.json()
        
        if data.get('success'):
            mature_coins = data.get('mature_coins', {})
            print_success(f"Получено {len(mature_coins)} зрелых монет")
            
            if len(mature_coins) > 0:
                # Проверяем структуру данных
                sample_coin = list(mature_coins.values())[0]
                if 'last_verified' in sample_coin and 'maturity_checks' in sample_coin:
                    print_success("Структура данных зрелых монет корректна")
                else:
                    print_error("Некорректная структура данных зрелых монет")
        else:
            print_error(f"Ошибка получения зрелых монет: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при получении зрелых монет: {e}")

# ==========================================
# ТЕСТЫ БОТОВ
# ==========================================

def test_bots_management():
    """Проверка управления ботами"""
    print_header("ПРОВЕРКА УПРАВЛЕНИЯ БОТАМИ")
    
    # Тест 7: Список активных ботов
    print_test("Получение списка активных ботов")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/list", timeout=5)
        data = response.json()
        
        if data.get('success'):
            bots = data.get('bots', [])
            print_success(f"Получено {len(bots)} ботов")
            
            if len(bots) > 0:
                # Анализируем статусы
                statuses = {}
                for bot in bots:
                    status = bot.get('status', 'unknown')
                    statuses[status] = statuses.get(status, 0) + 1
                
                print_info(f"Статусы ботов: {statuses}")
                
                # Проверяем структуру данных
                sample_bot = bots[0]
                required_fields = ['symbol', 'status', 'volume_mode', 'volume_value']
                missing_fields = [field for field in required_fields if field not in sample_bot]
                
                if not missing_fields:
                    print_success("Структура данных ботов корректна")
                else:
                    print_error(f"Отсутствуют поля: {missing_fields}")
                
                # Проверяем новые поля отслеживания
                tracking_fields = ['order_id', 'entry_timestamp', 'opened_by_autobot']
                has_tracking = all(field in sample_bot for field in tracking_fields)
                
                if has_tracking:
                    print_success("Поля отслеживания позиций присутствуют")
                else:
                    print_warning("Некоторые поля отслеживания отсутствуют")
            else:
                print_info("Нет активных ботов")
        else:
            print_error(f"Ошибка получения списка ботов: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при получении списка ботов: {e}")
    
    # Тест 8: Состояние процессов
    print_test("Проверка состояния процессов системы")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/process-state", timeout=5)
        data = response.json()
        
        if data.get('success'):
            process_state = data.get('process_state', {})
            system_info = data.get('system_info', {})
            
            print_success("Состояние процессов получено")
            print_info(f"Smart RSI Manager: {'✅ Запущен' if system_info.get('smart_rsi_manager_running') else '❌ Остановлен'}")
            print_info(f"Exchange: {'✅ Инициализирован' if system_info.get('exchange_initialized') else '❌ Не инициализирован'}")
            print_info(f"Всего ботов: {system_info.get('total_bots', 0)}")
            print_info(f"Auto Bot: {'✅ Включен' if system_info.get('auto_bot_enabled') else '❌ Выключен'}")
            print_info(f"Зрелых монет: {system_info.get('mature_coins_storage_size', 0)}")
        else:
            print_error(f"Ошибка получения состояния процессов: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при получении состояния процессов: {e}")

# ==========================================
# ТЕСТЫ ИСТОРИИ
# ==========================================

def test_bot_history():
    """Проверка истории ботов"""
    print_header("ПРОВЕРКА ИСТОРИИ БОТОВ")
    
    # Тест 9: История торговли
    print_test("Получение истории торговли")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/history", timeout=5)
        data = response.json()
        
        if data.get('success'):
            history = data.get('history', {})
            trades = history.get('trades', [])
            statistics = history.get('statistics', {})
            
            print_success(f"История получена: {len(trades)} записей")
            print_info(f"Всего сделок: {statistics.get('total_trades', 0)}")
            print_info(f"Прибыльных: {statistics.get('profitable_trades', 0)}")
            print_info(f"Убыточных: {statistics.get('losing_trades', 0)}")
            print_info(f"Общий PnL: {statistics.get('total_pnl', 0):.2f} USDT")
            
            if len(trades) > 0:
                # Проверяем структуру записей
                sample_trade = trades[0]
                required_fields = ['timestamp', 'type', 'symbol']
                missing_fields = [field for field in required_fields if field not in sample_trade]
                
                if not missing_fields:
                    print_success("Структура данных истории корректна")
                else:
                    print_error(f"Отсутствуют поля: {missing_fields}")
            else:
                print_info("История пуста (нет записей)")
        else:
            print_error(f"Ошибка получения истории: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при получении истории: {e}")

# ==========================================
# ТЕСТЫ ФИЛЬТРОВ
# ==========================================

def test_filters():
    """Проверка фильтров монет"""
    print_header("ПРОВЕРКА ФИЛЬТРОВ МОНЕТ")
    
    # Тест 10: Белый список
    print_test("Получение белого списка")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/whitelist", timeout=5)
        data = response.json()
        
        if data.get('success'):
            whitelist = data.get('whitelist', [])
            print_success(f"Белый список получен: {len(whitelist)} монет")
            if whitelist:
                print_info(f"Монеты: {whitelist[:10]}")
        else:
            print_error(f"Ошибка получения белого списка: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при получении белого списка: {e}")
    
    # Тест 11: Черный список
    print_test("Получение черного списка")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/blacklist", timeout=5)
        data = response.json()
        
        if data.get('success'):
            blacklist = data.get('blacklist', [])
            print_success(f"Черный список получен: {len(blacklist)} монет")
            if blacklist:
                print_info(f"Монеты: {blacklist[:10]}")
        else:
            print_error(f"Ошибка получения черного списка: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при получении черного списка: {e}")

# ==========================================
# ТЕСТЫ ЗАЩИТНЫХ МЕХАНИЗМОВ
# ==========================================

def test_protection_mechanisms():
    """Проверка защитных механизмов"""
    print_header("ПРОВЕРКА ЗАЩИТНЫХ МЕХАНИЗМОВ")
    
    # Тест 12: Проверка что Auto Bot выключен
    print_test("Auto Bot должен быть выключен при запуске")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/auto-bot", timeout=5)
        data = response.json()
        
        if data.get('success'):
            config = data.get('config', {})
            if config.get('enabled') == False:
                print_success("✅ Auto Bot выключен (безопасно)")
            else:
                print_error("❌ Auto Bot ВКЛЮЧЕН! Это небезопасно!")
        else:
            print_error(f"Ошибка проверки Auto Bot: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при проверке Auto Bot: {e}")
    
    # Тест 13: Проверка зрелости монет
    print_test("Проверка фильтра зрелости монет")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/auto-bot", timeout=5)
        data = response.json()
        
        if data.get('success'):
            config = data.get('config', {})
            if config.get('enable_maturity_check') == True:
                print_success("✅ Проверка зрелости включена")
                print_info(f"Минимум свечей: {config.get('min_candles_for_maturity', 200)}")
                print_info(f"RSI min: {config.get('min_rsi_low', 35)}")
                print_info(f"RSI max: {config.get('max_rsi_high', 65)}")
            else:
                print_error("❌ Проверка зрелости ВЫКЛЮЧЕНА!")
        else:
            print_error(f"Ошибка проверки фильтра зрелости: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при проверке фильтра зрелости: {e}")
    
    # Тест 14: Проверка ручных позиций
    print_test("Проверка фильтра ручных позиций")
    try:
        response = requests.get(f"{BOTS_SERVICE_URL}/api/bots/coins-with-rsi", timeout=5)
        data = response.json()
        
        if data.get('success'):
            manual_positions = data.get('manual_positions', [])
            coins = data.get('coins', {})
            
            if manual_positions:
                print_success(f"Ручные позиции определены: {len(manual_positions)} монет")
                print_info(f"Символы: {manual_positions[:5]}")
                
                # Проверяем что символы БЕЗ USDT
                has_usdt = any('USDT' in symbol for symbol in manual_positions)
                if not has_usdt:
                    print_success("✅ Символы без USDT (корректный формат)")
                else:
                    print_error("❌ Символы содержат USDT (некорректный формат)")
            else:
                print_info("Нет ручных позиций на бирже")
        else:
            print_error(f"Ошибка проверки ручных позиций: {data.get('error')}")
    except Exception as e:
        print_error(f"Исключение при проверке ручных позиций: {e}")

# ==========================================
# ТЕСТЫ СТРАНИЦ UI
# ==========================================

def test_ui_pages():
    """Проверка страниц UI"""
    print_header("ПРОВЕРКА СТРАНИЦ UI")
    
    # Тест 15: Главная страница
    print_test("Главная страница загружается")
    try:
        response = requests.get(f"{APP_SERVICE_URL}/", timeout=5)
        if response.status_code == 200 and len(response.text) > 1000:
            print_success("Главная страница загружена")
        else:
            print_error(f"Проблема с главной страницей: код {response.status_code}")
    except Exception as e:
        print_error(f"Исключение при загрузке главной страницы: {e}")
    
    # Тест 16: Страница ботов
    print_test("Страница ботов доступна")
    try:
        # Проверяем что HTML файл существует
        import os
        bots_page = 'templates/pages/bots.html'
        if os.path.exists(bots_page):
            print_success("Файл bots.html существует")
            
            # Проверяем наличие ключевых элементов
            with open(bots_page, 'r', encoding='utf-8') as f:
                content = f.read()
                
                required_elements = [
                    'id="saveConfigBtn"',
                    'id="coinSearchInput"',
                    'id="clearSearchBtn"',
                    'id="rsiTimeFilterEnabled"',
                    'id="rsiTimeFilterCandles"'
                ]
                
                missing_elements = [elem for elem in required_elements if elem not in content]
                
                if not missing_elements:
                    print_success("Все ключевые элементы UI присутствуют")
                else:
                    print_error(f"Отсутствуют элементы: {missing_elements}")
        else:
            print_error("Файл bots.html не найден")
    except Exception as e:
        print_error(f"Исключение при проверке страницы ботов: {e}")
    
    # Тест 17: JavaScript файлы
    print_test("JavaScript файлы существуют")
    try:
        import os
        js_files = [
            'static/js/managers/bots_manager.js',
            'static/js/app.js'
        ]
        
        all_exist = True
        for js_file in js_files:
            if os.path.exists(js_file):
                print_info(f"✅ {js_file}")
            else:
                print_error(f"❌ {js_file} не найден")
                all_exist = False
        
        if all_exist:
            print_success("Все JavaScript файлы на месте")
    except Exception as e:
        print_error(f"Исключение при проверке JavaScript файлов: {e}")

# ==========================================
# ТЕСТЫ ФАЙЛОВ ДАННЫХ
# ==========================================

def test_data_files():
    """Проверка файлов данных"""
    print_header("ПРОВЕРКА ФАЙЛОВ ДАННЫХ")
    
    # Тест 18: auto_bot_config.json
    print_test("Файл auto_bot_config.json")
    try:
        import os
        config_file = 'data/auto_bot_config.json'
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print_success("Файл конфигурации загружен")
            
            # Проверяем новые параметры
            if 'rsi_time_filter_enabled' in config:
                print_success(f"✅ rsi_time_filter_enabled: {config['rsi_time_filter_enabled']}")
            else:
                print_error("❌ rsi_time_filter_enabled отсутствует")
            
            if 'rsi_time_filter_candles' in config:
                print_success(f"✅ rsi_time_filter_candles: {config['rsi_time_filter_candles']}")
            else:
                print_error("❌ rsi_time_filter_candles отсутствует")
            
            # Проверяем критические настройки
            if config.get('enabled') == False:
                print_success("✅ Auto Bot выключен в файле")
            else:
                print_warning("⚠️ Auto Bot включен в файле!")
                
        else:
            print_error("Файл auto_bot_config.json не найден")
    except Exception as e:
        print_error(f"Исключение при проверке файла конфигурации: {e}")
    
    # Тест 19: bots_state.json
    print_test("Файл bots_state.json")
    try:
        import os
        state_file = 'data/bots_state.json'
        
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            bots = state.get('bots', {})
            print_success(f"Файл состояния загружен: {len(bots)} ботов")
            
            if len(bots) > 0:
                print_info(f"Боты в файле: {list(bots.keys())[:5]}")
            else:
                print_info("Нет сохраненных ботов")
        else:
            print_error("Файл bots_state.json не найден")
    except Exception as e:
        print_error(f"Исключение при проверке файла состояния: {e}")
    
    # Тест 20: mature_coins.json
    print_test("Файл mature_coins.json")
    try:
        import os
        mature_file = 'data/mature_coins.json'
        
        if os.path.exists(mature_file):
            with open(mature_file, 'r', encoding='utf-8') as f:
                mature_coins = json.load(f)
            
            print_success(f"Файл зрелых монет загружен: {len(mature_coins)} монет")
            
            if len(mature_coins) > 0:
                print_info(f"Зрелые монеты: {list(mature_coins.keys())[:5]}")
        else:
            print_warning("Файл mature_coins.json не найден (будет создан)")
    except Exception as e:
        print_error(f"Исключение при проверке файла зрелых монет: {e}")

# ==========================================
# ТЕСТЫ API ENDPOINTS
# ==========================================

def test_api_endpoints():
    """Проверка всех критических API endpoints"""
    print_header("ПРОВЕРКА API ENDPOINTS")
    
    endpoints = [
        ('GET', '/api/status', 'Статус сервиса'),
        ('GET', '/api/bots/list', 'Список ботов'),
        ('GET', '/api/bots/auto-bot', 'Конфигурация Auto Bot'),
        ('GET', '/api/bots/coins-with-rsi', 'Монеты с RSI'),
        ('GET', '/api/bots/mature-coins', 'Зрелые монеты'),
        ('GET', '/api/bots/history', 'История торговли'),
        ('GET', '/api/bots/whitelist', 'Белый список'),
        ('GET', '/api/bots/blacklist', 'Черный список'),
        ('GET', '/api/bots/process-state', 'Состояние процессов'),
        ('GET', '/api/bots/system-config', 'Системная конфигурация'),
    ]
    
    for method, endpoint, description in endpoints:
        print_test(f"{method} {endpoint} - {description}")
        try:
            url = f"{BOTS_SERVICE_URL}{endpoint}"
            response = requests.request(method, url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print_success(f"Endpoint работает")
                else:
                    print_error(f"Endpoint вернул success=false: {data.get('error')}")
            else:
                print_error(f"Endpoint вернул код {response.status_code}")
        except Exception as e:
            print_error(f"Ошибка запроса: {e}")

# ==========================================
# ИТОГОВЫЙ ОТЧЕТ
# ==========================================

def print_final_report():
    """Печатает итоговый отчет"""
    print_header("ИТОГОВЫЙ ОТЧЕТ")
    
    print(f"\n{Fore.CYAN}Всего тестов: {tests_total}")
    print(f"{Fore.GREEN}Успешно: {tests_passed}")
    print(f"{Fore.RED}Провалено: {tests_failed}")
    
    success_rate = (tests_passed / tests_total * 100) if tests_total > 0 else 0
    
    if success_rate == 100:
        print(f"\n{Fore.GREEN}{'='*80}")
        print(f"{Fore.GREEN}🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! СИСТЕМА РАБОТАЕТ ОТЛИЧНО! 🎉")
        print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    elif success_rate >= 80:
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}⚠️ БОЛЬШИНСТВО ТЕСТОВ ПРОЙДЕНО ({success_rate:.1f}%)")
        print(f"{Fore.YELLOW}{'='*80}{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}{'='*80}")
        print(f"{Fore.RED}❌ МНОГО ОШИБОК! ТРЕБУЕТСЯ ВНИМАНИЕ! ({success_rate:.1f}%)")
        print(f"{Fore.RED}{'='*80}{Style.RESET_ALL}")
    
    # Сохраняем отчет в файл
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': tests_total,
            'passed': tests_passed,
            'failed': tests_failed,
            'success_rate': success_rate
        }
        
        with open('logs/test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n{Fore.BLUE}📄 Отчет сохранен: logs/test_report.json{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Ошибка сохранения отчета: {e}{Style.RESET_ALL}")

# ==========================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==========================================

def main():
    """Главная функция тестирования"""
    print(f"{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ INFOBOT")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"\n{Fore.BLUE}Дата: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}{Style.RESET_ALL}")
    
    try:
        # Запускаем все тесты
        test_services_online()
        test_configuration()
        test_rsi_data_and_filters()
        test_bots_management()
        test_bot_history()
        test_filters()
        test_protection_mechanisms()
        test_api_endpoints()
        test_data_files()
        
        # Итоговый отчет
        print_final_report()
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}⚠️ Тестирование прервано пользователем{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n\n{Fore.RED}❌ Критическая ошибка: {e}{Style.RESET_ALL}")

if __name__ == '__main__':
    main()

