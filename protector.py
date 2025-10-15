#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROTECTOR ENHANCED - УНИВЕРСАЛЬНАЯ СИСТЕМА МОНИТОРИНГА
=====================================================

Постоянный мониторинг торговой системы с расширенными возможностями:
- Мониторинг автобота
- Мониторинг активных ботов
- Мониторинг торговых операций
- Проверка зрелости монет
- Анализ фильтров
- И многое другое...

Автор: AI Assistant
Дата: 2025-10-09
"""

import time
import psutil
import requests
import json
import sys
import os
import io
import logging
from datetime import datetime
from colorama import Fore, Style, init
from utils.log_rotation import setup_logger_with_rotation

# Исправление кодировки для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Инициализация Colorama
init(autoreset=True)

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

BOTS_PORT = 5001
CHECK_INTERVAL = 2  # секунды
BOTS_STATE_FILE = "data/bots_state.json"
AUTO_BOT_CONFIG_FILE = "data/auto_bot_config.json"
LOG_FILE = "logs/protector.log"
API_BASE_URL = f"http://127.0.0.1:{BOTS_PORT}/api/bots"

# Создаем директорию для логов если её нет
os.makedirs("logs", exist_ok=True)

# ============================================================================
# ЛОГИРОВАНИЕ С РОТАЦИЕЙ
# ============================================================================

# Создаем логгер с автоматической ротацией при превышении 10MB
file_logger = setup_logger_with_rotation(
    name='Protector',
    log_file=LOG_FILE,
    level=logging.INFO,
    max_bytes=10 * 1024 * 1024,  # 10MB
    format_string='%(asctime)s - %(levelname)s - %(message)s'
)

def log(message, level="INFO"):
    """Логирование с цветным выводом и автоматической ротацией файлов"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Цвета для разных уровней
    colors = {
        "INFO": Fore.CYAN,
        "SUCCESS": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT
    }
    
    color = colors.get(level, Fore.WHITE)
    formatted_message = f"[{timestamp}] {color}[{level}]{Style.RESET_ALL} {message}"
    
    # Вывод в консоль
    print(formatted_message)
    
    # Запись в лог файл через logger с ротацией
    try:
        log_level = getattr(logging, level, logging.INFO)
        file_logger.log(log_level, message)
    except Exception as e:
        print(f"Ошибка записи в лог: {e}")

# ============================================================================
# СИСТЕМНЫЕ ФУНКЦИИ
# ============================================================================

def find_bots_process():
    """Находит PID процесса, слушающего на порту 5001"""
    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == BOTS_PORT and conn.status == 'LISTEN':
                return conn.pid
        return None
    except Exception as e:
        log(f"Ошибка поиска процесса: {e}", "ERROR")
        return None

def kill_bots_process(pid):
    """Останавливает процесс bots.py"""
    try:
        process = psutil.Process(pid)
        log(f"Останавливаем процесс bots.py (PID: {pid})", "WARNING")
        
        # Сначала пытаемся graceful shutdown
        process.terminate()
        process.wait(timeout=5)
        
        # Если процесс еще работает - принудительно убиваем
        if process.is_running():
            process.kill()
            process.wait(timeout=3)
        
        log(f"Процесс bots.py (PID: {pid}) успешно остановлен", "SUCCESS")
        return True
        
    except psutil.NoSuchProcess:
        log(f"Процесс bots.py (PID: {pid}) уже не существует", "WARNING")
        return True
    except Exception as e:
        log(f"Ошибка остановки процесса bots.py (PID: {pid}): {e}", "ERROR")
        return False

def check_service_online():
    """Проверяет, отвечает ли сервис bots.py"""
    try:
        response = requests.get(f"http://127.0.0.1:{BOTS_PORT}/api/status", timeout=1)
        return response.status_code == 200
    except:
        return False

def check_auto_bot_enabled():
    """Проверяет статус автобота"""
    try:
        response = requests.get(f"{API_BASE_URL}/auto-bot", timeout=1)
        if response.status_code == 200:
            config = response.json()
            return config.get('config', {}).get('enabled', False)
        return False
    except:
        return False

def check_active_bots():
    """Проверяет количество активных ботов"""
    try:
        response = requests.get(f"{API_BASE_URL}/list", timeout=1)
        if response.status_code == 200:
            data = response.json()
            return len(data.get('bots', []))
        return -1
    except:
        return -1

def check_bots_state_file():
    """Проверяет состояние ботов в файле"""
    try:
        if os.path.exists(BOTS_STATE_FILE):
            with open(BOTS_STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return len(data.get('bots', {}))
        return 0
    except Exception as e:
        log(f"Ошибка чтения файла состояния: {e}", "ERROR")
        return -1

def check_auto_bot_config_file():
    """Проверяет конфигурацию автобота в файле"""
    try:
        if os.path.exists(AUTO_BOT_CONFIG_FILE):
            with open(AUTO_BOT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('enabled', False)
        return False
    except Exception as e:
        log(f"Ошибка чтения конфигурации: {e}", "ERROR")
        return False

# ============================================================================
# РАСШИРЕННЫЙ МОНИТОРИНГ
# ============================================================================

def perform_extended_monitoring(current_time):
    """Выполняет расширенный мониторинг системы"""
    try:
        # Проверяем зрелость монет
        mature_coins = check_mature_coins()
        if mature_coins is not None:
            log(f"📊 [{current_time}] Зрелых монет: {mature_coins}", "INFO")
        
        # Проверяем RSI данные
        rsi_status = check_rsi_data_status()
        if rsi_status:
            log(f"📈 [{current_time}] RSI данные: {rsi_status}", "INFO")
        
        # Проверяем позиции на бирже
        exchange_positions = check_exchange_positions()
        if exchange_positions is not None:
            log(f"💼 [{current_time}] Позиций на бирже: {exchange_positions}", "INFO")
        
        # Проверяем фильтры автобота
        filters_status = check_autobot_filters()
        if filters_status:
            log(f"🔍 [{current_time}] Фильтры: {filters_status}", "INFO")
            
    except Exception as e:
        log(f"Ошибка расширенного мониторинга: {e}", "ERROR")

def check_mature_coins():
    """Проверяет количество зрелых монет"""
    try:
        mature_file = "data/mature_coins.json"
        if os.path.exists(mature_file):
            with open(mature_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return len(data.get('mature_coins', {}))
        return 0
    except:
        return None

def check_rsi_data_status():
    """Проверяет статус RSI данных"""
    try:
        response = requests.get(f"{API_BASE_URL}/coins-with-rsi", timeout=2)
        if response.status_code == 200:
            data = response.json()
            total = data.get('total', 0)
            updating = data.get('update_in_progress', False)
            return f"{total} монет, обновление: {'да' if updating else 'нет'}"
        return None
    except:
        return None

def check_exchange_positions():
    """Проверяет позиции на бирже"""
    try:
        response = requests.get(f"{API_BASE_URL}/positions", timeout=2)
        if response.status_code == 200:
            data = response.json()
            positions = data.get('positions', [])
            return len([p for p in positions if abs(float(p.get('size', 0))) > 0])
        return None
    except:
        return None

def check_autobot_filters():
    """Проверяет статус фильтров автобота"""
    try:
        response = requests.get(f"{API_BASE_URL}/auto-bot-config", timeout=1)
        if response.status_code == 200:
            config = response.json()
            maturity_check = config.get('enable_maturity_check', False)
            dump_protection = config.get('enable_dump_protection', False)
            trend_filters = config.get('avoid_down_trend', False) or config.get('avoid_up_trend', False)
            
            filters = []
            if maturity_check:
                filters.append("зрелость")
            if dump_protection:
                filters.append("дампинг")
            if trend_filters:
                filters.append("тренды")
            
            return ", ".join(filters) if filters else "отключены"
        return None
    except:
        return None

# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def print_header():
    """Выводит заголовок программы"""
    print(f"{Fore.CYAN}{'='*60}")
    print(f"🛡️  PROTECTOR ENHANCED - УНИВЕРСАЛЬНАЯ СИСТЕМА МОНИТОРИНГА")
    print(f"{'='*60}")
    print(f"📊 Режим: ПОСТОЯННЫЙ МОНИТОРИНГ")
    print(f"🔄 Интервал проверки: {CHECK_INTERVAL} секунд")
    print(f"📝 Лог файл: {LOG_FILE}")
    print(f"🌐 API: http://127.0.0.1:{BOTS_PORT}")
    print(f"{'='*60}{Style.RESET_ALL}")
    print()

def main():
    """Основная функция мониторинга - ПОСТОЯННАЯ РАБОТА"""
    print_header()
    
    log("🛡️ PROTECTOR ENHANCED - УНИВЕРСАЛЬНАЯ СИСТЕМА МОНИТОРИНГА", "INFO")
    log("📊 Режим: ПОСТОЯННЫЙ МОНИТОРИНГ (без самоостановки)", "INFO")
    log("🔄 Нажмите Ctrl+C для остановки мониторинга\n", "INFO")
    
    check_counter = 0
    last_service_status = None
    last_bots_count = 0
    last_autobot_status = None
    
    try:
        while True:
            check_counter += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # === ПРОВЕРКА 1: СТАТУС СЕРВИСА ===
            service_online = check_service_online()
            if service_online != last_service_status:
                if service_online:
                    log(f"🟢 [{current_time}] Сервис bots.py ЗАПУЩЕН", "SUCCESS")
                else:
                    log(f"🔴 [{current_time}] Сервис bots.py НЕ ОТВЕЧАЕТ", "WARNING")
                last_service_status = service_online
            
            # Если сервис не отвечает - продолжаем мониторинг
            if not service_online:
                if check_counter % 15 == 0:  # Каждые 30 секунд
                    log(f"⏳ [{current_time}] Ожидание запуска сервиса... (проверка #{check_counter})", "INFO")
                time.sleep(CHECK_INTERVAL)
                continue
            
            # === ПРОВЕРКА 2: АВТОБОТ ===
            auto_bot_enabled = check_auto_bot_enabled()
            if auto_bot_enabled != last_autobot_status:
                if auto_bot_enabled:
                    log(f"🚨 [{current_time}] КРИТИЧЕСКАЯ УГРОЗА! АВТОБОТ ВКЛЮЧЕН!", "CRITICAL")
                    log(f"🛑 [{current_time}] НЕМЕДЛЕННО ОСТАНАВЛИВАЕМ СИСТЕМУ!", "CRITICAL")
                    pid = find_bots_process()
                    if pid:
                        if kill_bots_process(pid):
                            log(f"✅ [{current_time}] Система успешно остановлена!", "SUCCESS")
                            log(f"📝 [{current_time}] Причина: Автобот был включен", "CRITICAL")
                            # НЕ ВЫХОДИМ - продолжаем мониторинг!
                            log(f"🔄 [{current_time}] Продолжаем мониторинг...", "INFO")
                        else:
                            log(f"❌ [{current_time}] Не удалось остановить систему!", "ERROR")
                    else:
                        log(f"❌ [{current_time}] Не удалось найти процесс bots.py!", "ERROR")
                else:
                    log(f"✅ [{current_time}] Автобот отключен", "SUCCESS")
                last_autobot_status = auto_bot_enabled
            
            # === ПРОВЕРКА 3: АКТИВНЫЕ БОТЫ ===
            active_bots = check_active_bots()
            if active_bots != last_bots_count:
                if active_bots > 0:
                    log(f"🚨 [{current_time}] КРИТИЧЕСКАЯ УГРОЗА! Обнаружено {active_bots} активных ботов!", "CRITICAL")
                    log(f"🛑 [{current_time}] НЕМЕДЛЕННО ОСТАНАВЛИВАЕМ СИСТЕМУ!", "CRITICAL")
                    pid = find_bots_process()
                    if pid:
                        if kill_bots_process(pid):
                            log(f"✅ [{current_time}] Система успешно остановлена!", "SUCCESS")
                            log(f"📝 [{current_time}] Причина: Обнаружено {active_bots} активных ботов", "CRITICAL")
                            # НЕ ВЫХОДИМ - продолжаем мониторинг!
                            log(f"🔄 [{current_time}] Продолжаем мониторинг...", "INFO")
                        else:
                            log(f"❌ [{current_time}] Не удалось остановить систему!", "ERROR")
                    else:
                        log(f"❌ [{current_time}] Не удалось найти процесс bots.py!", "ERROR")
                else:
                    log(f"✅ [{current_time}] Активных ботов: {active_bots}", "SUCCESS")
                last_bots_count = active_bots
            
            # === ПРОВЕРКА 4: ДОПОЛНИТЕЛЬНЫЙ МОНИТОРИНГ ===
            # Проверяем зрелость монет, фильтры и другие параметры
            if check_counter % 10 == 0:  # Каждые 20 секунд
                perform_extended_monitoring(current_time)
            
            # === ПЕРИОДИЧЕСКИЙ ОТЧЕТ ===
            if check_counter % 30 == 0:  # Каждую минуту
                log(f"📊 [{current_time}] СТАТУС СИСТЕМЫ (проверка #{check_counter}):", "INFO")
                log(f"   🟢 Сервис: {'работает' if service_online else 'не отвечает'}", "INFO")
                log(f"   🤖 Автобот: {'включен' if auto_bot_enabled else 'отключен'}", "INFO")
                log(f"   📈 Активных ботов: {active_bots}", "INFO")
                log(f"   ⏱️ Время работы: {check_counter * CHECK_INTERVAL} сек", "INFO")
                log("", "INFO")
            
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        log(f"\n🛑 [{datetime.now().strftime('%H:%M:%S')}] Получен сигнал остановки (Ctrl+C)", "WARNING")
        log("🔄 Мониторинг остановлен пользователем", "INFO")
    except Exception as e:
        log(f"❌ [{datetime.now().strftime('%H:%M:%S')}] Неожиданная ошибка: {e}", "ERROR")
        log("🔄 Мониторинг остановлен из-за ошибки", "ERROR")
    finally:
        log(f"🏁 [{datetime.now().strftime('%H:%M:%S')}] Мониторинг завершен", "INFO")

# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)
