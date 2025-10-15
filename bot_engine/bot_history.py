#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль истории торговых ботов
Логирование всех действий ботов: запуск, остановка, сигналы, открытие/закрытие позиций
"""

import os
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Файл для хранения истории
HISTORY_FILE = 'data/bot_history.json'

# Типы действий
ACTION_TYPES = {
    'BOT_START': 'Запуск бота',
    'BOT_STOP': 'Остановка бота',
    'SIGNAL': 'Торговый сигнал',
    'POSITION_OPENED': 'Открытие позиции',
    'POSITION_CLOSED': 'Закрытие позиции',
    'STOP_LOSS': 'Срабатывание Stop Loss',
    'TAKE_PROFIT': 'Срабатывание Take Profit',
    'TRAILING_STOP': 'Срабатывание Trailing Stop',
    'ERROR': 'Ошибка бота'
}


class BotHistoryManager:
    """Менеджер истории торговых ботов"""
    
    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = history_file
        self.lock = threading.Lock()
        self.history = []
        self.trades = []
        
        # Создаем директорию data если её нет
        os.makedirs('data', exist_ok=True)
        
        # Загружаем историю из файла
        self._load_history()
    
    def _load_history(self):
        """Загружает историю из файла"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get('history', [])
                    self.trades = data.get('trades', [])
                    logger.info(f"[BOT_HISTORY] ✅ Загружено записей: {len(self.history)} действий, {len(self.trades)} сделок")
            else:
                logger.info("[BOT_HISTORY] 📝 Файл истории не найден, создается новый")
                self.history = []
                self.trades = []
        except Exception as e:
            logger.error(f"[BOT_HISTORY] ❌ Ошибка загрузки истории: {e}")
            self.history = []
            self.trades = []
    
    def _save_history(self):
        """Сохраняет историю в файл"""
        try:
            with self.lock:
                data = {
                    'history': self.history,
                    'trades': self.trades,
                    'last_update': datetime.now().isoformat()
                }
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[BOT_HISTORY] ❌ Ошибка сохранения истории: {e}")
    
    def _add_history_entry(self, entry: Dict[str, Any]):
        """Добавляет запись в историю"""
        with self.lock:
            self.history.append(entry)
            # Ограничиваем размер истории (последние 10000 записей)
            if len(self.history) > 10000:
                self.history = self.history[-10000:]
        self._save_history()
    
    def _add_trade_entry(self, trade: Dict[str, Any]):
        """Добавляет запись о сделке"""
        with self.lock:
            self.trades.append(trade)
            # Ограничиваем размер (последние 5000 сделок)
            if len(self.trades) > 5000:
                self.trades = self.trades[-5000:]
        self._save_history()
    
    # ==================== Функции логирования ====================
    
    def log_bot_start(self, bot_id: str, symbol: str, direction: str, config: Dict = None):
        """Логирование запуска бота"""
        entry = {
            'id': f"start_{bot_id}_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'action_type': 'BOT_START',
            'action_name': ACTION_TYPES['BOT_START'],
            'bot_id': bot_id,
            'symbol': symbol,
            'direction': direction,  # LONG или SHORT
            'config': config or {},
            'details': f"Запущен бот {direction} для {symbol}"
        }
        self._add_history_entry(entry)
        logger.info(f"[BOT_HISTORY] 🚀 {entry['details']}")
    
    def log_bot_stop(self, bot_id: str, symbol: str, reason: str = None, pnl: float = None):
        """Логирование остановки бота"""
        entry = {
            'id': f"stop_{bot_id}_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'action_type': 'BOT_STOP',
            'action_name': ACTION_TYPES['BOT_STOP'],
            'bot_id': bot_id,
            'symbol': symbol,
            'reason': reason or 'Ручная остановка',
            'pnl': pnl,
            'details': f"Остановлен бот для {symbol}: {reason or 'Ручная остановка'}"
        }
        if pnl is not None:
            entry['details'] += f" (PnL: {pnl:.2f} USDT)"
        
        self._add_history_entry(entry)
        logger.info(f"[BOT_HISTORY] 🛑 {entry['details']}")
    
    def log_bot_signal(self, symbol: str, signal_type: str, rsi: float, price: float, details: Dict = None):
        """Логирование торгового сигнала"""
        entry = {
            'id': f"signal_{symbol}_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'action_type': 'SIGNAL',
            'action_name': ACTION_TYPES['SIGNAL'],
            'symbol': symbol,
            'signal_type': signal_type,  # ENTER_LONG, ENTER_SHORT, EXIT
            'rsi': rsi,
            'price': price,
            'details_data': details or {},
            'details': f"Сигнал {signal_type} для {symbol} (RSI: {rsi:.2f}, цена: {price:.2f})"
        }
        self._add_history_entry(entry)
        logger.info(f"[BOT_HISTORY] 📊 {entry['details']}")
    
    def log_position_opened(self, bot_id: str, symbol: str, direction: str, size: float, 
                           entry_price: float, stop_loss: float = None, take_profit: float = None):
        """Логирование открытия позиции"""
        entry = {
            'id': f"open_{bot_id}_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'action_type': 'POSITION_OPENED',
            'action_name': ACTION_TYPES['POSITION_OPENED'],
            'bot_id': bot_id,
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'details': f"Открыта позиция {direction} для {symbol}: размер {size}, цена входа {entry_price:.4f}"
        }
        self._add_history_entry(entry)
        
        # Также добавляем в сделки
        trade = {
            'id': f"trade_{bot_id}_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'bot_id': bot_id,
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'exit_price': None,
            'pnl': None,
            'status': 'OPEN'
        }
        self._add_trade_entry(trade)
        
        logger.info(f"[BOT_HISTORY] 📈 {entry['details']}")
    
    def log_position_closed(self, bot_id: str, symbol: str, direction: str, exit_price: float, 
                           pnl: float, roi: float, reason: str = None):
        """Логирование закрытия позиции"""
        entry = {
            'id': f"close_{bot_id}_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'action_type': 'POSITION_CLOSED',
            'action_name': ACTION_TYPES['POSITION_CLOSED'],
            'bot_id': bot_id,
            'symbol': symbol,
            'direction': direction,
            'exit_price': exit_price,
            'pnl': pnl,
            'roi': roi,
            'reason': reason or 'Ручное закрытие',
            'details': f"Закрыта позиция {direction} для {symbol}: цена выхода {exit_price:.4f}, PnL: {pnl:.2f} USDT ({roi:.2f}%)"
        }
        self._add_history_entry(entry)
        
        # Обновляем сделку
        with self.lock:
            for trade in reversed(self.trades):
                if trade['bot_id'] == bot_id and trade['symbol'] == symbol and trade['status'] == 'OPEN':
                    trade['exit_price'] = exit_price
                    trade['pnl'] = pnl
                    trade['roi'] = roi
                    trade['status'] = 'CLOSED'
                    trade['close_timestamp'] = datetime.now().isoformat()
                    trade['close_reason'] = reason
                    break
        self._save_history()
        
        logger.info(f"[BOT_HISTORY] 💰 {entry['details']}")
    
    # ==================== Методы получения данных ====================
    
    def get_bot_history(self, symbol: Optional[str] = None, action_type: Optional[str] = None, 
                       limit: int = 100) -> List[Dict]:
        """
        Получает историю действий ботов
        
        Args:
            symbol: Фильтр по символу (например, BTCUSDT)
            action_type: Тип действия (BOT_START, BOT_STOP, SIGNAL и т.д.)
            limit: Максимальное количество записей
        
        Returns:
            Список записей истории (от новых к старым)
        """
        with self.lock:
            filtered = self.history.copy()
            
            # Фильтр по символу
            if symbol:
                filtered = [h for h in filtered if h.get('symbol') == symbol]
            
            # Фильтр по типу действия
            if action_type:
                filtered = [h for h in filtered if h.get('action_type') == action_type]
            
            # Сортируем от новых к старым
            filtered.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Ограничиваем количество
            return filtered[:limit]
    
    def get_bot_trades(self, symbol: Optional[str] = None, trade_type: Optional[str] = None,
                      limit: int = 100) -> List[Dict]:
        """
        Получает историю торговых сделок
        
        Args:
            symbol: Фильтр по символу
            trade_type: Фильтр по направлению (LONG/SHORT)
            limit: Максимальное количество записей
        
        Returns:
            Список сделок (от новых к старым)
        """
        with self.lock:
            filtered = self.trades.copy()
            
            # Фильтр по символу
            if symbol:
                filtered = [t for t in filtered if t.get('symbol') == symbol]
            
            # Фильтр по типу сделки
            if trade_type:
                filtered = [t for t in filtered if t.get('direction') == trade_type]
            
            # Сортируем от новых к старым
            filtered.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Ограничиваем количество
            return filtered[:limit]
    
    def get_bot_statistics(self, symbol: Optional[str] = None) -> Dict:
        """
        Получает статистику по ботам
        
        Args:
            symbol: Фильтр по символу (если None - вся статистика)
        
        Returns:
            Словарь со статистикой
        """
        with self.lock:
            trades = self.trades.copy()
            
            # Фильтр по символу
            if symbol:
                trades = [t for t in trades if t.get('symbol') == symbol]
            
            # Только закрытые сделки
            closed_trades = [t for t in trades if t.get('status') == 'CLOSED']
            
            if not closed_trades:
                return {
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'best_trade': None,
                    'worst_trade': None
                }
            
            # Расчет статистики
            profitable = [t for t in closed_trades if t.get('pnl', 0) > 0]
            losing = [t for t in closed_trades if t.get('pnl', 0) < 0]
            
            total_pnl = sum(t.get('pnl', 0) for t in closed_trades)
            avg_pnl = total_pnl / len(closed_trades) if closed_trades else 0
            
            # Лучшая и худшая сделки
            best_trade = max(closed_trades, key=lambda x: x.get('pnl', 0)) if closed_trades else None
            worst_trade = min(closed_trades, key=lambda x: x.get('pnl', 0)) if closed_trades else None
            
            return {
                'total_trades': len(closed_trades),
                'profitable_trades': len(profitable),
                'losing_trades': len(losing),
                'win_rate': (len(profitable) / len(closed_trades) * 100) if closed_trades else 0,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'symbol': symbol if symbol else 'ALL'
            }
    
    def clear_history(self, symbol: Optional[str] = None):
        """
        Очищает историю
        
        Args:
            symbol: Если указан, очищает только для этого символа, иначе всю историю
        """
        with self.lock:
            if symbol:
                self.history = [h for h in self.history if h.get('symbol') != symbol]
                self.trades = [t for t in self.trades if t.get('symbol') != symbol]
                logger.info(f"[BOT_HISTORY] 🗑️ Очищена история для {symbol}")
            else:
                self.history = []
                self.trades = []
                logger.info("[BOT_HISTORY] 🗑️ Вся история очищена")
        
        self._save_history()


# ==================== Глобальный экземпляр ====================

bot_history_manager = BotHistoryManager()


# ==================== Функции-обертки для удобства ====================

def log_bot_start(bot_id: str, symbol: str, direction: str, config: Dict = None):
    """Логирование запуска бота"""
    bot_history_manager.log_bot_start(bot_id, symbol, direction, config)


def log_bot_stop(bot_id: str, symbol: str, reason: str = None, pnl: float = None):
    """Логирование остановки бота"""
    bot_history_manager.log_bot_stop(bot_id, symbol, reason, pnl)


def log_bot_signal(symbol: str, signal_type: str, rsi: float, price: float, details: Dict = None):
    """Логирование торгового сигнала"""
    bot_history_manager.log_bot_signal(symbol, signal_type, rsi, price, details)


def log_position_opened(bot_id: str, symbol: str, direction: str, size: float, 
                       entry_price: float, stop_loss: float = None, take_profit: float = None):
    """Логирование открытия позиции"""
    bot_history_manager.log_position_opened(bot_id, symbol, direction, size, entry_price, 
                                           stop_loss, take_profit)


def log_position_closed(bot_id: str, symbol: str, direction: str, exit_price: float, 
                       pnl: float, roi: float, reason: str = None):
    """Логирование закрытия позиции"""
    bot_history_manager.log_position_closed(bot_id, symbol, direction, exit_price, 
                                           pnl, roi, reason)


# ==================== Демо-данные ====================

def create_demo_data() -> bool:
    """Создает демо-данные для тестирования"""
    try:
        import random
        from datetime import timedelta
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        
        logger.info("[BOT_HISTORY] 📝 Создание демо-данных...")
        
        for i in range(20):
            symbol = random.choice(symbols)
            direction = random.choice(['LONG', 'SHORT'])
            bot_id = f"demo_bot_{i}"
            
            # Запуск бота
            log_bot_start(bot_id, symbol, direction, {'mode': 'demo'})
            
            # Сигнал
            rsi = random.uniform(25, 75)
            price = random.uniform(1000, 50000)
            log_bot_signal(symbol, f"ENTER_{direction}", rsi, price)
            
            # Открытие позиции
            entry_price = price
            size = random.uniform(0.001, 0.1)
            log_position_opened(bot_id, symbol, direction, size, entry_price)
            
            # Закрытие позиции (80% сделок)
            if random.random() < 0.8:
                exit_price = entry_price * random.uniform(0.95, 1.10)
                pnl = (exit_price - entry_price) * size if direction == 'LONG' else (entry_price - exit_price) * size
                roi = ((exit_price - entry_price) / entry_price * 100) if direction == 'LONG' else ((entry_price - exit_price) / entry_price * 100)
                
                log_position_closed(bot_id, symbol, direction, exit_price, pnl, roi, 
                                  random.choice(['Stop Loss', 'Take Profit', 'Ручное закрытие']))
                
                log_bot_stop(bot_id, symbol, 'Позиция закрыта', pnl)
        
        logger.info("[BOT_HISTORY] ✅ Демо-данные созданы успешно!")
        return True
        
    except Exception as e:
        logger.error(f"[BOT_HISTORY] ❌ Ошибка создания демо-данных: {e}")
        return False


if __name__ == '__main__':
    # Тест модуля
    print("=== Test modula bot_history.py ===\n")
    
    # Очистка
    bot_history_manager.clear_history()
    
    # Создаем демо-данные
    create_demo_data()
    
    # Получаем статистику
    stats = bot_history_manager.get_bot_statistics()
    print(f"\n[STATISTIKA]")
    print(f"  Vsego sdelok: {stats['total_trades']}")
    print(f"  Pribylnyh: {stats['profitable_trades']}")
    print(f"  Ubytochnyh: {stats['losing_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.2f}%")
    print(f"  Obschiy PnL: {stats['total_pnl']:.2f} USDT")
    
    # Получаем последние действия
    history = bot_history_manager.get_bot_history(limit=5)
    print(f"\n[POSLEDNIE 5 DEYSTVIY]:")
    for h in history:
        print(f"  [{h['timestamp']}] {h['action_name']}: {h.get('symbol', 'N/A')}")

