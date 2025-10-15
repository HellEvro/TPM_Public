"""
Умный менеджер обновления RSI
Обновляет RSI регулярно, но торговые сигналы только при закрытии свечи 6H
"""

import time
import threading
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Callable
from bot_engine.bot_config import SystemConfig

logger = logging.getLogger('SmartRSIManager')

class SmartRSIManager:
    """Умный менеджер обновления RSI с торговыми сигналами только при закрытии свечи"""
    
    def __init__(self, rsi_update_callback: Callable, trading_signal_callback: Optional[Callable] = None, exchange_obj=None):
        """
        Args:
            rsi_update_callback: Функция для обновления RSI данных
            trading_signal_callback: Функция для обработки торговых сигналов (опционально)
            exchange_obj: Объект биржи для передачи в callback
        """
        self.rsi_update_callback = rsi_update_callback
        self.trading_signal_callback = trading_signal_callback
        self.exchange_obj = exchange_obj
        self.shutdown_flag = threading.Event()
        self.last_update_time = 0
        
        # Настройки обновления RSI 
        self.monitoring_interval = 300  # 5 минут (плановое обновление)
        self.candle_close_tolerance = 600  # 10 минут допуска после закрытия свечи (для учета задержек)
        
        self.processed_candles = set()  # Уже обработанные свечи (по timestamp)
        
        logger.info(f"[SMART_RSI] 🧠 Умный менеджер RSI инициализирован")
        logger.info(f"[SMART_RSI] 📊 Плановое обновление: каждые {self.monitoring_interval//60} минут")
        logger.info(f"[SMART_RSI] 🎯 Торговые сигналы: только при обновлении после закрытия свечи 6H")
        logger.info(f"[SMART_RSI] ⚡ Оптимизация: нет частых проверок API, только плановые обновления")
    
    def get_next_6h_candle_close(self) -> int:
        """Возвращает timestamp следующего закрытия свечи 6H"""
        current_time = int(time.time())
        
        # Свечи 6H закрываются в: 00:00, 06:00, 12:00, 18:00 UTC
        current_dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
        current_hour = current_dt.hour
        
        # Определяем следующее время закрытия свечи
        next_closes = [0, 6, 12, 18]
        next_close_hour = None
        
        for close_hour in next_closes:
            if close_hour > current_hour:
                next_close_hour = close_hour
                break
        
        if next_close_hour is None:
            # Если все времена закрытия в текущем дне прошли, берем 00:00 следующего дня
            next_close_hour = 24
        
        # Создаем datetime для следующего закрытия
        next_close_dt = current_dt.replace(
            hour=next_close_hour % 24, 
            minute=0, 
            second=0, 
            microsecond=0
        )
        
        if next_close_hour == 24:
            next_close_dt = next_close_dt.replace(day=next_close_dt.day + 1, hour=0)
        
        return int(next_close_dt.timestamp())
    
    def get_time_to_candle_close(self) -> int:
        """Возвращает время в секундах до закрытия текущей свечи 6H"""
        next_close = self.get_next_6h_candle_close()
        current_time = int(time.time())
        return max(0, next_close - current_time)
    
    def get_last_6h_candle_close(self) -> int:
        """Возвращает timestamp последнего закрытия свечи 6H"""
        current_time = int(time.time())
        next_close = self.get_next_6h_candle_close()
        return next_close - (6 * 3600)  # Предыдущая свеча 6H назад
    

    
    def should_update_rsi(self) -> tuple[bool, str]:
        """
        Определяет, нужно ли обновлять RSI для мониторинга
        """
        current_time = int(time.time())
        time_since_last_update = current_time - self.last_update_time
        
        # 1. Если это первое обновление
        if self.last_update_time == 0:
            return True, "первое обновление"
        
        # 2. Регулярное обновление для мониторинга
        if time_since_last_update >= self.monitoring_interval:
            return True, f"регулярное обновление ({time_since_last_update//60}м прошло)"
        
        return False, f"обновление не требуется (следующее через {self.monitoring_interval - time_since_last_update}с)"
    
    def should_process_trading_signals_after_update(self) -> tuple[bool, str, int]:
        """
        Определяет, нужно ли обрабатывать торговые сигналы после обновления RSI
        (только если свеча недавно закрылась)
        """
        current_time = int(time.time())
        last_candle_close = self.get_last_6h_candle_close()
        
        # Проверяем, закрылась ли свеча недавно (в пределах допуска)
        time_since_close = current_time - last_candle_close
        
        if 0 <= time_since_close <= self.candle_close_tolerance:
            # Проверяем, что эту свечу мы еще не обрабатывали
            if last_candle_close not in self.processed_candles:
                return True, f"свеча закрылась {time_since_close//60}м назад", last_candle_close
        
        return False, f"свеча закрылась {time_since_close//60}м назад (слишком давно или уже обработана)", last_candle_close
    
    def check_significant_price_changes(self) -> bool:
        """
        Проверяет, произошли ли значительные изменения цен
        Пока заглушка - в будущем можно добавить мониторинг тикеров
        """
        # TODO: Реализовать мониторинг изменений цен через WebSocket или тикеры
        return False
    
    def get_next_update_time(self) -> int:
        """Возвращает время следующего планового обновления RSI"""
        return self.last_update_time + self.monitoring_interval
    

    
    def update_rsi_data(self):
        """Выполняет обновление RSI данных и проверяет необходимость торговых сигналов"""
        try:
            self.last_update_time = int(time.time())
            
            # Вызываем функцию обновления RSI
            logger.info(f"[SMART_RSI] 📊 Плановое обновление RSI данных...")
            self.rsi_update_callback()
            
            time_to_close = self.get_time_to_candle_close()
            hours = time_to_close // 3600
            minutes = (time_to_close % 3600) // 60
            
            if hours > 0:
                time_str = f"{hours}ч {minutes}м"
            else:
                time_str = f"{minutes}м"
                
            logger.info(f"[SMART_RSI] ✅ RSI данные обновлены | До закрытия свечи 6H: {time_str}")
            
            # Проверяем, нужно ли активировать торговые сигналы
            should_trade, trade_reason, candle_timestamp = self.should_process_trading_signals_after_update()
            if should_trade:
                logger.info("=" * 80)
                logger.info(f"[SMART_RSI] 🎯 ОБНАРУЖЕНО ЗАКРЫТИЕ СВЕЧИ 6H! ({trade_reason})")
                logger.info(f"[SMART_RSI] 🚨 АКТИВАЦИЯ ТОРГОВЫХ СИГНАЛОВ - ПРОВЕРКА УСЛОВИЙ ВХОДА/ВЫХОДА")
                logger.info("=" * 80)
                
                # Помечаем свечу как обработанную
                self.processed_candles.add(candle_timestamp)
                
                # Если есть callback для торговых сигналов, вызываем его
                if self.trading_signal_callback:
                    self.trading_signal_callback(candle_timestamp, exchange_obj=self.exchange_obj)
                else:
                    logger.warning(f"[SMART_RSI] ⚠️ Торговый callback не настроен")
                
                # Очищаем старые обработанные свечи (оставляем только последние 10)
                if len(self.processed_candles) > 10:
                    oldest_candles = sorted(self.processed_candles)[:-10]
                    for old_candle in oldest_candles:
                        self.processed_candles.remove(old_candle)
            else:
                logger.debug(f"[SMART_RSI] 💤 Торговые сигналы не требуются: {trade_reason}")
            
        except Exception as e:
            logger.error(f"[SMART_RSI] ❌ Ошибка обновления RSI: {e}")
    

    
    def run_smart_worker(self):
        """Основной цикл умного обновления RSI и проверки торговых сигналов"""
        logger.info("=" * 80)
        logger.info("[SMART_RSI] 🚀 ЗАПУСК ОПТИМИЗИРОВАННОЙ СИСТЕМЫ RSI")
        logger.info("[SMART_RSI] 📊 Режим: Плановое обновление каждые 60 минут")
        logger.info("[SMART_RSI] 🎯 Торговые сигналы: автоматически при обновлении после закрытия свечи 6H")
        logger.info("[SMART_RSI] ⚡ Нет частых проверок API - только эффективные плановые обновления")
        logger.info("=" * 80)
        
        # Первое обновление сразу
        self.update_rsi_data()
        
        while not self.shutdown_flag.is_set():
            try:
                # Проверяем только плановые обновления RSI
                should_update, update_reason = self.should_update_rsi()
                if should_update:
                    logger.info(f"[SMART_RSI] 📊 Время планового обновления: {update_reason}")
                    self.update_rsi_data()
                
                # Ждем 5 минут до следующей проверки (вместо каждой минуты)
                if self.shutdown_flag.wait(300):  # 5 минут
                    break
                    
            except Exception as e:
                logger.error(f"[SMART_RSI] ❌ Ошибка в умном воркере: {e}")
                if self.shutdown_flag.wait(30):
                    break
        
        logger.info("[SMART_RSI] 🛑 Умный воркер RSI остановлен")
    
    def start(self):
        """Запускает умный воркер в отдельном потоке"""
        self.worker_thread = threading.Thread(target=self.run_smart_worker, daemon=True)
        self.worker_thread.start()
        logger.info("[SMART_RSI] 🎯 Умный воркер RSI запущен в отдельном потоке")
    
    def stop(self):
        """Останавливает умный воркер"""
        logger.info("[SMART_RSI] 🛑 Остановка умного воркера RSI...")
        self.shutdown_flag.set()
        
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=5)
        
        logger.info("[SMART_RSI] ✅ Умный воркер RSI остановлен")
    
    def update_monitoring_interval(self, new_interval: int):
        """Обновляет интервал мониторинга RSI"""
        old_interval = self.monitoring_interval
        self.monitoring_interval = new_interval
        logger.info(f"[SMART_RSI] 🔄 Интервал мониторинга обновлен: {old_interval}с → {new_interval}с")
        logger.info(f"[SMART_RSI] 📊 Новый интервал: каждые {new_interval//60} минут")

    def get_status(self) -> dict:
        """Возвращает статус умного менеджера"""
        current_time = int(time.time())
        time_to_close = self.get_time_to_candle_close()
        next_update = self.get_next_update_time()
        last_candle_close = self.get_last_6h_candle_close()
        
        return {
            'monitoring_interval': self.monitoring_interval,
            'time_to_candle_close': time_to_close,
            'time_to_candle_close_formatted': f"{time_to_close//3600}ч {(time_to_close%3600)//60}м {time_to_close%60}с",
            'last_rsi_update': self.last_update_time,
            'last_rsi_update_ago': current_time - self.last_update_time if self.last_update_time > 0 else 0,
            'next_rsi_update': next_update,
            'next_rsi_update_in': max(0, next_update - current_time),
            'last_candle_close': last_candle_close,
            'processed_candles_count': len(self.processed_candles),
            'is_active': not self.shutdown_flag.is_set(),
            'trading_callback_enabled': self.trading_signal_callback is not None
        }
