"""
Умный менеджер обновления RSI
Обновляет RSI регулярно, но торговые сигналы только при закрытии свечи текущего таймфрейма
"""

import time
import threading
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Callable
from bot_engine.config_loader import SystemConfig, get_current_timeframe

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

        # ✅ КРИТИЧНО: Получаем текущий таймфрейм из конфига
        try:
            self.current_timeframe = get_current_timeframe()
        except Exception:
            self.current_timeframe = '1m'  # Fallback в соответствии с дефолтом SYSTEM_TIMEFRAME

        logger.info(f"[SMART_RSI] 🧠 Умный менеджер RSI инициализирован")
        logger.info(f"[SMART_RSI] 📊 Плановое обновление: каждые {self.monitoring_interval//60} минут")
        logger.info(f"[SMART_RSI] 🎯 Торговые сигналы: только при обновлении после закрытия свечи {self.current_timeframe}")
        logger.info(f"[SMART_RSI] ⚡ Оптимизация: нет частых проверок API, только плановые обновления")

    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """Возвращает количество секунд в таймфрейме"""
        timeframe_map = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600,
            '8h': 28800, '12h': 43200, '1d': 86400, '3d': 259200,
            '1w': 604800, '1M': 2592000  # ~30 дней
        }
        return timeframe_map.get(timeframe, 60)  # По умолчанию 1m

    def get_next_candle_close(self) -> int:
        """Возвращает timestamp следующего закрытия свечи для текущего таймфрейма"""
        current_time = int(time.time())
        current_dt = datetime.fromtimestamp(current_time, tz=timezone.utc)
        timeframe_seconds = self._get_timeframe_seconds(self.current_timeframe)

        # Для минутных таймфреймов (1m, 3m, 5m, 15m, 30m)
        if self.current_timeframe.endswith('m'):
            minutes = int(self.current_timeframe[:-1])
            # Округляем до следующей минуты, кратной интервалу
            current_minute = current_dt.minute
            next_minute = ((current_minute // minutes) + 1) * minutes
            if next_minute >= 60:
                next_dt = current_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            else:
                next_dt = current_dt.replace(minute=next_minute, second=0, microsecond=0)
            return int(next_dt.timestamp())

        # Для часовых таймфреймов (1h, 2h, 4h, 6h, 8h, 12h)
        elif self.current_timeframe.endswith('h'):
            hours = int(self.current_timeframe[:-1])
            current_hour = current_dt.hour
            # Определяем следующие часы закрытия
            next_closes = list(range(0, 24, hours))
            next_close_hour = None

            for close_hour in next_closes:
                if close_hour > current_hour or (close_hour == current_hour and current_dt.minute > 0):
                    next_close_hour = close_hour
                    break

            if next_close_hour is None:
                # Если все времена закрытия в текущем дне прошли, берем 00:00 следующего дня
                next_close_hour = 24

            next_dt = current_dt.replace(
                hour=next_close_hour % 24,
                minute=0,
                second=0,
                microsecond=0
            )

            if next_close_hour == 24:
                next_dt = next_dt + timedelta(days=1)

            return int(next_dt.timestamp())

        # Для дневных таймфреймов (1d, 3d)
        elif self.current_timeframe.endswith('d'):
            days = int(self.current_timeframe[:-1])
            # Закрывается в полночь UTC
            next_dt = current_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            if current_dt.hour > 0 or current_dt.minute > 0 or current_dt.second > 0:
                next_dt = next_dt + timedelta(days=days)
            return int(next_dt.timestamp())

        # Для недельных таймфреймов (1w)
        elif self.current_timeframe.endswith('w'):
            # Закрывается в понедельник в 00:00 UTC
            days_until_monday = (7 - current_dt.weekday()) % 7
            if days_until_monday == 0 and (current_dt.hour > 0 or current_dt.minute > 0):
                days_until_monday = 7
            next_dt = current_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday)
            return int(next_dt.timestamp())

        # Для месячных таймфреймов (1M)
        elif self.current_timeframe.endswith('M'):
            # Закрывается в первый день месяца в 00:00 UTC
            next_dt = current_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if current_dt.day > 1 or current_dt.hour > 0:
                # Переходим на следующий месяц
                if current_dt.month == 12:
                    next_dt = next_dt.replace(year=current_dt.year + 1, month=1)
                else:
                    next_dt = next_dt.replace(month=current_dt.month + 1)
            return int(next_dt.timestamp())

        # Fallback для неизвестных таймфреймов - используем интервал в секундах
        return current_time + timeframe_seconds

    def get_time_to_candle_close(self) -> int:
        """Возвращает время в секундах до закрытия текущей свечи"""
        next_close = self.get_next_candle_close()
        current_time = int(time.time())
        return max(0, next_close - current_time)

    def get_last_candle_close(self) -> int:
        """Возвращает timestamp последнего закрытия свечи"""
        current_time = int(time.time())
        timeframe_seconds = self._get_timeframe_seconds(self.current_timeframe)
        next_close = self.get_next_candle_close()
        return next_close - timeframe_seconds

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
        ВСЕГДА обрабатываем сигналы - убираем глупое условие закрытия свечи!
        """
        current_time = int(time.time())
        last_candle_close = self.get_last_candle_close()

        # ВСЕГДА обрабатываем торговые сигналы!
        # Убираем глупое условие ожидания закрытия свечи
        return True, f"обработка сигналов включена всегда", last_candle_close

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

            # ⚡ БЫСТРАЯ ЗАГРУЗКА: Сначала грузим ТОЛЬКО свечи
            logger.info(f"[SMART_RSI] 🚀 Быстрая загрузка свечей...")
            from bots_modules.filters import load_all_coins_candles_fast
            if load_all_coins_candles_fast():
                logger.info(f"[SMART_RSI] ✅ Свечи загружены! Теперь локальные расчеты...")
                # Потом вызываем полную загрузку с расчетами (она будет использовать кэш свечей)
                self.rsi_update_callback()
            else:
                logger.error(f"[SMART_RSI] ❌ Не удалось загрузить свечи")

            time_to_close = self.get_time_to_candle_close()
            hours = time_to_close // 3600
            minutes = (time_to_close % 3600) // 60

            if hours > 0:
                time_str = f"{hours}ч {minutes}м"
            else:
                time_str = f"{minutes}м"

            logger.info(f"[SMART_RSI] ✅ RSI данные обновлены | До закрытия свечи {self.current_timeframe}: {time_str}")

            # Проверяем, нужно ли активировать торговые сигналы
            should_trade, trade_reason, candle_timestamp = self.should_process_trading_signals_after_update()
            if should_trade:
                logger.info("=" * 80)
                logger.info(f"[SMART_RSI] 🎯 ОБНАРУЖЕНО ЗАКРЫТИЕ СВЕЧИ {self.current_timeframe}! ({trade_reason})")
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
                                pass

        except Exception as e:
            logger.error(f"[SMART_RSI] ❌ Ошибка обновления RSI: {e}")

    def run_smart_worker(self):
        """Основной цикл умного обновления RSI и проверки торговых сигналов"""
        # ⚡ АКТИВИРУЕМ ТРЕЙСИНГ для этого потока (если включен)
        if SystemConfig.ENABLE_CODE_TRACING:
            try:
                from trace_debug import enable_trace
                enable_trace()
                logger.info("[SMART_RSI] 🔍 Трейсинг активирован в потоке Smart RSI")
            except:
                pass

        logger.info("=" * 80)
        logger.info("[SMART_RSI] 🚀 ЗАПУСК ОПТИМИЗИРОВАННОЙ СИСТЕМЫ RSI")
        logger.info("[SMART_RSI] 📊 Режим: Плановое обновление каждые 60 минут")
        logger.info(f"[SMART_RSI] 🎯 Торговые сигналы: автоматически при обновлении после закрытия свечи {self.current_timeframe}")
        logger.info("[SMART_RSI] ⚡ Нет частых проверок API - только эффективные плановые обновления")
        logger.info("=" * 80)

        # Первое обновление сразу
        logger.info("[SMART_RSI] 📡 Начинаем первое обновление RSI...")
        self.update_rsi_data()
        logger.info("[SMART_RSI] ✅ Первое обновление RSI завершено")

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
        last_candle_close = self.get_last_candle_close()

        return {
            'monitoring_interval': self.monitoring_interval,
            'current_timeframe': self.current_timeframe,
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
