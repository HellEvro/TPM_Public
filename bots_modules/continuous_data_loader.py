"""
🔄 НЕПРЕРЫВНЫЙ ЗАГРУЗЧИК ДАННЫХ
Независимый воркер который работает по кругу, постоянно обновляя все данные
Все остальные сервисы просто читают актуальные данные из глобального хранилища
"""

import threading
import time
from datetime import datetime
import logging

logger = logging.getLogger('BotsService')
# Добавляем префикс для легкого поиска в логах
class PrefixedLogger:
    def __init__(self, logger, prefix):
        self.logger = logger
        self.prefix = prefix
    
    def info(self, msg):
        self.logger.info(f"{self.prefix} {msg}")
    
    def warning(self, msg):
        self.logger.warning(f"{self.prefix} {msg}")
    
    def error(self, msg):
        self.logger.error(f"{self.prefix} {msg}")
    
    def debug(self, msg):
        self.logger.debug(f"{self.prefix} {msg}")

logger = PrefixedLogger(logger, "🔄 [CONTINUOUS]")

class ContinuousDataLoader:
    def __init__(self, exchange_obj=None, update_interval=180):
        """
        Args:
            exchange_obj: Объект биржи
            update_interval: Интервал обновления в секундах (по умолчанию 180 = 3 минуты)
        """
        self.exchange = exchange_obj
        self.update_interval = update_interval
        self.is_running = False
        self.thread = None
        self.last_update_time = None
        self.update_count = 0
        self.error_count = 0
        
    def start(self):
        """🚀 Запускает воркер в отдельном потоке"""
        if self.is_running:
            logger.warning("[CONTINUOUS] ⚠️ Воркер уже запущен")
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._continuous_loop, daemon=True)
        self.thread.start()
        logger.info(f"[CONTINUOUS] 🚀 Воркер запущен (интервал: {self.update_interval}с)")
        
    def stop(self):
        """🛑 Останавливает воркер"""
        if not self.is_running:
            return
            
        logger.info("[CONTINUOUS] 🛑 Останавливаем воркер...")
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("[CONTINUOUS] ✅ Воркер остановлен")
        
    def _continuous_loop(self):
        """🔄 Основной цикл обновления данных"""
        logger.info("[CONTINUOUS] 🔄 Входим в непрерывный цикл обновления...")
        
        # ⚡ ТРЕЙСИНГ ОТКЛЮЧЕН - проблема решена (deadlock на bots_data_lock)
        # try:
        #     from trace_debug import enable_trace
        #     enable_trace()
        #     logger.info("🔍 [CONTINUOUS] Трейсинг включен для диагностики зависаний")
        # except Exception as e:
        #     logger.warning(f"⚠️ [CONTINUOUS] Не удалось включить трейсинг: {e}")
        
        # Небольшая задержка перед первым обновлением (даем системе запуститься)
        time.sleep(5)
        
        # Импортируем shutdown_flag для корректной остановки
        from bots_modules.imports_and_globals import shutdown_flag
        
        while self.is_running and not shutdown_flag.is_set():
            try:
                cycle_start = time.time()
                self.update_count += 1
                
                # ✅ КРИТИЧНО: ПОЛНАЯ БЛОКИРОВКА UI ОБНОВЛЕНИЙ с самого начала цикла
                from bots_modules.imports_and_globals import coins_rsi_data
                coins_rsi_data['ui_update_paused'] = True
                coins_rsi_data['processing_cycle'] = True  # ✅ Дополнительный флаг для полной блокировки
                logger.info("[CONTINUOUS] 🎨 UI обновления ПОЛНОСТЬЮ заблокированы - начинаем обработку данных")
                
                logger.info("=" * 80)
                logger.info(f"[CONTINUOUS] 🔄 РАУНД #{self.update_count} НАЧАТ")
                logger.info(f"[CONTINUOUS] 🕐 Время: {datetime.now().strftime('%H:%M:%S')}")
                logger.info("=" * 80)
                
                # ✅ Этап 1: Загружаем свечи всех монет (15-20 сек)
                success = self._load_candles()
                if not success:
                    logger.error("[CONTINUOUS] ❌ Не удалось загрузить свечи, пропускаем раунд")
                    self.error_count += 1
                    time.sleep(30)  # Пауза перед следующей попыткой
                    continue
                
                # ✅ Этап 2: Рассчитываем RSI для всех монет (30-40 сек)
                success = self._calculate_rsi()
                if not success:
                    logger.error("[CONTINUOUS] ❌ Не удалось рассчитать RSI, пропускаем раунд")
                    self.error_count += 1
                    time.sleep(30)
                    continue
                
                # ✅ Этап 3: Рассчитываем зрелость (только для незрелых монет) (10-20 сек)
                self._calculate_maturity()
                
                # ✅ Этап 4: Определяем тренд для сигнальных монет (RSI ≤29 или ≥71) (5-10 сек)
                self._analyze_trends()
                
                # ✅ Этап 5: Обрабатываем лонг/шорт монеты фильтрами (5 сек)
                filtered_coins = self._process_filters()
                
                # ✅ Этап 6: Передаем отфильтрованные монеты автоботу
                self._set_filtered_coins_for_autobot(filtered_coins)
                
                cycle_duration = time.time() - cycle_start
                self.last_update_time = datetime.now()
                
                logger.info("=" * 80)
                logger.info(f"[CONTINUOUS] ✅ РАУНД #{self.update_count} ЗАВЕРШЕН")
                logger.info(f"[CONTINUOUS] ⏱️ Длительность: {cycle_duration:.1f}с")
                logger.info(f"[CONTINUOUS] 📊 Статистика: обновлений={self.update_count}, ошибок={self.error_count}")
                logger.info(f"[CONTINUOUS] 🎯 Отфильтровано монет: {len(filtered_coins)}")
                logger.info("=" * 80)
                
                # ✅ ПОЛНОСТЬЮ СНИМАЕМ БЛОКИРОВКУ UI И УВЕЛИЧИВАЕМ ВЕРСИЮ ДАННЫХ - все этапы завершены!
                # Теперь UI может безопасно обновляться с финальными данными
                from bots_modules.imports_and_globals import coins_rsi_data
                coins_rsi_data['ui_update_paused'] = False
                coins_rsi_data['processing_cycle'] = False  # ✅ Снимаем дополнительный флаг
                coins_rsi_data['data_version'] += 1  # ✅ Увеличиваем версию данных
                logger.info(f"[CONTINUOUS] 🎨 UI обновления ПОЛНОСТЬЮ разблокированы - все этапы завершены (версия данных: {coins_rsi_data['data_version']})")
                
                # 🚀 БЕЗ ПАУЗ: Раунды идут максимально быстро один за другим!
                # Чем быстрее железо - тем быстрее обновляются данные
                logger.info(f"[CONTINUOUS] 🚀 Сразу запускаем следующий раунд...")
                
                # Минимальная пауза 1 секунда для стабильности (с проверкой shutdown)
                if shutdown_flag.wait(1):  # Прерываемый sleep
                    break
                    
            except Exception as e:
                logger.error(f"[CONTINUOUS] ❌ Ошибка в цикле обновления: {e}")
                self.error_count += 1
                
                # ✅ ПОЛНОСТЬЮ СНИМАЕМ БЛОКИРОВКУ UI даже при ошибке - чтобы UI не завис
                from bots_modules.imports_and_globals import coins_rsi_data
                coins_rsi_data['ui_update_paused'] = False
                coins_rsi_data['processing_cycle'] = False  # ✅ Снимаем дополнительный флаг даже при ошибке
                coins_rsi_data['data_version'] += 1  # ✅ Увеличиваем версию даже при ошибке
                logger.info(f"[CONTINUOUS] 🎨 UI обновления ПОЛНОСТЬЮ разблокированы (после ошибки, версия данных: {coins_rsi_data['data_version']})")
                
                time.sleep(30)  # Пауза перед следующей попыткой
                
        logger.info("[CONTINUOUS] 🏁 Выход из непрерывного цикла")
    
    def _load_candles(self):
        """📦 Загружает свечи всех монет"""
        try:
            logger.info("[CONTINUOUS] 📦 Этап 1/6: Загружаем свечи...")
            start = time.time()
            
            logger.info("[CONTINUOUS] 🔄 Вызываем load_all_coins_candles_fast()...")
            from bots_modules.filters import load_all_coins_candles_fast
            success = load_all_coins_candles_fast()
            logger.info(f"[CONTINUOUS] 📊 load_all_coins_candles_fast() вернула: {success}")
            
            duration = time.time() - start
            if success:
                logger.info(f"[CONTINUOUS] ✅ Свечи загружены за {duration:.1f}с")
                return True
            else:
                logger.error(f"[CONTINUOUS] ❌ Не удалось загрузить свечи")
                return False
                
        except Exception as e:
            logger.error(f"[CONTINUOUS] ❌ Ошибка загрузки свечей: {e}")
            import traceback
            logger.error(f"[CONTINUOUS] ❌ Traceback: {traceback.format_exc()}")
            return False
    
    def _calculate_rsi(self):
        """📊 Рассчитывает RSI для всех монет"""
        try:
            logger.info("[CONTINUOUS] 📊 Этап 2/6: Рассчитываем RSI...")
            start = time.time()
            
            # ⚡ ТРЕЙСИНГ ОТКЛЮЧЕН - проблема решена (deadlock на bots_data_lock)
            # try:
            #     from trace_debug import enable_trace
            #     enable_trace()
            #     logger.info("🔍 [CONTINUOUS] Трейсинг включен для load_all_coins_rsi()")
            # except Exception as trace_error:
            #     logger.warning(f"⚠️ [CONTINUOUS] Не удалось включить трейсинг: {trace_error}")
            
            # ⚡ УПРОЩЕНИЕ: Запускаем напрямую без threading timeout
            # Threading timeout может вызывать проблемы в Windows
            logger.info("[CONTINUOUS] 🔄 Вызываем load_all_coins_rsi()...")
            from bots_modules.filters import load_all_coins_rsi
            success = load_all_coins_rsi()
            logger.info(f"[CONTINUOUS] 📊 load_all_coins_rsi() вернула: {success}")
            
            duration = time.time() - start
            if success:
                logger.info(f"[CONTINUOUS] ✅ RSI рассчитан за {duration:.1f}с")
                return True
            else:
                logger.error(f"[CONTINUOUS] ❌ Не удалось рассчитать RSI")
                return False
                
        except Exception as e:
            logger.error(f"[CONTINUOUS] ❌ Ошибка расчета RSI: {e}")
            import traceback
            logger.error(f"[CONTINUOUS] ❌ Traceback: {traceback.format_exc()}")
            return False
    
    def _calculate_maturity(self):
        """🧮 Рассчитывает зрелость монет (только незрелые)"""
        try:
            logger.info("[CONTINUOUS] 🧮 Этап 3/6: Рассчитываем зрелость...")
            start = time.time()
            
            # Простой таймаут через threading (работает в Windows)
            from threading import Thread
            
            result = [None]
            exception = [None]
            
            def run_maturity():
                try:
                    from bots_modules.maturity import calculate_all_coins_maturity
                    calculate_all_coins_maturity()
                    result[0] = True
                except Exception as e:
                    exception[0] = e
            
            # Запускаем в отдельном потоке
            thread = Thread(target=run_maturity)
            thread.daemon = True
            thread.start()
            
            # Ждем максимум 60 секунд
            thread.join(timeout=60)
            
            if thread.is_alive():
                logger.error("[CONTINUOUS] ⚠️ Таймаут расчета зрелости (60с)")
                return
            
            if exception[0]:
                raise exception[0]
                
            duration = time.time() - start
            logger.info(f"[CONTINUOUS] ✅ Зрелость рассчитана за {duration:.1f}с")
            
        except Exception as e:
            logger.error(f"[CONTINUOUS] ⚠️ Ошибка расчета зрелости: {e}")
            # Не критично, продолжаем
    
    def _analyze_trends(self):
        """📈 Определяет тренд для сигнальных монет"""
        try:
            logger.info("[CONTINUOUS] 📈 Этап 4/6: Анализируем тренды...")
            start = time.time()
            
            from bots_modules.filters import analyze_trends_for_signal_coins
            analyze_trends_for_signal_coins()
            
            duration = time.time() - start
            logger.info(f"[CONTINUOUS] ✅ Тренды проанализированы за {duration:.1f}с")
            
        except Exception as e:
            logger.error(f"[CONTINUOUS] ⚠️ Ошибка анализа трендов: {e}")
            # Не критично, продолжаем
    
    def _process_filters(self):
        """🔍 Обрабатывает лонг/шорт монеты фильтрами"""
        try:
            logger.info("[CONTINUOUS] 🔍 Этап 5/6: Обрабатываем фильтрами...")
            start = time.time()
            
            from bots_modules.filters import process_long_short_coins_with_filters
            filtered_coins = process_long_short_coins_with_filters()
            
            duration = time.time() - start
            logger.info(f"[CONTINUOUS] ✅ Фильтры обработаны за {duration:.1f}с ({len(filtered_coins)} монет)")
            return filtered_coins
            
        except Exception as e:
            logger.error(f"[CONTINUOUS] ⚠️ Ошибка обработки фильтрами: {e}")
            return []
    
    def _set_filtered_coins_for_autobot(self, filtered_coins):
        """✅ Передает отфильтрованные монеты автоботу"""
        try:
            logger.info("[CONTINUOUS] ✅ Этап 6/6: Передаем монеты автоботу...")
            start = time.time()
            
            from bots_modules.filters import set_filtered_coins_for_autobot
            set_filtered_coins_for_autobot(filtered_coins)
            
            duration = time.time() - start
            logger.info(f"[CONTINUOUS] ✅ Монеты переданы за {duration:.3f}с")
            
        except Exception as e:
            logger.error(f"[CONTINUOUS] ⚠️ Ошибка передачи монет автоботу: {e}")

    def get_status(self):
        """📊 Возвращает статус воркера"""
        return {
            'is_running': self.is_running,
            'update_count': self.update_count,
            'error_count': self.error_count,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'update_interval': self.update_interval
        }

# Глобальный экземпляр воркера
_continuous_loader = None

def start_continuous_loader(exchange_obj=None, update_interval=180):
    """🚀 Запускает непрерывный загрузчик данных"""
    global _continuous_loader
    
    if _continuous_loader and _continuous_loader.is_running:
        logger.warning("[CONTINUOUS] ⚠️ Загрузчик уже запущен")
        return _continuous_loader
    
    _continuous_loader = ContinuousDataLoader(exchange_obj, update_interval)
    _continuous_loader.start()
    return _continuous_loader

def stop_continuous_loader():
    """🛑 Останавливает непрерывный загрузчик данных"""
    global _continuous_loader
    
    if _continuous_loader:
        _continuous_loader.stop()
        _continuous_loader = None

def get_continuous_loader():
    """📊 Возвращает экземпляр загрузчика"""
    return _continuous_loader

