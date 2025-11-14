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

logger = PrefixedLogger(logger, "🔄")

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
            logger.warning("⚠️ Воркер уже запущен")
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._continuous_loop, daemon=True)
        self.thread.start()
        logger.info(f"Воркер запущен (интервал: {self.update_interval}с)")
        
    def stop(self):
        """🛑 Останавливает воркер"""
        if not self.is_running:
            return
            
        logger.info("🛑 Останавливаем воркер...")
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("✅ Воркер остановлен")
        
    def _continuous_loop(self):
        """🔄 Основной цикл обновления данных"""
        logger.info("Входим в непрерывный цикл обновления...")
        
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
                
                # ✅ ОПТИМИЗАЦИЯ: НЕ блокируем UI обновления - загрузка свечей и RSI теперь неблокирующие
                from bots_modules.imports_and_globals import coins_rsi_data
                coins_rsi_data['processing_cycle'] = True  # Только флаг обработки
                logger.info("Начинаем обработку данных (неблокирующий режим)")
                
                logger.info("=" * 80)
                logger.info(f"РАУНД #{self.update_count} НАЧАТ")
                logger.info(f"🕐 Время: {datetime.now().strftime('%H:%M:%S')}")
                logger.info("=" * 80)
                
                # ✅ Этап 1: Загружаем свечи всех монет (15-20 сек) - БЛОКИРУЮЩИЙ
                success = self._load_candles()
                if not success:
                    logger.error("❌ Не удалось загрузить свечи, пропускаем раунд")
                    self.error_count += 1
                    time.sleep(30)  # Пауза перед следующей попыткой
                    continue
                
                # ✅ Этап 2: Рассчитываем RSI для всех монет (30-40 сек) - БЛОКИРУЮЩИЙ
                success = self._calculate_rsi()
                if not success:
                    logger.error("❌ Не удалось рассчитать RSI, пропускаем раунд")
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
                logger.info(f"✅ РАУНД #{self.update_count} ЗАВЕРШЕН")
                logger.info(f"⏱️ Длительность: {cycle_duration:.1f}с")
                logger.info(f"📊 Статистика: обновлений={self.update_count}, ошибок={self.error_count}")
                logger.info(f"🎯 Отфильтровано монет: {len(filtered_coins)}")
                logger.info("=" * 80)
                
                # ✅ ЗАВЕРШАЕМ ОБРАБОТКУ - увеличиваем версию данных
                from bots_modules.imports_and_globals import coins_rsi_data
                coins_rsi_data['processing_cycle'] = False  # Снимаем флаг обработки
                coins_rsi_data['data_version'] += 1  # Увеличиваем версию данных
                logger.info(f"✅ Обработка завершена (версия данных: {coins_rsi_data['data_version']})")
                
                # 🚀 БЕЗ ПАУЗ: Раунды идут максимально быстро один за другим!
                # Чем быстрее железо - тем быстрее обновляются данные
                logger.info(f"🚀 Сразу запускаем следующий раунд...")
                
                # Минимальная пауза 1 секунда для стабильности (с проверкой shutdown)
                if shutdown_flag.wait(1):  # Прерываемый sleep
                    break
                    
            except Exception as e:
                logger.error(f"❌ Ошибка в цикле обновления: {e}")
                self.error_count += 1
                
                # ✅ ЗАВЕРШАЕМ ОБРАБОТКУ даже при ошибке
                from bots_modules.imports_and_globals import coins_rsi_data
                coins_rsi_data['processing_cycle'] = False  # Снимаем флаг обработки даже при ошибке
                coins_rsi_data['data_version'] += 1  # Увеличиваем версию даже при ошибке
                logger.info(f"✅ Обработка завершена (после ошибки, версия данных: {coins_rsi_data['data_version']})")
                
                time.sleep(30)  # Пауза перед следующей попыткой
                
        logger.info("🏁 Выход из непрерывного цикла")
    
    def _load_candles(self):
        """📦 Загружает свечи всех монет"""
        try:
            logger.info("📦 Этап 1/6: Загружаем свечи...")
            start = time.time()
            
            logger.info("Вызываем load_all_coins_candles_fast()...")
            from bots_modules.filters import load_all_coins_candles_fast
            success = load_all_coins_candles_fast()
            logger.info(f"📊 load_all_coins_candles_fast() вернула: {success}")
            
            duration = time.time() - start
            if success:
                logger.info(f"✅ Свечи загружены за {duration:.1f}с")
                return True
            else:
                logger.error(f"❌ Не удалось загрузить свечи")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки свечей: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
    
    def _load_candles_non_blocking(self):
        """📦 Загружает свечи всех монет в отдельном потоке (НЕБЛОКИРУЮЩИЙ)"""
        try:
            logger.info("📦 Этап 1/6: Загружаем свечи (неблокирующий)...")
            start = time.time()
            
            # Проверяем, есть ли уже свечи в кэше
            from bots_modules.imports_and_globals import coins_rsi_data
            if 'candles_cache' in coins_rsi_data and coins_rsi_data['candles_cache']:
                last_update = coins_rsi_data.get('last_candles_update', '')
                if last_update:
                    from datetime import datetime, timedelta
                    try:
                        last_update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        time_diff = datetime.now() - last_update_time.replace(tzinfo=None)
                        if time_diff.total_seconds() < 300:  # Если свечи обновлялись менее 5 минут назад
                            logger.info("✅ Используем свежие свечи из кэша")
                            return True
                    except:
                        pass
            
            # Запускаем загрузку в отдельном потоке
            import threading
            def load_candles_thread():
                try:
                    logger.info("Запускаем load_all_coins_candles_fast() в отдельном потоке...")
                    from bots_modules.filters import load_all_coins_candles_fast
                    success = load_all_coins_candles_fast()
                    logger.info(f"📊 load_all_coins_candles_fast() завершена: {success}")
                except Exception as e:
                    logger.error(f"❌ Ошибка в потоке загрузки свечей: {e}")
            
            # Запускаем поток
            candles_thread = threading.Thread(target=load_candles_thread, daemon=True)
            candles_thread.start()
            
            # Ждем максимум 2 секунды для инициализации
            candles_thread.join(timeout=2)
            
            duration = time.time() - start
            logger.info(f"✅ Загрузка свечей запущена в фоне за {duration:.1f}с")
            return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка запуска загрузки свечей: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
    
    def _calculate_rsi(self):
        """📊 Рассчитывает RSI для всех монет"""
        try:
            logger.info("📊 Этап 2/6: Рассчитываем RSI...")
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
            logger.info("Вызываем load_all_coins_rsi()...")
            from bots_modules.filters import load_all_coins_rsi
            success = load_all_coins_rsi()
            logger.info(f"📊 load_all_coins_rsi() вернула: {success}")
            
            duration = time.time() - start
            if success:
                logger.info(f"✅ RSI рассчитан за {duration:.1f}с")
                return True
            else:
                logger.error(f"❌ Не удалось рассчитать RSI")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка расчета RSI: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
    
    def _calculate_rsi_non_blocking(self):
        """📊 Рассчитывает RSI для всех монет в отдельном потоке (НЕБЛОКИРУЮЩИЙ)"""
        try:
            logger.info("📊 Этап 2/6: Рассчитываем RSI (неблокирующий)...")
            start = time.time()
            
            # Проверяем, есть ли уже RSI данные в кэше
            from bots_modules.imports_and_globals import coins_rsi_data
            if 'rsi_data' in coins_rsi_data and coins_rsi_data['rsi_data']:
                last_update = coins_rsi_data.get('last_rsi_update', '')
                if last_update:
                    from datetime import datetime
                    try:
                        last_update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        time_diff = datetime.now() - last_update_time.replace(tzinfo=None)
                        if time_diff.total_seconds() < 600:  # Если RSI обновлялся менее 10 минут назад
                            logger.info("✅ Используем свежие RSI данные из кэша")
                            return True
                    except:
                        pass
            
            # Запускаем расчет в отдельном потоке
            import threading
            def calculate_rsi_thread():
                try:
                    logger.info("Запускаем load_all_coins_rsi() в отдельном потоке...")
                    from bots_modules.filters import load_all_coins_rsi
                    success = load_all_coins_rsi()
                    logger.info(f"📊 load_all_coins_rsi() завершена: {success}")
                except Exception as e:
                    logger.error(f"❌ Ошибка в потоке расчета RSI: {e}")
            
            # Запускаем поток
            rsi_thread = threading.Thread(target=calculate_rsi_thread, daemon=True)
            rsi_thread.start()
            
            # Ждем максимум 3 секунды для инициализации
            rsi_thread.join(timeout=3)
            
            duration = time.time() - start
            logger.info(f"✅ Расчет RSI запущен в фоне за {duration:.1f}с")
            return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка запуска расчета RSI: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
    
    def _calculate_maturity(self):
        """🧮 Рассчитывает зрелость монет (только незрелые)"""
        try:
            logger.info("🧮 Этап 3/6: Рассчитываем зрелость...")
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
                logger.error("⚠️ Таймаут расчета зрелости (60с)")
                return
            
            if exception[0]:
                raise exception[0]
                
            duration = time.time() - start
            logger.info(f"✅ Зрелость рассчитана за {duration:.1f}с")
            
        except Exception as e:
            logger.error(f"⚠️ Ошибка расчета зрелости: {e}")
            # Не критично, продолжаем
    
    def _analyze_trends(self):
        """📈 Определяет тренд для сигнальных монет"""
        try:
            logger.info("📈 Этап 4/6: Анализируем тренды...")
            start = time.time()
            
            from bots_modules.filters import analyze_trends_for_signal_coins
            analyze_trends_for_signal_coins()
            
            duration = time.time() - start
            logger.info(f"✅ Тренды проанализированы за {duration:.1f}с")
            
        except Exception as e:
            logger.error(f"⚠️ Ошибка анализа трендов: {e}")
            # Не критично, продолжаем
    
    def _process_filters(self):
        """🔍 Обрабатывает лонг/шорт монеты фильтрами"""
        try:
            logger.info("🔍 Этап 5/6: Обрабатываем фильтрами...")
            start = time.time()
            
            from bots_modules.filters import process_long_short_coins_with_filters
            filtered_coins = process_long_short_coins_with_filters()
            
            duration = time.time() - start
            logger.info(f"✅ Фильтры обработаны за {duration:.1f}с ({len(filtered_coins)} монет)")
            return filtered_coins
            
        except Exception as e:
            logger.error(f"⚠️ Ошибка обработки фильтрами: {e}")
            return []
    
    def _set_filtered_coins_for_autobot(self, filtered_coins):
        """✅ Передает отфильтрованные монеты автоботу"""
        try:
            logger.info("✅ Этап 6/6: Передаем монеты автоботу...")
            start = time.time()
            
            from bots_modules.filters import set_filtered_coins_for_autobot
            set_filtered_coins_for_autobot(filtered_coins)
            
            duration = time.time() - start
            logger.info(f"✅ Монеты переданы за {duration:.3f}с")
            
        except Exception as e:
            logger.error(f"⚠️ Ошибка передачи монет автоботу: {e}")

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
        logger.warning("⚠️ Загрузчик уже запущен")
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

