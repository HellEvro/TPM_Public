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
                pass

logger = PrefixedLogger(logger, "🔄")

# Таймаут этапа расчёта зрелости (сек). При большом числе монет и ТФ 1m 60с может не хватать.
MATURITY_CALCULATION_TIMEOUT = 120

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

        logger.warning("🛑 Останавливаем воркер...")
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.warning("✅ Воркер остановлен")

    def _continuous_loop(self):
        """🔄 Основной цикл обновления данных"""
        logger.info("🔄 Поток непрерывного загрузчика ЗАПУЩЕН (через 5 сек — первый раунд)")

        # ⚡ ТРЕЙСИНГ ОТКЛЮЧЕН - проблема решена (deadlock на bots_data_lock)
        # try:
        #     from trace_debug import enable_trace
        #     enable_trace()
        #     logger.info("🔍 [CONTINUOUS] Трейсинг включен для диагностики зависаний")
        # except Exception as e:
        #     logger.warning(f"⚠️ [CONTINUOUS] Не удалось включить трейсинг: {e}")

        # Получаем текущий таймфрейм при старте цикла
        try:
            from bot_engine.config_loader import get_current_timeframe
            startup_timeframe = get_current_timeframe()
            logger.info(f"⏱️ [CONTINUOUS] Таймфрейм при старте загрузчика: {startup_timeframe}")
        except Exception as tf_err:
            logger.warning(f"⚠️ [CONTINUOUS] Не удалось получить таймфрейм при старте: {tf_err}")

        # Небольшая задержка перед первым обновлением (даем системе запуститься)
        time.sleep(5)
        logger.info("🔄 Начинаем первый раунд обновления данных...")

        # Импортируем shutdown_flag для корректной остановки
        from bots_modules.imports_and_globals import shutdown_flag

        while self.is_running and not shutdown_flag.is_set():
            try:
                cycle_start = time.time()
                self.update_count += 1

                # ✅ Блокировка только для первой загрузки: автобот и мониторинг позиций ждут first_round_complete;
                # после первой загрузки они не блокируются — читают актуальные данные из кэша. Здесь блокируется только поток загрузчика.
                from bots_modules.imports_and_globals import coins_rsi_data
                coins_rsi_data['processing_cycle'] = True  # Флаг для UI (опционально)
                logger.info("Начинаем обработку данных")

                # Получаем текущий таймфрейм для логирования
                try:
                    from bot_engine.config_loader import get_current_timeframe, TIMEFRAME
                    current_timeframe = get_current_timeframe()
                except Exception:
                    current_timeframe = TIMEFRAME

                logger.info("=" * 80)
                logger.info(f"РАУНД #{self.update_count} НАЧАТ")
                logger.info(f"🕐 Время: {datetime.now().strftime('%H:%M:%S')}")
                logger.info(f"⏱️ Таймфрейм: {current_timeframe}")
                logger.info("=" * 80)

                # ✅ Когда автобот ВЫКЛЮЧЕН: не ищем новые сделки; этапы 3–6 пропускаем. Но свечи и RSI — ВСЕГДА (для UI).
                logger.info("🔄 [РАУНД] Получаем флаг автобота (lock)...")
                from bots_modules.imports_and_globals import bots_data, bots_data_lock, BOT_STATUS
                with bots_data_lock:
                    auto_bot_enabled = bots_data.get('auto_bot_config', {}).get('enabled', False)
                    active_bots_count = sum(
                        1 for b in (bots_data.get('bots') or {}).values()
                        if b.get('status') not in [BOT_STATUS.get('IDLE'), BOT_STATUS.get('PAUSED')]
                    )
                if not auto_bot_enabled and active_bots_count == 0:
                    logger.info("⏹️ Автобот выключен, активных ботов нет — загружаем только свечи и RSI для UI")
                logger.info("🔄 [РАУНД] Lock получен, запускаем этап 1 (свечи)...")

                # ✅ Предзаполнение списка монет для UI: пока первый раунд не завершён, список не пустой
                if not coins_rsi_data.get('coins') or len(coins_rsi_data.get('coins', {})) == 0:
                    self._seed_coins_placeholder()

                # ✅ Этап 1: Загрузка НОВЫХ свечей с биржи. Без свечей работа системы бессмысленна.
                success_candles = self._load_candles()
                if not success_candles:
                    logger.error("КРИТИЧНО: загрузка свечей с биржи не удалась. Без свечей RSI не считается. Проверьте биржу, сеть, rate limit.")
                    self.error_count += 1
                    time.sleep(30)
                    continue

                # ✅ Этап 2: Расчёт RSI по загруженным свечам
                success_rsi = self._calculate_rsi()
                if not success_rsi:
                    logger.error("КРИТИЧНО: расчёт RSI не выполнен. Данные для торговли отсутствуют. Проверьте логи, биржу и конфиг.")
                    self.error_count += 1
                    time.sleep(30)
                    continue

                # ✅ КРИТИЧНО: Первая загрузка (свечи + RSI) завершена — только до этого момента другие системы ждут;
                # далее блокировка не используется: автобот и мониторинг уже работают по данным из кэша.
                if not coins_rsi_data.get('first_round_complete'):
                    coins_rsi_data['first_round_complete'] = True
                    logger.info("✅ ПЕРВАЯ ЗАГРУЗКА ЗАВЕРШЕНА: свечи + RSI готовы → запуск системы (автобот, мониторинг позиций)")

                # ✅ Этапы 3–6 только при включённом автоботе (поиск новых сделок)
                if auto_bot_enabled:
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

    def _seed_coins_placeholder(self):
        """Заполняет список монет заглушками (RSI=50, WAIT), чтобы UI не был пустым до первого раунда."""
        try:
            from bots_modules.imports_and_globals import get_exchange, coins_rsi_data
            from bot_engine.config_loader import get_current_timeframe, get_rsi_key, get_trend_key
            exch = get_exchange()
            if not exch:
                return
            try:
                tf = get_current_timeframe()
            except Exception:
                tf = '1m'
            rsi_key = get_rsi_key(tf)
            trend_key = get_trend_key(tf)
            pairs = exch.get_all_pairs()
            if not pairs or not isinstance(pairs, list):
                return
            now = datetime.now().isoformat()
            placeholders = {}
            for symbol in pairs:
                if not symbol or str(symbol).strip().upper() == 'ALL':
                    continue
                placeholders[symbol] = {
                    'symbol': symbol,
                    rsi_key: 50,
                    trend_key: 'NEUTRAL',
                    'rsi_zone': 'NEUTRAL',
                    'signal': 'WAIT',
                    'price': 0,
                    'change24h': 0,
                    'last_update': now,
                    'rsi': 50,
                    'trend': 'NEUTRAL',
                    'rsi6h': 50,
                    'trend6h': 'NEUTRAL',
                    'is_mature': True,
                    'has_existing_position': False,
                    'enhanced_rsi': {'enabled': False},
                }
            if placeholders:
                coins_rsi_data['coins'] = placeholders
                coins_rsi_data['total_coins'] = len(placeholders)
                coins_rsi_data['last_update'] = now
                logger.info(f"📋 Предзаполнено {len(placeholders)} монет для UI (RSI обновится после первого раунда)")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось предзаполнить список монет: {e}")

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

            # Проверяем, есть ли уже свечи в кэше с ПРАВИЛЬНЫМ таймфреймом
            from bots_modules.imports_and_globals import coins_rsi_data
            from bot_engine.config_loader import get_current_timeframe
            current_timeframe = get_current_timeframe()

            if 'candles_cache' in coins_rsi_data and coins_rsi_data['candles_cache']:
                # Проверяем таймфрейм первой монеты в кэше
                cache_sample = next(iter(coins_rsi_data['candles_cache'].values()), None)
                if cache_sample and cache_sample.get('timeframe') == current_timeframe:
                    last_update = coins_rsi_data.get('last_candles_update', '')
                    if last_update:
                        from datetime import datetime, timedelta
                        try:
                            last_update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                            time_diff = datetime.now() - last_update_time.replace(tzinfo=None)
                            if time_diff.total_seconds() < 300:  # Если свечи обновлялись менее 5 минут назад
                                logger.info(f"✅ Используем свежие свечи из кэша (таймфрейм: {current_timeframe})")
                                return True
                        except:
                            pass
                else:
                    # Таймфрейм не совпадает - очищаем кэш
                    logger.info(f"🗑️ Таймфрейм кэша не совпадает (кэш: {cache_sample.get('timeframe') if cache_sample else 'нет'}, текущий: {current_timeframe}), очищаем кэш")
                    coins_rsi_data['candles_cache'] = {}
                    coins_rsi_data['last_candles_update'] = None

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

            # Ждем до MATURITY_CALCULATION_TIMEOUT секунд
            thread.join(timeout=MATURITY_CALCULATION_TIMEOUT)

            if thread.is_alive():
                logger.error(f"⚠️ Таймаут расчета зрелости ({MATURITY_CALCULATION_TIMEOUT}с)")
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
            start = time.time()

            from bots_modules.filters import process_long_short_coins_with_filters
            filtered_coins = process_long_short_coins_with_filters()

            duration = time.time() - start
            pass
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
