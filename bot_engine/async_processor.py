"""
Асинхронный процессор для ботов
Обеспечивает параллельную обработку RSI данных, сигналов и ботов
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AsyncTask:
    id: str
    task_type: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None

class AsyncRSILoader:
    """Асинхронный загрузчик RSI данных"""
    
    def __init__(self, exchange, max_concurrent_requests: int = 10):
        self.exchange = exchange
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None
        logger.info(f"[ASYNC_RSI] AsyncRSILoader инициализирован с биржей: {type(exchange)}")
        
        if exchange is None:
            logger.warning("[ASYNC_RSI] ⚠️ Биржа равна None, будет использоваться fallback логика")
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def load_coin_rsi_async(self, symbol: str) -> Optional[Dict]:
        """Асинхронно загружает RSI данные для одной монеты"""
        async with self.semaphore:
            try:
                # Используем ThreadPoolExecutor для синхронных вызовов к бирже
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor, 
                        self._get_coin_rsi_data_sync, 
                        symbol
                    )
                return result
            except Exception as e:
                logger.error(f"[ASYNC_RSI] Ошибка загрузки RSI для {symbol}: {e}")
                return None
    
    def _get_coin_rsi_data_sync(self, symbol: str) -> Optional[Dict]:
        """Синхронная функция для получения RSI данных с проверкой maturity"""
        try:
            # Импортируем функции из основного модуля
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Вызываем существующую функцию get_coin_rsi_data с передачей биржи
            from bots import get_coin_rsi_data, is_coin_mature_stored, add_mature_coin_to_storage
            
            # Определяем какую биржу использовать
            exchange_to_use = self.exchange
            pass
            
            if not exchange_to_use:
                # Пытаемся получить глобальную переменную exchange
                try:
                    import bots
                    exchange_to_use = getattr(bots, 'exchange', None)
                    pass
                    
                    if exchange_to_use:
                        logger.info(f"[ASYNC_RSI] Используем глобальную биржу для {symbol}")
                        # Дополнительная проверка, что у биржи есть нужный метод
                        if not hasattr(exchange_to_use, 'get_chart_data'):
                            logger.error(f"[ASYNC_RSI] Глобальная биржа не имеет метода get_chart_data для {symbol}")
                            return None
                    else:
                        logger.error(f"[ASYNC_RSI] Ошибка получения данных для {symbol}: биржа не инициализирована")
                        return None
                except Exception as e:
                    logger.error(f"[ASYNC_RSI] Ошибка получения глобальной биржи для {symbol}: {e}")
                    return None
            else:
                pass
            
            # Получаем RSI данные, передавая биржу
            rsi_data = get_coin_rsi_data(symbol, exchange_to_use)
            if not rsi_data:
                return None
            
            # ❌ УДАЛЕНО: Неправильная логика добавления RSI данных в maturity storage
            # Зрелость монеты проверяется в get_coin_rsi_data() и check_coin_maturity_with_storage()
            # НЕ нужно добавлять монеты в maturity storage здесь - это делается только
            # когда монета ДЕЙСТВИТЕЛЬНО прошла проверку зрелости!
            
            return rsi_data
            
        except Exception as e:
            logger.error(f"[ASYNC_RSI] Синхронная ошибка для {symbol}: {e}")
            return None
    
    async def load_batch_rsi_async(self, symbols: List[str]) -> Dict[str, Dict]:
        """Асинхронно загружает RSI данные для пакета монет"""
        logger.info(f"[ASYNC_RSI] 🚀 Начинаем асинхронную загрузку {len(symbols)} монет")
        
        # ⚡ БАЛАНС: Семафор на 10 - стабильность + скорость
        semaphore = asyncio.Semaphore(10)  # БАЛАНС: 10 одновременных
        
        async def load_with_semaphore(symbol):
            async with semaphore:
                # Минимальная задержка для стабильности
                await asyncio.sleep(0.1)
                return await self.load_coin_rsi_async(symbol)
        
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(load_with_semaphore(symbol))  # С семафором и задержкой
            tasks.append((symbol, task))
        
        results = {}
        completed = 0
        failed = 0
        
        for symbol, task in tasks:
            try:
                result = await task
                if result:
                    results[symbol] = result
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"[ASYNC_RSI] Ошибка задачи для {symbol}: {e}")
                failed += 1
        
        logger.info(f"[ASYNC_RSI] ✅ Завершено: {completed} успешно, {failed} ошибок")
        
        # Сохраняем maturity данные пакетно в конце
        try:
            from bots import save_mature_coins_storage
            save_mature_coins_storage()
            pass
        except Exception as e:
            logger.warning(f"[ASYNC_RSI] Ошибка сохранения maturity данных: {e}")
        
        return results

class AsyncBotProcessor:
    """Асинхронный процессор ботов"""
    
    def __init__(self, max_concurrent_bots: int = 5):
        self.max_concurrent_bots = max_concurrent_bots
        self.semaphore = asyncio.Semaphore(max_concurrent_bots)
        self.active_tasks: Dict[str, AsyncTask] = {}
        
    async def process_bot_async(self, bot, rsi_data: Dict) -> Optional[Dict]:
        """Асинхронно обрабатывает одного бота"""
        async with self.semaphore:
            task_id = f"bot_{bot.symbol}_{int(time.time())}"
            task = AsyncTask(
                id=task_id,
                task_type="bot_processing",
                status=TaskStatus.RUNNING,
                created_at=time.time(),
                started_at=time.time()
            )
            
            self.active_tasks[task_id] = task
            
            try:
                # Используем ThreadPoolExecutor для синхронных вызовов
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        self._process_bot_sync,
                        bot,
                        rsi_data
                    )
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result
                
                return result
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                task.error = str(e)
                logger.error(f"[ASYNC_BOT] Ошибка обработки бота {bot.symbol}: {e}")
                return None
            finally:
                # Удаляем задачу из активных
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
    
    def _process_bot_sync(self, bot, rsi_data: Dict) -> Optional[Dict]:
        """Синхронная функция для обработки бота"""
        try:
            # Здесь вызываем существующие методы бота
            # Пока что возвращаем заглушку
            return {
                'symbol': bot.symbol,
                'status': 'processed',
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"[ASYNC_BOT] Синхронная ошибка для {bot.symbol}: {e}")
            return None
    
    async def process_bots_batch_async(self, bots: List, rsi_data: Dict) -> Dict[str, Dict]:
        """Асинхронно обрабатывает пакет ботов"""
        logger.info(f"[ASYNC_BOT] 🚀 Начинаем асинхронную обработку {len(bots)} ботов")
        
        tasks = []
        for bot in bots:
            task = asyncio.create_task(self.process_bot_async(bot, rsi_data))
            tasks.append((bot.symbol, task))
        
        results = {}
        completed = 0
        failed = 0
        
        for symbol, task in tasks:
            try:
                result = await task
                if result:
                    results[symbol] = result
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"[ASYNC_BOT] Ошибка задачи для {symbol}: {e}")
                failed += 1
        
        logger.info(f"[ASYNC_BOT] ✅ Завершено: {completed} успешно, {failed} ошибок")
        return results

class AsyncSignalProcessor:
    """Асинхронный процессор сигналов"""
    
    def __init__(self, max_concurrent_signals: int = 10):
        self.max_concurrent_signals = max_concurrent_signals
        self.semaphore = asyncio.Semaphore(max_concurrent_signals)
        
    async def process_signal_async(self, symbol: str, rsi_data: Dict) -> Optional[Dict]:
        """Асинхронно обрабатывает сигнал для одной монеты"""
        async with self.semaphore:
            try:
                # Используем ThreadPoolExecutor для синхронных вызовов
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        self._process_signal_sync,
                        symbol,
                        rsi_data
                    )
                return result
            except Exception as e:
                logger.error(f"[ASYNC_SIGNAL] Ошибка обработки сигнала для {symbol}: {e}")
                return None
    
    def _process_signal_sync(self, symbol: str, rsi_data: Dict) -> Optional[Dict]:
        """Синхронная функция для обработки сигнала"""
        try:
            # Получаем RSI с учетом текущего таймфрейма
            from bot_engine.config_loader import get_rsi_from_coin_data
            # Здесь вызываем существующие функции обработки сигналов
            # Пока что возвращаем заглушку
            return {
                'symbol': symbol,
                'signal': 'NEUTRAL',
                'rsi': get_rsi_from_coin_data(rsi_data) or 50,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"[ASYNC_SIGNAL] Синхронная ошибка для {symbol}: {e}")
            return None
    
    async def process_signals_batch_async(self, symbols: List[str], rsi_data: Dict) -> Dict[str, Dict]:
        """Асинхронно обрабатывает пакет сигналов"""
        logger.info(f"[ASYNC_SIGNAL] 🚀 Начинаем асинхронную обработку {len(symbols)} сигналов")
        
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self.process_signal_async(symbol, rsi_data))
            tasks.append((symbol, task))
        
        results = {}
        completed = 0
        failed = 0
        
        for symbol, task in tasks:
            try:
                result = await task
                if result:
                    results[symbol] = result
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"[ASYNC_SIGNAL] Ошибка задачи для {symbol}: {e}")
                failed += 1
        
        logger.info(f"[ASYNC_SIGNAL] ✅ Завершено: {completed} успешно, {failed} ошибок")
        return results

class AsyncDataSaver:
    """Асинхронный сохранщик данных"""
    
    def __init__(self, max_concurrent_saves: int = 3):
        self.max_concurrent_saves = max_concurrent_saves
        self.semaphore = asyncio.Semaphore(max_concurrent_saves)
        
    async def save_data_async(self, data: Dict, filename: str) -> bool:
        """Асинхронно сохраняет данные в файл"""
        async with self.semaphore:
            try:
                # Используем ThreadPoolExecutor для синхронных операций с файлами
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        self._save_data_sync,
                        data,
                        filename
                    )
                return result
            except Exception as e:
                logger.error(f"[ASYNC_SAVER] Ошибка сохранения {filename}: {e}")
                return False
    
    def _save_data_sync(self, data: Dict, filename: str) -> bool:
        """Синхронная функция для сохранения данных"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"[ASYNC_SAVER] Синхронная ошибка сохранения {filename}: {e}")
            return False

class AsyncMainProcessor:
    """Главный асинхронный процессор"""
    
    def __init__(self, exchange, config: Dict):
        self.exchange = exchange
        self.config = config
        
        logger.info(f"[ASYNC_MAIN] Инициализация с биржей: {type(exchange)}")
        
        # Инициализируем компоненты
        self.rsi_loader = AsyncRSILoader(
            exchange, 
            max_concurrent_requests=config.get('max_rsi_requests', 10)
        )
        self.bot_processor = AsyncBotProcessor(
            max_concurrent_bots=config.get('max_concurrent_bots', 5)
        )
        self.signal_processor = AsyncSignalProcessor(
            max_concurrent_signals=config.get('max_concurrent_signals', 10)
        )
        self.data_saver = AsyncDataSaver(
            max_concurrent_saves=config.get('max_concurrent_saves', 3)
        )
        
        # Состояние
        self.is_running = False
        self.last_update = 0
        
    async def start(self):
        """Запускает асинхронный процессор"""
        self.is_running = True
        logger.info("[ASYNC_MAIN] 🚀 Запуск асинхронного процессора")
        
        # Запускаем основные задачи параллельно
        tasks = [
            asyncio.create_task(self._rsi_update_loop()),
            asyncio.create_task(self._bot_processing_loop()),
            asyncio.create_task(self._signal_processing_loop()),
            asyncio.create_task(self._data_saving_loop()),
            asyncio.create_task(self._position_sync_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"[ASYNC_MAIN] Ошибка в главном цикле: {e}")
        finally:
            self.is_running = False
    
    async def _rsi_update_loop(self):
        """Цикл обновления RSI данных"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.get('rsi_update_interval', 1800))
                await self._update_rsi_data()
            except Exception as e:
                logger.error(f"[ASYNC_MAIN] Ошибка в цикле RSI: {e}")
                await asyncio.sleep(60)  # Ждем минуту перед повтором
    
    async def _bot_processing_loop(self):
        """Цикл обработки ботов"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.get('bot_processing_interval', 30))
                await self._process_bots()
            except Exception as e:
                logger.error(f"[ASYNC_MAIN] Ошибка в цикле ботов: {e}")
                await asyncio.sleep(30)
    
    async def _signal_processing_loop(self):
        """Цикл обработки сигналов"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.get('signal_processing_interval', 10))
                await self._process_signals()
            except Exception as e:
                logger.error(f"[ASYNC_MAIN] Ошибка в цикле сигналов: {e}")
                await asyncio.sleep(10)
    
    async def _data_saving_loop(self):
        """Цикл сохранения данных"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.get('data_saving_interval', 30))
                await self._save_data()
                
                # Дополнительно сохраняем maturity данные
                await self._save_maturity_data()
            except Exception as e:
                logger.error(f"[ASYNC_MAIN] Ошибка в цикле сохранения: {e}")
                await asyncio.sleep(30)
    
    async def _position_sync_loop(self):
        """Цикл синхронизации позиций с биржей"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.get('position_sync_interval', 60))  # Каждую минуту
                await self._sync_positions()
            except Exception as e:
                logger.error(f"[ASYNC_MAIN] Ошибка в цикле синхронизации позиций: {e}")
                await asyncio.sleep(60)
    
    async def _save_maturity_data(self):
        """Сохраняет maturity данные"""
        try:
            from bots import save_mature_coins_storage
            save_mature_coins_storage()
            pass
        except Exception as e:
            logger.warning(f"[ASYNC_MAIN] Ошибка сохранения maturity данных: {e}")
    
    async def _update_rsi_data(self):
        """Обновляет RSI данные"""
        logger.info("[ASYNC_MAIN] 🔄 Начинаем обновление RSI данных")
        
        # Получаем список монет
        pairs = self.exchange.get_all_pairs()
        if not pairs:
            logger.error("[ASYNC_MAIN] ❌ Не удалось получить список пар")
            return
        
        # Загружаем RSI данные асинхронно с maturity проверкой
        async with self.rsi_loader as loader:
            rsi_data = await loader.load_batch_rsi_async(pairs)
        
        # Обновляем глобальные данные RSI
        try:
            from bots import coins_rsi_data, rsi_data_lock
            with rsi_data_lock:
                coins_rsi_data['coins'] = rsi_data
                coins_rsi_data['last_update'] = time.time()
                coins_rsi_data['update_in_progress'] = False
            logger.info(f"[ASYNC_MAIN] ✅ RSI данные обновлены в глобальном кэше: {len(rsi_data)} монет")
        except Exception as e:
            logger.error(f"[ASYNC_MAIN] Ошибка обновления глобального кэша RSI: {e}")
        
        self.last_update = time.time()
    
    async def _process_bots(self):
        """Обрабатывает ботов"""
        # ✅ Не логируем частые вызовы (обработка в основном процессе)
        try:
            # Обработка торговых сигналов перенесена в основной процесс
            # для доступа к bots_data в том же процессе
            pass
            
        except Exception as e:
            logger.error(f"[ASYNC_MAIN] ❌ Ошибка обработки ботов: {e}")
            import traceback
            logger.error(f"[ASYNC_MAIN] ❌ Traceback: {traceback.format_exc()}")
    
    async def _process_signals(self):
        """Обрабатывает сигналы"""
        # ✅ Не логируем частые вызовы
        
        # Здесь получаем список монет для анализа
        # Пока что заглушка
        symbols = []
        
        if symbols:
            # Получаем RSI данные
            rsi_data = {}  # Здесь должны быть актуальные RSI данные
            
            # Обрабатываем сигналы асинхронно
            results = await self.signal_processor.process_signals_batch_async(symbols, rsi_data)
            logger.info(f"[ASYNC_MAIN] ✅ Сигналы обработаны: {len(results)} результатов")
    
    async def _sync_positions(self):
        """Синхронизирует позиции с биржей"""
        logger.info("[ASYNC_MAIN] 🔄 Начинаем синхронизацию позиций с биржей")
        
        try:
            # Импортируем функцию синхронизации
            from bots import sync_bots_with_exchange
            
            # Выполняем синхронизацию в отдельном потоке
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(executor, sync_bots_with_exchange)
                
            if result:
                logger.info("[ASYNC_MAIN] ✅ Синхронизация позиций завершена успешно")
            else:
                logger.warning("[ASYNC_MAIN] ⚠️ Синхронизация позиций завершена с предупреждениями")
                
        except Exception as e:
            logger.error(f"[ASYNC_MAIN] ❌ Ошибка синхронизации позиций: {e}")
    
    async def _save_data(self):
        """Сохраняет данные"""
        # ✅ Не логируем частые вызовы
        
        # Здесь собираем данные для сохранения
        # Пока что заглушка
        data = {
            'timestamp': time.time(),
            'last_update': self.last_update
        }
        
        # Сохраняем данные асинхронно
        success = await self.data_saver.save_data_async(data, 'data/async_state.json')
        # ✅ Не логируем успешное сохранение (слишком часто)
        if not success:
            logger.error("[ASYNC_MAIN] ❌ Ошибка сохранения данных")
    
    def stop(self):
        """Останавливает асинхронный процессор"""
        self.is_running = False
        logger.info("[ASYNC_MAIN] 🛑 Остановка асинхронного процессора")
