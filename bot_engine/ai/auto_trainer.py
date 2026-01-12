"""
Автоматический тренер ИИ моделей

Автоматически обновляет исторические данные и переобучает модели по расписанию.
Запускается как фоновый процесс вместе с ботом.
"""

import logging
import threading
import time
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from bot_engine.bot_config import AIConfig

logger = logging.getLogger('AI.AutoTrainer')


class AutoTrainer:
    """Автоматический тренер для ИИ моделей"""
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.last_data_update = None
        self.last_training = None
        
        # Защита от множественных запусков
        self._training_in_progress = False
        self._data_update_in_progress = False
        self._retrain_check_in_progress = False
        
        # Путь к скриптам
        self.scripts_dir = Path('scripts/ai')
        self.collect_script = self.scripts_dir / 'collect_historical_data.py'
        self.train_anomaly_script = self.scripts_dir / 'train_anomaly_on_real_data.py'
        self.train_lstm_script = self.scripts_dir / 'train_lstm_predictor.py'
        self.train_pattern_script = self.scripts_dir / 'train_pattern_detector.py'
    
    def start(self):
        """Запускает автоматический тренер в фоновом режиме"""
        if self.running:
            logger.warning("[AutoTrainer] Уже запущен")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="AI_AutoTrainer")
        self.thread.start()
        
        logger.info("[AutoTrainer] ✅ Запущен в фоновом режиме")
        logger.info(f"[AutoTrainer] Расписание:")
        logger.info(f"[AutoTrainer]   - Обновление данных: каждые {AIConfig.AI_DATA_UPDATE_INTERVAL/3600:.0f}ч")
        logger.info(f"[AutoTrainer]   - Переобучение: каждые {AIConfig.AI_RETRAIN_INTERVAL/3600:.0f}ч")
    
    def stop(self):
        """Останавливает автоматический тренер"""
        if not self.running:
            return
        
        logger.warning("[AutoTrainer] Остановка...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.warning("[AutoTrainer] ✅ Остановлен")
    
    def _run(self):
        """Основной цикл автоматического тренера"""
        logger.info("[AutoTrainer] 🔄 Фоновый процесс запущен")
        
        # Проверяем нужно ли обучение при старте
        self._check_initial_training()
        
        while self.running:
            try:
                current_time = time.time()
                
                # 1. Проверяем нужно ли обновить данные
                data_updated = False
                if self._should_update_data(current_time) and not self._data_update_in_progress:
                    data_updated = self._update_data()
                
                # 2. Проверяем нужно ли переобучить модель
                # ВАЖНО: Переобучаем только если данные НЕ обновлялись или обновились успешно
                if self._should_retrain(current_time) and not self._training_in_progress:
                    if not data_updated or data_updated == True:  # Данные не обновлялись или обновились успешно
                        self._retrain()
                    else:
                        logger.warning("[AutoTrainer] ⚠️ Переобучение отложено из-за ошибки обновления данных")
                
                # 3. УЛУЧШЕНИЕ: Проверяем нужно ли переобучить основные модели на реальных сделках
                if not self._retrain_check_in_progress:
                    self._check_real_trades_retrain()
                
                # Спим до следующей проверки (каждые 10 минут)
                time.sleep(600)
                
            except KeyboardInterrupt:
                logger.warning("[AutoTrainer] ⚠️ Получен сигнал остановки (Ctrl+C)")
                self.running = False
                break
            except Exception as e:
                logger.error(f"[AutoTrainer] Ошибка в цикле: {e}")
                time.sleep(60)
        
        logger.warning("[AutoTrainer] 🛑 Auto Trainer остановлен")
    
    def _check_initial_training(self):
        """Проверяет нужно ли обучение при старте"""
        # Проверяем несколько моделей
        models_found = []
        
        # 1. Проверяем Anomaly Detector
        anomaly_model_path = Path(AIConfig.AI_ANOMALY_MODEL_PATH)
        if anomaly_model_path.exists():
            models_found.append("anomaly_detector")
            logger.info("[AutoTrainer] ✅ Anomaly Detector найден в файле")
        else:
            # Проверяем БД на наличие модели
            try:
                from bot_engine.ai.ai_database import AIDatabase
                ai_db = AIDatabase()
                model_version = ai_db.get_latest_model_version(
                    model_type='anomaly_detector',
                    symbol=None
                )
                if model_version:
                    models_found.append("anomaly_detector")
                    logger.info("[AutoTrainer] ✅ Anomaly Detector найден в БД")
            except Exception as e:
                logger.debug(f"[AutoTrainer] Ошибка проверки Anomaly Detector в БД: {e}")
        
        # 2. Проверяем Parameter Quality Predictor
        try:
            from bot_engine.ai.ai_database import _get_project_root
            project_root = _get_project_root()
            param_quality_model_path = project_root / 'data' / 'ai' / 'models' / 'parameter_quality_predictor.pkl'
        except:
            # Fallback: используем относительный путь
            param_quality_model_path = Path('data/ai/models/parameter_quality_predictor.pkl')
        
        if param_quality_model_path.exists():
            models_found.append("parameter_quality_predictor")
            logger.info("[AutoTrainer] ✅ Parameter Quality Predictor найден в файле")
        else:
            # Проверяем БД на наличие образцов для обучения
            try:
                from bot_engine.ai.ai_database import AIDatabase
                ai_db = AIDatabase()
                samples_count = ai_db.count_parameter_training_samples()
                if samples_count >= 50:  # Минимум для обучения
                    # Модель может быть обучена, но файл не найден
                    # Это нормально - модель будет обучена при следующем запуске обучения
                    logger.info(f"[AutoTrainer] ℹ️ Parameter Quality Predictor: {samples_count} образцов в БД (достаточно для обучения)")
            except Exception as e:
                logger.debug(f"[AutoTrainer] Ошибка проверки Parameter Quality Predictor: {e}")
        
        # Если хотя бы одна модель найдена - считаем что обучение не требуется
        if len(models_found) > 0:
            logger.info(f"[AutoTrainer] ✅ Найдено моделей: {', '.join(models_found)}, первичное обучение не требуется")
        else:
            logger.warning("[AutoTrainer] ⚠️ Модель не найдена, требуется первичное обучение")
            
            if AIConfig.AI_AUTO_TRAIN_ON_STARTUP:
                logger.info("[AutoTrainer] 🚀 Запускаем первичное обучение...")
                self._initial_setup()
    
    def _initial_setup(self):
        """Первичная настройка - сбор данных и обучение"""
        logger.info("[AutoTrainer] Первичная настройка...")
        
        # 1. Собираем данные
        logger.info("[AutoTrainer] Шаг 1/2: Сбор исторических данных...")
        success = self._update_data(initial=True)
        
        if not success:
            logger.error("[AutoTrainer] ❌ Не удалось собрать данные")
            return
        
        # 2. Обучаем модель
        logger.info("[AutoTrainer] Шаг 2/2: Обучение модели...")
        success = self._retrain()
        
        if success:
            logger.info("[AutoTrainer] ✅ Первичная настройка завершена")
        else:
            logger.error("[AutoTrainer] ❌ Ошибка первичного обучения")
    
    def _should_update_data(self, current_time: float) -> bool:
        """Проверяет нужно ли обновить данные"""
        if not AIConfig.AI_AUTO_UPDATE_DATA:
            return False
        
        # При первом запуске НЕ обновляем сразу (данные уже есть)
        if self.last_data_update is None:
            self.last_data_update = current_time  # Инициализируем текущим временем
            return False
        
        elapsed = current_time - self.last_data_update
        return elapsed >= AIConfig.AI_DATA_UPDATE_INTERVAL
    
    def _should_retrain(self, current_time: float) -> bool:
        """Проверяет нужно ли переобучить модель"""
        if not AIConfig.AI_AUTO_RETRAIN:
            return False
        
        # При первом запуске НЕ переобучаем сразу (модель уже обучена)
        if self.last_training is None:
            self.last_training = current_time  # Инициализируем текущим временем
            return False
        
        elapsed = current_time - self.last_training
        return elapsed >= AIConfig.AI_RETRAIN_INTERVAL
    
    def _update_data(self, initial: bool = False) -> bool:
        """
        Обновляет исторические данные
        
        Args:
            initial: True если это первичная настройка
        
        Returns:
            True если успешно
        """
        # Защита от множественных запусков
        if self._data_update_in_progress:
            logger.debug("[AutoTrainer] Обновление данных уже выполняется, пропускаем...")
            return False
        
        self._data_update_in_progress = True
        try:
            logger.info("[AutoTrainer] 📥 Обновление исторических данных...")
            
            # Определяем количество монет
            if initial:
                # Первичная настройка - собираем больше данных
                limit = AIConfig.AI_INITIAL_COINS_COUNT
                days = 730  # 2 года для первичной настройки
            else:
                # Обновление
                limit = AIConfig.AI_UPDATE_COINS_COUNT
                days = 30  # Обновляем только последние 30 дней
            
            # Запускаем скрипт сбора данных
            cmd = [
                sys.executable,
                str(self.collect_script),
                '--days', str(days)
            ]
            
            # Если limit=0, собираем все монеты (флаг --all)
            if limit == 0:
                cmd.append('--all')
                logger.info("[AutoTrainer] Режим: ВСЕ монеты с биржи")
            else:
                cmd.extend(['--limit', str(limit)])
                logger.info(f"[AutoTrainer] Режим: Топ {limit} монет")
            
            logger.info(f"[AutoTrainer] Запуск: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 час таймаут
            )
            
            if result.returncode == 0:
                logger.info("[AutoTrainer] ✅ Данные успешно обновлены")
                self.last_data_update = time.time()
                return True
            else:
                # Проверяем, не был ли прерван скрипт (KeyboardInterrupt)
                stderr_text = result.stderr or ""
                if "KeyboardInterrupt" in stderr_text:
                    logger.warning("[AutoTrainer] ⚠️ Сбор данных прерван пользователем")
                    # Останавливаем Auto Trainer при прерывании
                    self.running = False
                else:
                    # Ограничиваем вывод ошибки (первые 500 символов)
                    error_preview = stderr_text[:500] if len(stderr_text) > 500 else stderr_text
                    if len(stderr_text) > 500:
                        error_preview += f"\n... (еще {len(stderr_text) - 500} символов)"
                    logger.error(f"[AutoTrainer] ❌ Ошибка обновления данных: {error_preview}")
                return False
        
        except subprocess.TimeoutExpired:
            logger.error("[AutoTrainer] ❌ Таймаут при обновлении данных")
            return False
        except KeyboardInterrupt:
            logger.warning("[AutoTrainer] ⚠️ Обновление данных прервано пользователем")
            # Останавливаем Auto Trainer
            self.running = False
            return False
        except Exception as e:
            logger.error(f"[AutoTrainer] ❌ Ошибка обновления данных: {e}")
            return False
        finally:
            self._data_update_in_progress = False
    
    def _retrain(self) -> bool:
        """
        Переобучает модели на обновленных данных
        
        Returns:
            True если успешно
        """
        # Защита от множественных запусков
        if self._training_in_progress:
            logger.debug("[AutoTrainer] Обучение уже выполняется, пропускаем...")
            return False
        
        self._training_in_progress = True
        try:
            logger.info("[AutoTrainer] 🧠 Переобучение моделей...")
            
            all_success = True
            
            # 1. Обучаем Anomaly Detector
            if AIConfig.AI_ANOMALY_DETECTION_ENABLED:
                logger.info("[AutoTrainer] 📊 Обучение Anomaly Detector...")
                success = self._train_model(
                    self.train_anomaly_script,
                    "Anomaly Detector",
                    timeout=600
                )
                if not success:
                    all_success = False
            
            # 2. Обучаем LSTM Predictor
            if AIConfig.AI_LSTM_ENABLED:
                logger.info("[AutoTrainer] 🧠 Обучение LSTM Predictor...")
                success = self._train_model(
                    self.train_lstm_script,
                    "LSTM Predictor",
                    timeout=1800,  # 30 минут для LSTM
                    args=['--coins', '0', '--epochs', '50']
                )
                if not success:
                    all_success = False
            
            # 3. Обучаем Pattern Detector
            if AIConfig.AI_PATTERN_ENABLED:
                logger.info("[AutoTrainer] 📊 Обучение Pattern Detector...")
                success = self._train_model(
                    self.train_pattern_script,
                    "Pattern Detector",
                    timeout=600,
                    args=['--coins', '0']
                )
                if not success:
                    all_success = False
            
            if all_success:
                logger.info("[AutoTrainer] ✅ Все модели успешно переобучены")
                self.last_training = time.time()
                
                # Перезагружаем модели в AI Manager
                self._reload_models()
                
                return True
            else:
                logger.warning("[AutoTrainer] ⚠️ Не все модели обучены успешно")
                return False
        
        except KeyboardInterrupt:
            logger.warning("[AutoTrainer] ⚠️ Переобучение прервано пользователем")
            # Останавливаем Auto Trainer
            self.running = False
            return False
        except Exception as e:
            logger.error(f"[AutoTrainer] ❌ Ошибка обучения: {e}")
            return False
        finally:
            self._training_in_progress = False
    
    def _train_model(self, script_path: Path, model_name: str, timeout: int = 600, args: list = None) -> bool:
        """
        Обучает конкретную модель
        
        Args:
            script_path: Путь к скрипту обучения
            model_name: Название модели для логов
            timeout: Таймаут в секундах
            args: Дополнительные аргументы для скрипта
        
        Returns:
            True если успешно
        """
        try:
            # УЛУЧШЕНИЕ: Проверяем существование скрипта перед запуском
            if not script_path.exists():
                logger.error(f"[AutoTrainer] ❌ Скрипт не найден: {script_path}")
                logger.error(f"[AutoTrainer]    Полный путь: {script_path.absolute()}")
                return False
            
            cmd = [sys.executable, str(script_path)]
            if args:
                cmd.extend([str(arg) for arg in args])
            
            logger.info(f"[AutoTrainer] Запуск: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                logger.info(f"[AutoTrainer] ✅ {model_name} успешно обучен")
                return True
            else:
                # УЛУЧШЕНИЕ: Логируем и stdout и stderr для полной диагностики
                error_output = ""
                
                # Собираем вывод из stderr
                if result.stderr:
                    error_output += f"STDERR:\n{result.stderr}\n"
                
                # Собираем вывод из stdout (могут быть ошибки и там)
                if result.stdout:
                    # Проверяем, есть ли в stdout признаки ошибки
                    stdout_lines = result.stdout.strip().split('\n')
                    error_lines = [line for line in stdout_lines if any(keyword in line.upper() for keyword in ['ERROR', 'EXCEPTION', 'TRACEBACK', 'FAILED', 'FAIL'])]
                    if error_lines:
                        error_output += f"STDOUT (ошибки):\n" + "\n".join(error_lines) + "\n"
                    # Если stderr пустой, показываем последние строки stdout
                    elif not result.stderr:
                        error_output += f"STDOUT (последние строки):\n" + "\n".join(stdout_lines[-10:]) + "\n"
                
                # Ограничиваем длину вывода (первые 1000 символов)
                if len(error_output) > 1000:
                    error_preview = error_output[:1000]
                    error_preview += f"\n... (еще {len(error_output) - 1000} символов)"
                    logger.error(f"[AutoTrainer] ❌ Ошибка обучения {model_name}:\n{error_preview}")
                else:
                    logger.error(f"[AutoTrainer] ❌ Ошибка обучения {model_name}:\n{error_output}")
                
                return False
        
        except subprocess.TimeoutExpired:
            logger.error(f"[AutoTrainer] ❌ Таймаут при обучении {model_name}")
            return False
        except Exception as e:
            logger.error(f"[AutoTrainer] ❌ Ошибка обучения {model_name}: {e}")
            return False
    
    def _reload_models(self):
        """Перезагружает все модели в AI Manager без перезапуска бота"""
        try:
            from bot_engine.ai.ai_manager import get_ai_manager
            
            ai_manager = get_ai_manager()
            
            if not ai_manager:
                logger.debug("[AutoTrainer] AI Manager не инициализирован")
                return
            
            # 1. Перезагружаем Anomaly Detector
            if ai_manager.anomaly_detector:
                try:
                    model_path = AIConfig.AI_ANOMALY_MODEL_PATH
                    scaler_path = AIConfig.AI_ANOMALY_SCALER_PATH
                    
                    success = ai_manager.anomaly_detector.load_model(model_path, scaler_path)
                    
                    if success:
                        logger.info("[AutoTrainer] ✅ Anomaly Detector перезагружен (hot reload)")
                    else:
                        logger.error("[AutoTrainer] ❌ Ошибка перезагрузки Anomaly Detector")
                except Exception as e:
                    logger.error(f"[AutoTrainer] Ошибка hot reload Anomaly Detector: {e}")
            
            # 2. Перезагружаем LSTM Predictor
            if ai_manager.lstm_predictor:
                try:
                    ai_manager.lstm_predictor.load_model()
                    logger.info("[AutoTrainer] ✅ LSTM Predictor перезагружен (hot reload)")
                except Exception as e:
                    logger.error(f"[AutoTrainer] Ошибка hot reload LSTM Predictor: {e}")
            
            # 3. Перезагружаем Pattern Detector
            if ai_manager.pattern_detector:
                try:
                    ai_manager.pattern_detector.load_model()
                    logger.info("[AutoTrainer] ✅ Pattern Detector перезагружен (hot reload)")
                except Exception as e:
                    logger.error(f"[AutoTrainer] Ошибка hot reload Pattern Detector: {e}")
        
        except Exception as e:
            logger.error(f"[AutoTrainer] Ошибка hot reload: {e}")
    
    def _check_real_trades_retrain(self):
        """
        Проверяет и запускает переобучение основных моделей на реальных сделках
        
        Это улучшение позволяет AI автоматически обучаться на реальных результатах торговли
        """
        # Защита от множественных запусков
        if self._retrain_check_in_progress:
            return
        
        self._retrain_check_in_progress = True
        try:
            from bot_engine.ai import get_ai_system
            
            ai_system = get_ai_system()
            if not ai_system or not ai_system.trainer:
                return
            
            # Проверяем, нужно ли переобучение
            should_retrain = ai_system.trainer._should_retrain_real_trades_models()
            
            if should_retrain['retrain']:
                logger.info(f"[AutoTrainer] 🔄 Обнаружена необходимость переобучения на реальных сделках: {should_retrain['reason']}")
                logger.info(f"[AutoTrainer] 📊 Текущее количество сделок: {should_retrain['trades_count']}")
                
                # Запускаем переобучение в отдельном потоке, чтобы не блокировать основной цикл
                import threading
                retrain_thread = threading.Thread(
                    target=ai_system.trainer.auto_retrain_real_trades_models,
                    args=(False,),
                    daemon=True,
                    name="AutoRetrainRealTrades"
                )
                retrain_thread.start()
                logger.info("[AutoTrainer] 🚀 Запущено автоматическое переобучение на реальных сделках (в фоне)")
        except Exception as e:
            logger.debug(f"[AutoTrainer] ⚠️ Ошибка проверки переобучения на реальных сделках: {e}")
        finally:
            self._retrain_check_in_progress = False
    
    def force_update(self) -> bool:
        """
        Принудительное обновление данных и переобучение
        
        Returns:
            True если успешно
        """
        logger.info("[AutoTrainer] 🔄 Принудительное обновление...")
        
        success = self._update_data()
        if success:
            success = self._retrain()
        
        return success
    
    def get_status(self) -> dict:
        """
        Возвращает статус автоматического тренера
        
        Returns:
            Словарь со статусом
        """
        return {
            'running': self.running,
            'last_data_update': datetime.fromtimestamp(self.last_data_update).isoformat() if self.last_data_update else None,
            'last_training': datetime.fromtimestamp(self.last_training).isoformat() if self.last_training else None,
            'next_data_update': datetime.fromtimestamp(self.last_data_update + AIConfig.AI_DATA_UPDATE_INTERVAL).isoformat() if self.last_data_update else None,
            'next_training': datetime.fromtimestamp(self.last_training + AIConfig.AI_RETRAIN_INTERVAL).isoformat() if self.last_training else None
        }


# Глобальный экземпляр
_auto_trainer: Optional[AutoTrainer] = None


def get_auto_trainer() -> AutoTrainer:
    """
    Получает глобальный экземпляр автоматического тренера
    
    Returns:
        Экземпляр AutoTrainer
    """
    global _auto_trainer
    
    if _auto_trainer is None:
        _auto_trainer = AutoTrainer()
    
    return _auto_trainer


def start_auto_trainer():
    """Запускает автоматический тренер"""
    if AIConfig.AI_AUTO_TRAIN_ENABLED:
        trainer = get_auto_trainer()
        trainer.start()
    else:
        logger.info("[AutoTrainer] Автоматическое обучение отключено в конфиге")


def stop_auto_trainer():
    """Останавливает автоматический тренер"""
    global _auto_trainer
    
    if _auto_trainer:
        _auto_trainer.stop()

