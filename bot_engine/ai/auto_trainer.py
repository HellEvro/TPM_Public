"""
Автоматический тренер ИИ моделей

Автоматически обновляет исторические данные и переобучает модели по расписанию.
Запускается как фоновый процесс вместе с ботом.

Включает ExperimentTracker для логирования экспериментов (опционально MLflow).
"""

import logging
import threading
import time
import subprocess
import sys
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from bot_engine.bot_config import AIConfig

logger = logging.getLogger('AI.AutoTrainer')

# Проверяем доступность MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """
    Трекер экспериментов для AI моделей
    
    Поддерживает:
    - MLflow (если установлен)
    - Локальное сохранение в JSON (fallback)
    
    Использование:
        tracker = ExperimentTracker("lstm_training")
        tracker.start_run("run_001")
        tracker.log_params({"epochs": 100, "lr": 0.001})
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=10)
        tracker.end_run()
    """
    
    def __init__(
        self,
        experiment_name: str = "ai_training",
        tracking_uri: str = "data/ai/mlruns",
        use_mlflow: bool = True
    ):
        """
        Args:
            experiment_name: Название эксперимента
            tracking_uri: Путь для хранения логов (для MLflow или JSON)
            use_mlflow: Использовать MLflow если доступен
        """
        self.experiment_name = experiment_name
        self.tracking_uri = Path(tracking_uri)
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        
        self.current_run = None
        self.current_run_data = {}
        
        # Создаём директорию для логов
        self.tracking_uri.mkdir(parents=True, exist_ok=True)
        
        if self.use_mlflow:
            mlflow.set_tracking_uri(str(self.tracking_uri))
            mlflow.set_experiment(experiment_name)
            logger.info(f"[ExperimentTracker] MLflow эксперимент: {experiment_name}")
        else:
            self.local_log_file = self.tracking_uri / f"{experiment_name}_runs.json"
            self.runs_history = self._load_local_runs()
            logger.info(f"[ExperimentTracker] Локальный трекинг: {self.local_log_file}")
    
    def _load_local_runs(self) -> List[Dict]:
        """Загружает историю запусков из локального файла"""
        if self.local_log_file.exists():
            try:
                with open(self.local_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_local_runs(self):
        """Сохраняет историю запусков в локальный файл"""
        try:
            with open(self.local_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.runs_history, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"[ExperimentTracker] Ошибка сохранения: {e}")
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Начинает новый запуск эксперимента
        
        Returns:
            run_id: Идентификатор запуска
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.use_mlflow:
            self.current_run = mlflow.start_run(run_name=run_name)
            run_id = self.current_run.info.run_id
        else:
            run_id = f"{run_name}_{int(time.time())}"
            self.current_run_data = {
                'run_id': run_id,
                'run_name': run_name,
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'params': {},
                'metrics': {},
                'tags': {},
                'status': 'RUNNING'
            }
        
        logger.debug(f"[ExperimentTracker] Запуск: {run_name} ({run_id})")
        return run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Логирует гиперпараметры"""
        if not self.current_run and not self.current_run_data:
            logger.warning("[ExperimentTracker] Нет активного запуска")
            return
        
        if self.use_mlflow:
            mlflow.log_params(params)
        else:
            self.current_run_data['params'].update(params)
        
        logger.debug(f"[ExperimentTracker] Params: {params}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Логирует метрики"""
        if not self.current_run and not self.current_run_data:
            logger.warning("[ExperimentTracker] Нет активного запуска")
            return
        
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        else:
            for key, value in metrics.items():
                if key not in self.current_run_data['metrics']:
                    self.current_run_data['metrics'][key] = []
                self.current_run_data['metrics'][key].append({
                    'value': value,
                    'step': step,
                    'timestamp': datetime.now().isoformat()
                })
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Логирует одну метрику"""
        self.log_metrics({key: value}, step=step)
    
    def set_tag(self, key: str, value: str):
        """Устанавливает тег"""
        if self.use_mlflow:
            mlflow.set_tag(key, value)
        elif self.current_run_data:
            self.current_run_data['tags'][key] = value
    
    def log_model(self, model, model_name: str):
        """Логирует модель (только для MLflow)"""
        if self.use_mlflow:
            try:
                # Пробуем разные форматы
                if hasattr(model, 'state_dict'):
                    # PyTorch
                    mlflow.pytorch.log_model(model, model_name)
                else:
                    # Sklearn или другие
                    mlflow.sklearn.log_model(model, model_name)
            except Exception as e:
                logger.debug(f"[ExperimentTracker] Не удалось залогировать модель: {e}")
    
    def end_run(self, status: str = 'FINISHED'):
        """Завершает текущий запуск"""
        if self.use_mlflow:
            if self.current_run:
                mlflow.end_run(status=status)
                self.current_run = None
        else:
            if self.current_run_data:
                self.current_run_data['end_time'] = datetime.now().isoformat()
                self.current_run_data['status'] = status
                self.runs_history.append(self.current_run_data)
                self._save_local_runs()
                self.current_run_data = {}
        
        logger.debug(f"[ExperimentTracker] Запуск завершён: {status}")
    
    def get_best_run(self, metric: str = 'accuracy', maximize: bool = True) -> Optional[Dict]:
        """Возвращает лучший запуск по метрике"""
        if self.use_mlflow:
            try:
                runs = mlflow.search_runs(order_by=[f"metrics.{metric} {'DESC' if maximize else 'ASC'}"])
                if not runs.empty:
                    return runs.iloc[0].to_dict()
            except:
                pass
            return None
        else:
            if not self.runs_history:
                return None
            
            best_run = None
            best_value = None
            
            for run in self.runs_history:
                if metric in run.get('metrics', {}):
                    values = run['metrics'][metric]
                    if values:
                        last_value = values[-1]['value']
                        if best_value is None or (maximize and last_value > best_value) or (not maximize and last_value < best_value):
                            best_value = last_value
                            best_run = run
            
            return best_run
    
    def get_runs_history(self, limit: int = 10) -> List[Dict]:
        """Возвращает историю запусков"""
        if self.use_mlflow:
            try:
                runs = mlflow.search_runs(max_results=limit)
                return runs.to_dict('records') if not runs.empty else []
            except:
                return []
        else:
            return self.runs_history[-limit:]


# Глобальный экземпляр трекера
_experiment_tracker: Optional[ExperimentTracker] = None


def get_experiment_tracker(experiment_name: str = "ai_training") -> ExperimentTracker:
    """Получает глобальный экземпляр трекера экспериментов"""
    global _experiment_tracker
    if _experiment_tracker is None:
        _experiment_tracker = ExperimentTracker(experiment_name)
    return _experiment_tracker


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
        
        # Счетчики для триггеров остановки обучения
        self._training_attempts = 0  # Количество попыток обучения
        self._last_model_accuracy = None  # Последняя точность модели
        self._training_stopped = False  # Флаг остановки обучения
        
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
        logger.info(f"[AutoTrainer] Режим: НЕПРЕРЫВНОЕ ОБУЧЕНИЕ")
        logger.info(f"[AutoTrainer] Расписание:")
        logger.info(f"[AutoTrainer]   - Обновление данных: каждые {AIConfig.AI_DATA_UPDATE_INTERVAL/3600:.0f}ч")
        logger.info(f"[AutoTrainer]   - Переобучение: НЕПРЕРЫВНО (сразу после завершения предыдущего)")
        if AIConfig.AI_STOP_TRAINING_ON_HIGH_ACCURACY:
            logger.info(f"[AutoTrainer]   - Триггер остановки: точность >= {AIConfig.AI_HIGH_ACCURACY_THRESHOLD:.0%}")
        if AIConfig.AI_STOP_TRAINING_ON_DEGRADATION:
            logger.info(f"[AutoTrainer]   - Триггер остановки: ухудшение >= {AIConfig.AI_DEGRADATION_THRESHOLD:.0%}")
        if AIConfig.AI_RETRAIN_ON_REAL_PERFORMANCE_DEGRADATION:
            logger.info(f"[AutoTrainer]   - Триггер переобучения на реальных сделках:")
            logger.info(f"[AutoTrainer]     * Win_rate < {AIConfig.AI_REAL_WIN_RATE_THRESHOLD:.0%}")
            logger.info(f"[AutoTrainer]     * Avg_pnl < {AIConfig.AI_REAL_AVG_PNL_THRESHOLD:.2f} USDT")
            logger.info(f"[AutoTrainer]     * Разница виртуальных/реальных > {AIConfig.AI_REAL_VS_SIMULATED_DIFF_THRESHOLD:.0%}")
        if AIConfig.AI_TRAIN_ON_SIMULATIONS:
            logger.info(f"[AutoTrainer]   - Обучение на симуляциях: ВКЛЮЧЕНО")
            logger.info(f"[AutoTrainer]     * Целевой win_rate: {AIConfig.AI_SIMULATIONS_TARGET_WIN_RATE:.0%}")
            logger.info(f"[AutoTrainer]     * Максимум симуляций: {AIConfig.AI_SIMULATIONS_MAX_ITERATIONS}")
            logger.info(f"[AutoTrainer]     * Автопереключение: {'ДА' if AIConfig.AI_USE_SIMULATIONS_WHEN_REAL_LOW else 'НЕТ'}")
    
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
                    model_type='anomaly_detector'
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
        """Проверяет нужно ли переобучить модель
        
        НЕПРЕРЫВНОЕ ОБУЧЕНИЕ с триггерами остановки:
        - Останавливается при достижении высокой точности (90%+)
        - Останавливается при ухудшении качества
        """
        if not AIConfig.AI_AUTO_RETRAIN:
            return False
        
        # Если обучение уже идет, не запускаем новое
        if self._training_in_progress:
            return False
        
        # Если обучение остановлено триггерами, не запускаем новое
        if self._training_stopped:
            return False
        
        # При первом запуске НЕ переобучаем сразу (модель уже обучена или будет обучена при старте)
        if self.last_training is None:
            self.last_training = current_time  # Инициализируем текущим временем
            return False
        
        # Проверяем качество модели перед запуском обучения
        if self._training_attempts >= AIConfig.AI_MIN_TRAINING_ATTEMPTS:
            should_stop = self._check_should_stop_training()
            if should_stop:
                if not self._training_stopped:
                    logger.info("[AutoTrainer] 🛑 Обучение остановлено: достигнута высокая точность или обнаружено ухудшение качества")
                    self._training_stopped = True
                return False
        
        # НЕПРЕРЫВНОЕ ОБУЧЕНИЕ: Всегда возвращаем True, если обучение завершено и нет триггеров остановки
        # Это обеспечивает непрерывное обучение без перерыва
        return True
    
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
            
            # Показываем команду без полного пути к Python
            cmd_display = ['python'] + cmd[1:]
            logger.info(f"[AutoTrainer] Запуск: {' '.join(cmd_display)}")
            
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
        
        # Инициализируем трекер экспериментов
        tracker = get_experiment_tracker("auto_training")
        run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tracker.start_run(run_name)
        
        # Логируем параметры
        tracker.log_params({
            'anomaly_enabled': AIConfig.AI_ANOMALY_DETECTION_ENABLED,
            'lstm_enabled': AIConfig.AI_LSTM_ENABLED,
            'pattern_enabled': AIConfig.AI_PATTERN_ENABLED,
            'training_mode': 'continuous',
        })
        
        try:
            logger.info("[AutoTrainer] 🧠 Переобучение моделей...")
            
            all_success = True
            models_trained = 0
            
            # 1. Обучаем Anomaly Detector
            if AIConfig.AI_ANOMALY_DETECTION_ENABLED:
                logger.info("[AutoTrainer] 📊 Обучение Anomaly Detector...")
                success = self._train_model(
                    self.train_anomaly_script,
                    "Anomaly Detector",
                    timeout=600
                )
                tracker.log_metric('anomaly_success', 1 if success else 0)
                if success:
                    models_trained += 1
                if not success:
                    all_success = False
            
            # 2. Обучаем основные модели (signal_predictor, profit_predictor)
            # Проверяем количество реальных сделок
            from bot_engine.ai import get_ai_system
            ai_system = get_ai_system()
            if ai_system and ai_system.trainer:
                trainer = ai_system.trainer
                
                # Проверяем количество реальных сделок
                real_trades_count = trainer.get_trades_count()
                
                if real_trades_count < trainer._real_trades_min_samples and AIConfig.AI_USE_SIMULATIONS_WHEN_REAL_LOW:
                    logger.info(f"[AutoTrainer] 📊 Реальных сделок мало ({real_trades_count} < {trainer._real_trades_min_samples})")
                    logger.info("[AutoTrainer] 🎲 Переключаемся на обучение на симуляциях с оптимизацией параметров...")
                    
                    if AIConfig.AI_TRAIN_ON_SIMULATIONS:
                        success = trainer.train_on_simulations(
                            target_win_rate=AIConfig.AI_SIMULATIONS_TARGET_WIN_RATE,
                            max_simulations=AIConfig.AI_SIMULATIONS_MAX_ITERATIONS
                        )
                        if not success:
                            all_success = False
                    else:
                        logger.warning("[AutoTrainer] ⚠️ Обучение на симуляциях отключено в конфиге")
                else:
                    # Обучаем на реальных сделках
                    logger.info("[AutoTrainer] 📊 Обучение на реальных сделках...")
                    trainer.train_on_history()
            
            # 3. Обучаем LSTM Predictor
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
            
            # 4. Обучаем Pattern Detector
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
                self._training_attempts += 1
                
                # Логируем финальные метрики
                tracker.log_metrics({
                    'models_trained': models_trained,
                    'all_success': 1,
                    'training_attempts': self._training_attempts,
                })
                tracker.set_tag('status', 'SUCCESS')
                tracker.end_run('FINISHED')
                
                # Проверяем качество модели после обучения
                self._check_model_quality_after_training()
                
                # Перезагружаем модели в AI Manager
                self._reload_models()
                
                return True
            else:
                logger.warning("[AutoTrainer] ⚠️ Не все модели обучены успешно")
                tracker.log_metric('all_success', 0)
                tracker.set_tag('status', 'PARTIAL')
                tracker.end_run('FINISHED')
                return False
        
        except KeyboardInterrupt:
            logger.warning("[AutoTrainer] ⚠️ Переобучение прервано пользователем")
            tracker.set_tag('status', 'INTERRUPTED')
            tracker.end_run('KILLED')
            # Останавливаем Auto Trainer
            self.running = False
            return False
        except Exception as e:
            logger.error(f"[AutoTrainer] ❌ Ошибка обучения: {e}")
            tracker.set_tag('status', 'FAILED')
            tracker.set_tag('error', str(e))
            tracker.end_run('FAILED')
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
            
            # Показываем команду без полного пути к Python
            cmd_display = ['python'] + cmd[1:]
            logger.info(f"[AutoTrainer] Запуск: {' '.join(cmd_display)}")
            
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
    
    def _check_model_quality_after_training(self):
        """Проверяет качество модели после обучения и обновляет метрики"""
        try:
            from bot_engine.ai.ai_database import AIDatabase
            ai_db = AIDatabase()
            
            # Получаем последние версии моделей
            models_to_check = [
                ('signal_predictor', 'signal_predictor'),
                ('profit_predictor', 'profit_predictor'),
                ('ai_decision_model', 'ai_decision_model'),
            ]
            
            max_accuracy = 0.0
            for model_name, model_type in models_to_check:
                model_version = ai_db.get_latest_model_version(model_type=model_type)
                if model_version:
                    # Проверяем accuracy или signal_accuracy
                    accuracy = model_version.get('accuracy') or model_version.get('signal_accuracy')
                    if accuracy is not None:
                        accuracy = float(accuracy)
                        max_accuracy = max(max_accuracy, accuracy)
                        logger.debug(f"[AutoTrainer] 📊 {model_name}: accuracy = {accuracy:.2%}")
            
            # Обновляем последнюю точность
            if max_accuracy > 0:
                self._last_model_accuracy = max_accuracy
                logger.info(f"[AutoTrainer] 📊 Максимальная точность модели: {max_accuracy:.2%}")
            
            # Проверяем производительность на реальных сделках
            self._check_real_trades_performance()
        
        except Exception as e:
            logger.debug(f"[AutoTrainer] ⚠️ Ошибка проверки качества модели: {e}")
    
    def _check_real_trades_performance(self):
        """
        Проверяет производительность модели на реальных сделках и запускает переобучение при ухудшении
        
        Если модель показывает 90%+ на виртуальных сделках, но на реальных сделках результаты отрицательные,
        запускается переобучение на реальных данных.
        """
        if not AIConfig.AI_RETRAIN_ON_REAL_PERFORMANCE_DEGRADATION:
            return
        
        try:
            from bot_engine.ai.ai_database import AIDatabase
            ai_db = AIDatabase()
            
            # Получаем статистику виртуальных vs реальных сделок
            comparison = ai_db.compare_simulated_vs_real()
            
            sim_stats = comparison.get('simulated', {})
            real_stats = comparison.get('real', {})
            comp_data = comparison.get('comparison', {})
            
            sim_win_rate = sim_stats.get('win_rate') or 0
            real_win_rate = real_stats.get('win_rate') or 0
            real_avg_pnl = real_stats.get('avg_pnl') or 0
            real_count = real_stats.get('count') or 0
            win_rate_diff = comp_data.get('win_rate_diff', 0)
            
            # Проверяем, достаточно ли реальных сделок для оценки
            if real_count < AIConfig.AI_REAL_PERFORMANCE_WINDOW:
                logger.debug(f"[AutoTrainer] 📊 Недостаточно реальных сделок для оценки: {real_count} < {AIConfig.AI_REAL_PERFORMANCE_WINDOW}")
                return
            
            logger.info(f"[AutoTrainer] 📊 Производительность на реальных сделках:")
            logger.info(f"   Виртуальные: win_rate = {sim_win_rate:.2%}, avg_pnl = {sim_stats.get('avg_pnl', 0):.2f} USDT")
            logger.info(f"   Реальные: win_rate = {real_win_rate:.2%}, avg_pnl = {real_avg_pnl:.2f} USDT")
            logger.info(f"   Разница win_rate: {win_rate_diff:.2%}")
            
            # Триггер 1: Низкий win_rate на реальных сделках
            if real_win_rate < AIConfig.AI_REAL_WIN_RATE_THRESHOLD:
                logger.warning(f"[AutoTrainer] ⚠️ Низкий win_rate на реальных сделках: {real_win_rate:.2%} < {AIConfig.AI_REAL_WIN_RATE_THRESHOLD:.2%}")
                logger.warning(f"[AutoTrainer] 🔄 Запуск переобучения на реальных данных...")
                self._trigger_retrain_on_real_trades()
                return
            
            # Триггер 2: Отрицательный средний PnL на реальных сделках
            if real_avg_pnl < AIConfig.AI_REAL_AVG_PNL_THRESHOLD:
                logger.warning(f"[AutoTrainer] ⚠️ Отрицательный avg_pnl на реальных сделках: {real_avg_pnl:.2f} < {AIConfig.AI_REAL_AVG_PNL_THRESHOLD:.2f} USDT")
                logger.warning(f"[AutoTrainer] 🔄 Запуск переобучения на реальных данных...")
                self._trigger_retrain_on_real_trades()
                return
            
            # Триггер 3: Большая разница между виртуальными и реальными сделками
            if win_rate_diff > AIConfig.AI_REAL_VS_SIMULATED_DIFF_THRESHOLD:
                logger.warning(f"[AutoTrainer] ⚠️ Большая разница win_rate: виртуальные {sim_win_rate:.2%} vs реальные {real_win_rate:.2%} (разница: {win_rate_diff:.2%})")
                logger.warning(f"[AutoTrainer] 🔄 Запуск переобучения на реальных данных...")
                self._trigger_retrain_on_real_trades()
                return
        
        except Exception as e:
            logger.debug(f"[AutoTrainer] ⚠️ Ошибка проверки производительности на реальных сделках: {e}")
    
    def _trigger_retrain_on_real_trades(self):
        """Запускает переобучение на реальных сделках"""
        try:
            from bot_engine.ai import get_ai_system
            ai_system = get_ai_system()
            if not ai_system or not ai_system.trainer:
                logger.warning("[AutoTrainer] ⚠️ AI System недоступен для переобучения")
                return
            
            # Запускаем переобучение в отдельном потоке
            import threading
            retrain_thread = threading.Thread(
                target=ai_system.trainer.train_on_history,
                daemon=True,
                name="RetrainOnRealTrades"
            )
            retrain_thread.start()
            logger.info("[AutoTrainer] 🚀 Запущено переобучение на реальных сделках (в фоне)")
        
        except Exception as e:
            logger.error(f"[AutoTrainer] ❌ Ошибка запуска переобучения на реальных сделках: {e}")
    
    def _check_should_stop_training(self) -> bool:
        """
        Проверяет, нужно ли остановить обучение на основе качества модели
        
        Returns:
            True если обучение должно быть остановлено
        """
        if not AIConfig.AI_STOP_TRAINING_ON_HIGH_ACCURACY and not AIConfig.AI_STOP_TRAINING_ON_DEGRADATION:
            return False
        
        try:
            from bot_engine.ai.ai_database import AIDatabase
            ai_db = AIDatabase()
            
            # Получаем последние версии моделей
            models_to_check = [
                ('signal_predictor', 'signal_predictor'),
                ('profit_predictor', 'profit_predictor'),
                ('ai_decision_model', 'ai_decision_model'),
            ]
            
            max_accuracy = 0.0
            for model_name, model_type in models_to_check:
                model_version = ai_db.get_latest_model_version(model_type=model_type)
                if model_version:
                    # Проверяем accuracy или signal_accuracy
                    accuracy = model_version.get('accuracy') or model_version.get('signal_accuracy')
                    if accuracy is not None:
                        accuracy = float(accuracy)
                        max_accuracy = max(max_accuracy, accuracy)
            
            if max_accuracy == 0:
                return False  # Нет данных о качестве
            
            # Триггер 1: Высокая точность (90%+)
            if AIConfig.AI_STOP_TRAINING_ON_HIGH_ACCURACY:
                if max_accuracy >= AIConfig.AI_HIGH_ACCURACY_THRESHOLD:
                    logger.info(f"[AutoTrainer] 🎯 Достигнута высокая точность: {max_accuracy:.2%} >= {AIConfig.AI_HIGH_ACCURACY_THRESHOLD:.2%}")
                    logger.info(f"[AutoTrainer] 🛑 Остановка непрерывного обучения: модель достигла целевой точности")
                    return True
            
            # Триггер 2: Ухудшение качества
            if AIConfig.AI_STOP_TRAINING_ON_DEGRADATION and self._last_model_accuracy is not None:
                accuracy_diff = self._last_model_accuracy - max_accuracy
                if accuracy_diff >= AIConfig.AI_DEGRADATION_THRESHOLD:
                    logger.warning(f"[AutoTrainer] ⚠️ Обнаружено ухудшение качества: {accuracy_diff:.2%}")
                    logger.warning(f"[AutoTrainer] 🛑 Остановка непрерывного обучения: качество модели ухудшилось")
                    return True
            
            return False
        
        except Exception as e:
            logger.debug(f"[AutoTrainer] ⚠️ Ошибка проверки триггеров остановки: {e}")
            return False
    
    def resume_training(self):
        """
        Возобновляет обучение после остановки триггерами
        
        Сбрасывает флаг остановки и позволяет продолжить непрерывное обучение
        """
        if self._training_stopped:
            logger.info("[AutoTrainer] 🔄 Возобновление непрерывного обучения...")
            self._training_stopped = False
            # Сбрасываем счетчик попыток для новой проверки качества
            self._training_attempts = 0
            self._last_model_accuracy = None
            logger.info("[AutoTrainer] ✅ Обучение возобновлено")
        else:
            logger.info("[AutoTrainer] ℹ️ Обучение уже активно")
    
    def force_update(self) -> bool:
        """
        Принудительное обновление данных и переобучение
        
        Returns:
            True если успешно
        """
        logger.info("[AutoTrainer] 🔄 Принудительное обновление...")
        
        # Принудительное обновление сбрасывает флаг остановки
        self._training_stopped = False
        
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
            'next_training': 'continuous' if self.last_training and not self._training_stopped else None,  # Непрерывное обучение - сразу после завершения предыдущего
            'training_mode': 'continuous',
            'training_stopped': self._training_stopped,  # Остановлено ли обучение триггерами
            'training_attempts': self._training_attempts,  # Количество попыток обучения
            'last_model_accuracy': self._last_model_accuracy,  # Последняя точность модели
            'stop_triggers': {
                'high_accuracy_enabled': AIConfig.AI_STOP_TRAINING_ON_HIGH_ACCURACY,
                'high_accuracy_threshold': AIConfig.AI_HIGH_ACCURACY_THRESHOLD,
                'degradation_enabled': AIConfig.AI_STOP_TRAINING_ON_DEGRADATION,
                'degradation_threshold': AIConfig.AI_DEGRADATION_THRESHOLD,
            }
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

