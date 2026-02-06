"""
API endpoints для управления ИИ модулями
"""

from flask import jsonify, request
import logging

logger = logging.getLogger('API.AI')

# Словарь названий AI параметров для детального логирования
AI_CONFIG_NAMES = {
    # AI Master Switch
    'ai_enabled': 'AI модули включены',

    # Anomaly Detection
    'anomaly_detection_enabled': 'Обнаружение аномалий',
    'anomaly_block_threshold': 'Порог блокировки аномалий',
    'anomaly_log_enabled': 'Логирование аномалий',

    # LSTM Predictor
    'lstm_enabled': 'LSTM предсказание цены',
    'lstm_min_confidence': 'Минимальная уверенность LSTM',
    'lstm_weight': 'Вес LSTM в голосовании',

    # Pattern Recognition
    'pattern_enabled': 'Распознавание паттернов',
    'pattern_min_confidence': 'Минимальная уверенность паттернов',
    'pattern_weight': 'Вес паттернов в голосовании',

    # Risk Management
    'risk_management_enabled': 'Умное управление рисками',
    'risk_update_interval': 'Интервал обновления рисков (сек)',

    # Optimal Entry Detection
    'optimal_entry_enabled': 'Определение оптимальной точки входа',

    # Auto Training
    'auto_train_enabled': 'Автоматическое обучение',
    'auto_update_data': 'Обновление данных',
    'data_update_interval': 'Интервал обновления данных (сек)',
    'auto_retrain': 'Автоматическое переобучение',
    'retrain_interval': 'Интервал переобучения (сек)',
    'retrain_hour': 'Время переобучения (час)',

    # AI Logging
    'log_predictions': 'Логирование предсказаний',
    'log_anomalies': 'Логирование аномалий',
    'log_patterns': 'Логирование паттернов',
}

def log_ai_config_change(key, old_value, new_value):
    """Логирует изменение AI конфигурации только если значение изменилось"""
    if old_value != new_value:
        arrow = '→'
        display_name = AI_CONFIG_NAMES.get(key, key)
        logger.info(f"[AI_CONFIG] ✓ {display_name}: {old_value} {arrow} {new_value}")
        return True
    return False

def register_ai_endpoints(app):
    """Регистрирует API endpoints для ИИ"""

    logger.info("🔧 Регистрация AI endpoints...")

    @app.route('/api/ai/status', methods=['GET'])
    def get_ai_status():
        """Получить статус ИИ системы"""
        try:
            from bot_engine.ai import get_ai_manager
            from bot_engine.ai.auto_trainer import get_auto_trainer
            from bot_engine.config_loader import AIConfig, RiskConfig

            ai_manager = get_ai_manager()
            auto_trainer = get_auto_trainer()

            return jsonify({
                'success': True,
                'ai_status': ai_manager.get_status(),
                'auto_trainer': auto_trainer.get_status(),
                'config': {
                    'enabled': AIConfig.AI_ENABLED,
                    'auto_train_enabled': AIConfig.AI_AUTO_TRAIN_ENABLED,
                    'auto_update_data': AIConfig.AI_AUTO_UPDATE_DATA,
                    'auto_retrain': AIConfig.AI_AUTO_RETRAIN,
                    'data_update_interval_hours': AIConfig.AI_DATA_UPDATE_INTERVAL / 3600,
                    'retrain_interval_days': AIConfig.AI_RETRAIN_INTERVAL / 86400
                }
            })

        except Exception as e:
            logger.error(f"Ошибка получения статуса ИИ: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/ai/force-update', methods=['POST'])
    def force_ai_update():
        """Принудительное обновление данных и переобучение"""
        try:
            from bot_engine.ai.auto_trainer import get_auto_trainer

            auto_trainer = get_auto_trainer()
            success = auto_trainer.force_update()

            return jsonify({
                'success': success,
                'message': 'Обновление запущено' if success else 'Ошибка обновления'
            })

        except Exception as e:
            logger.error(f"Ошибка принудительного обновления: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    def _get_ai_config_path():
        """Абсолютный путь к configs/bot_config.py (не зависит от cwd)."""
        import os
        try:
            _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            return os.path.join(_root, 'configs', 'bot_config.py')
        except Exception:
            return os.path.abspath('configs/bot_config.py')

    @app.route('/api/ai/config', methods=['GET'])
    def get_ai_config():
        """Получить конфигурацию AI (всегда из файла — перезагружаем модуль перед ответом)."""
        try:
            from bot_engine.config_loader import reload_config
            reload_config()
            from bot_engine.config_loader import AIConfig, RiskConfig
            from bot_engine.ai import get_ai_manager

            ai_manager = get_ai_manager()

            # Проверяем доступность AI (лицензия)
            license_status = ai_manager.get_status()

            return jsonify({
                'success': True,
                'license': {
                    'valid': ai_manager.is_available(),
                    'type': license_status.get('license_type'),
                    'expires_at': license_status.get('expires_at'),
                    'features': license_status.get('features', {})
                },
                'config': {
                    # Основные настройки
                    'ai_enabled': AIConfig.AI_ENABLED,
                    'ai_confidence_threshold': AIConfig.AI_CONFIDENCE_THRESHOLD,

                    # Anomaly Detection
                    'anomaly_detection_enabled': AIConfig.AI_ANOMALY_DETECTION_ENABLED,
                    'anomaly_block_threshold': AIConfig.AI_ANOMALY_BLOCK_THRESHOLD,
                    'anomaly_log_enabled': AIConfig.AI_LOG_ANOMALIES,

                    # Risk Management
                    'risk_management_enabled': AIConfig.AI_RISK_MANAGEMENT_ENABLED,
                    'risk_update_interval': AIConfig.AI_RISK_UPDATE_INTERVAL,

                    # Optimal Entry Detection
                    'optimal_entry_enabled': RiskConfig.AI_OPTIMAL_ENTRY_ENABLED,

                    # LSTM Predictor
                    'lstm_enabled': AIConfig.AI_LSTM_ENABLED,
                    'lstm_min_confidence': AIConfig.AI_LSTM_MIN_CONFIDENCE,
                    'lstm_weight': AIConfig.AI_LSTM_WEIGHT,

                    # Pattern Recognition
                    'pattern_enabled': AIConfig.AI_PATTERN_ENABLED,
                    'pattern_min_confidence': AIConfig.AI_PATTERN_MIN_CONFIDENCE,
                    'pattern_weight': AIConfig.AI_PATTERN_WEIGHT,

                    # Auto Training
                    'auto_train_enabled': AIConfig.AI_AUTO_TRAIN_ENABLED,
                    'auto_update_data': AIConfig.AI_AUTO_UPDATE_DATA,
                    'auto_retrain': AIConfig.AI_AUTO_RETRAIN,
                    'data_update_interval': AIConfig.AI_DATA_UPDATE_INTERVAL,
                    'retrain_interval': AIConfig.AI_RETRAIN_INTERVAL,
                    'retrain_hour': AIConfig.AI_RETRAIN_HOUR,
                    'update_coins_count': AIConfig.AI_UPDATE_COINS_COUNT,

                    # Logging
                    'log_predictions': AIConfig.AI_LOG_PREDICTIONS,
                    'log_anomalies': AIConfig.AI_LOG_ANOMALIES,
                    'log_patterns': AIConfig.AI_LOG_PATTERNS,

                    # Самообучение в реальном времени
                    'self_learning_enabled': AIConfig.AI_SELF_LEARNING_ENABLED,

                    # Smart Money Concepts (можно выключить на минутках)
                    'smc_enabled': getattr(AIConfig, 'AI_SMC_ENABLED', True),
                }
            })

        except Exception as e:
            logger.error(f"Ошибка получения AI конфигурации: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/ai/config', methods=['POST'])
    def save_ai_config():
        """Сохранить конфигурацию AI (разрешено даже при неактивной лицензии — сохраняем только настройки в файл)."""
        try:
            # Сохранение конфига в файл разрешено всегда; проверку лицензии не делаем.

            # Проверяем формат данных
            content_type = request.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                return jsonify({
                    'success': False,
                    'error': f'Требуется Content-Type: application/json, получен: {content_type}'
                }), 415

            try:
                data = request.get_json(force=True)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Ошибка парсинга JSON: {str(e)}'
                }), 400

            if not data:
                return jsonify({
                    'success': False,
                    'error': 'Пустые данные'
                }), 400

            logger.info(f"[AI_CONFIG] Получены данные: {data}")

            # Нормализуем булевы значения (чтобы в файл всегда писались True/False)
            _BOOL_KEYS = (
                'ai_enabled', 'anomaly_detection_enabled', 'anomaly_log_enabled',
                'lstm_enabled', 'pattern_enabled', 'risk_management_enabled',
                'optimal_entry_enabled', 'auto_train_enabled', 'auto_update_data', 'auto_retrain',
                'log_predictions', 'log_anomalies', 'log_patterns',
                'self_learning_enabled', 'smc_enabled'
            )
            for k in _BOOL_KEYS:
                if k in data and data[k] is not None:
                    data[k] = bool(data[k])

            # Мастер-переключатель «Включить AI модули» выключает все AI-настройки
            if data.get('ai_enabled') is False:
                _AI_CHILD_FLAGS = (
                    'anomaly_detection_enabled', 'anomaly_log_enabled', 'lstm_enabled', 'pattern_enabled',
                    'risk_management_enabled', 'optimal_entry_enabled', 'auto_train_enabled', 'auto_update_data',
                    'auto_retrain', 'log_predictions', 'log_anomalies', 'log_patterns',
                    'self_learning_enabled', 'smc_enabled'
                )
                for k in _AI_CHILD_FLAGS:
                    data[k] = False
                logger.info("[AI_CONFIG] ai_enabled=False → все дочерние AI флаги принудительно выключены")

            # Получаем текущие значения для сравнения
            from bot_engine.config_loader import AIConfig, RiskConfig
            old_config = {
                'ai_enabled': AIConfig.AI_ENABLED,
                'ai_confidence_threshold': AIConfig.AI_CONFIDENCE_THRESHOLD,
                'anomaly_detection_enabled': AIConfig.AI_ANOMALY_DETECTION_ENABLED,
                'anomaly_block_threshold': AIConfig.AI_ANOMALY_BLOCK_THRESHOLD,
                'lstm_enabled': AIConfig.AI_LSTM_ENABLED,
                'lstm_min_confidence': AIConfig.AI_LSTM_MIN_CONFIDENCE,
                'lstm_weight': AIConfig.AI_LSTM_WEIGHT,
                'pattern_enabled': AIConfig.AI_PATTERN_ENABLED,
                'pattern_min_confidence': AIConfig.AI_PATTERN_MIN_CONFIDENCE,
                'pattern_weight': AIConfig.AI_PATTERN_WEIGHT,
                'risk_management_enabled': AIConfig.AI_RISK_MANAGEMENT_ENABLED,
                'risk_update_interval': AIConfig.AI_RISK_UPDATE_INTERVAL,
                'optimal_entry_enabled': RiskConfig.AI_OPTIMAL_ENTRY_ENABLED,
                'auto_train_enabled': AIConfig.AI_AUTO_TRAIN_ENABLED,
                'auto_update_data': AIConfig.AI_AUTO_UPDATE_DATA,
                'auto_retrain': AIConfig.AI_AUTO_RETRAIN,
                'data_update_interval': AIConfig.AI_DATA_UPDATE_INTERVAL,
                'retrain_interval': AIConfig.AI_RETRAIN_INTERVAL,
                'retrain_hour': AIConfig.AI_RETRAIN_HOUR,
                'log_predictions': AIConfig.AI_LOG_PREDICTIONS,
                'log_anomalies': AIConfig.AI_LOG_ANOMALIES,
                'log_patterns': AIConfig.AI_LOG_PATTERNS,
                'self_learning_enabled': AIConfig.AI_SELF_LEARNING_ENABLED,
                'smc_enabled': getattr(AIConfig, 'AI_SMC_ENABLED', True),
            }

            # Читаем текущий файл bot_config.py (абсолютный путь — не зависит от cwd)
            config_path = _get_ai_config_path()

            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Находим блок AIConfig и RiskConfig и обновляем значения
            in_ai_config = False
            in_risk_config = False
            updated_lines = []
            changes_count = 0

            for line in lines:
                # Определяем начало и конец AIConfig
                if 'class AIConfig:' in line:
                    in_ai_config = True
                    in_risk_config = False
                elif 'class RiskConfig:' in line:
                    in_ai_config = False
                    in_risk_config = True
                elif (in_ai_config or in_risk_config) and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    in_ai_config = False
                    in_risk_config = False

                # Обновляем значения в AIConfig
                if in_ai_config:
                    # Основные настройки
                    if 'AI_ENABLED =' in line and 'ai_enabled' in data:
                        if log_ai_config_change('ai_enabled', old_config['ai_enabled'], data['ai_enabled']):
                            changes_count += 1
                        line = f"    AI_ENABLED = {data['ai_enabled']}\n"
                    elif 'AI_CONFIDENCE_THRESHOLD =' in line and 'ai_confidence_threshold' in data:
                        if log_ai_config_change('ai_confidence_threshold', old_config['ai_confidence_threshold'], data['ai_confidence_threshold']):
                            changes_count += 1
                        line = f"    AI_CONFIDENCE_THRESHOLD = {data['ai_confidence_threshold']}\n"

                    # Anomaly Detection
                    elif 'AI_ANOMALY_DETECTION_ENABLED =' in line and 'anomaly_detection_enabled' in data:
                        if log_ai_config_change('anomaly_detection_enabled', old_config['anomaly_detection_enabled'], data['anomaly_detection_enabled']):
                            changes_count += 1
                        line = f"    AI_ANOMALY_DETECTION_ENABLED = {data['anomaly_detection_enabled']}\n"
                    elif 'AI_ANOMALY_BLOCK_THRESHOLD =' in line and 'anomaly_block_threshold' in data:
                        if log_ai_config_change('anomaly_block_threshold', old_config['anomaly_block_threshold'], data['anomaly_block_threshold']):
                            changes_count += 1
                        line = f"    AI_ANOMALY_BLOCK_THRESHOLD = {data['anomaly_block_threshold']}\n"

                    # Risk Management
                    elif 'AI_RISK_MANAGEMENT_ENABLED =' in line and 'risk_management_enabled' in data:
                        if log_ai_config_change('risk_management_enabled', old_config['risk_management_enabled'], data['risk_management_enabled']):
                            changes_count += 1
                        line = f"    AI_RISK_MANAGEMENT_ENABLED = {data['risk_management_enabled']}\n"
                    elif 'AI_RISK_UPDATE_INTERVAL =' in line and 'risk_update_interval' in data:
                        if log_ai_config_change('risk_update_interval', old_config['risk_update_interval'], data['risk_update_interval']):
                            changes_count += 1
                        line = f"    AI_RISK_UPDATE_INTERVAL = {data['risk_update_interval']}\n"

                    # Auto Training
                    elif 'AI_AUTO_TRAIN_ENABLED =' in line and 'auto_train_enabled' in data:
                        if log_ai_config_change('auto_train_enabled', old_config['auto_train_enabled'], data['auto_train_enabled']):
                            changes_count += 1
                        line = f"    AI_AUTO_TRAIN_ENABLED = {data['auto_train_enabled']}\n"
                    elif 'AI_AUTO_UPDATE_DATA =' in line and 'auto_update_data' in data:
                        if log_ai_config_change('auto_update_data', old_config['auto_update_data'], data['auto_update_data']):
                            changes_count += 1
                        line = f"    AI_AUTO_UPDATE_DATA = {data['auto_update_data']}\n"
                    elif 'AI_AUTO_RETRAIN =' in line and 'auto_retrain' in data:
                        if log_ai_config_change('auto_retrain', old_config['auto_retrain'], data['auto_retrain']):
                            changes_count += 1
                        line = f"    AI_AUTO_RETRAIN = {data['auto_retrain']}\n"
                    elif 'AI_DATA_UPDATE_INTERVAL =' in line and 'data_update_interval' in data:
                        if log_ai_config_change('data_update_interval', old_config['data_update_interval'], data['data_update_interval']):
                            changes_count += 1
                        line = f"    AI_DATA_UPDATE_INTERVAL = {data['data_update_interval']}\n"
                    elif 'AI_RETRAIN_INTERVAL =' in line and 'retrain_interval' in data:
                        if log_ai_config_change('retrain_interval', old_config['retrain_interval'], data['retrain_interval']):
                            changes_count += 1
                        line = f"    AI_RETRAIN_INTERVAL = {data['retrain_interval']}\n"
                    elif 'AI_RETRAIN_HOUR =' in line and 'retrain_hour' in data:
                        if log_ai_config_change('retrain_hour', old_config['retrain_hour'], data['retrain_hour']):
                            changes_count += 1
                        line = f"    AI_RETRAIN_HOUR = {data['retrain_hour']}\n"

                    # LSTM Predictor
                    elif 'AI_LSTM_ENABLED =' in line and 'lstm_enabled' in data:
                        old_value = old_config.get('lstm_enabled', AIConfig.AI_LSTM_ENABLED)
                        if log_ai_config_change('lstm_enabled', old_value, data['lstm_enabled']):
                            changes_count += 1
                        line = f"    AI_LSTM_ENABLED = {data['lstm_enabled']}\n"
                    elif 'AI_LSTM_MIN_CONFIDENCE =' in line and 'lstm_min_confidence' in data:
                        old_value = old_config.get('lstm_min_confidence', AIConfig.AI_LSTM_MIN_CONFIDENCE)
                        if log_ai_config_change('lstm_min_confidence', old_value, data['lstm_min_confidence']):
                            changes_count += 1
                        line = f"    AI_LSTM_MIN_CONFIDENCE = {data['lstm_min_confidence']}\n"
                    elif 'AI_LSTM_WEIGHT =' in line and 'lstm_weight' in data:
                        old_value = old_config.get('lstm_weight', AIConfig.AI_LSTM_WEIGHT)
                        if log_ai_config_change('lstm_weight', old_value, data['lstm_weight']):
                            changes_count += 1
                        line = f"    AI_LSTM_WEIGHT = {data['lstm_weight']}\n"

                    # Pattern Recognition
                    elif 'AI_PATTERN_ENABLED =' in line and 'pattern_enabled' in data:
                        old_value = old_config.get('pattern_enabled', AIConfig.AI_PATTERN_ENABLED)
                        if log_ai_config_change('pattern_enabled', old_value, data['pattern_enabled']):
                            changes_count += 1
                        line = f"    AI_PATTERN_ENABLED = {data['pattern_enabled']}\n"
                    elif 'AI_PATTERN_MIN_CONFIDENCE =' in line and 'pattern_min_confidence' in data:
                        old_value = old_config.get('pattern_min_confidence', AIConfig.AI_PATTERN_MIN_CONFIDENCE)
                        if log_ai_config_change('pattern_min_confidence', old_value, data['pattern_min_confidence']):
                            changes_count += 1
                        line = f"    AI_PATTERN_MIN_CONFIDENCE = {data['pattern_min_confidence']}\n"
                    elif 'AI_PATTERN_WEIGHT =' in line and 'pattern_weight' in data:
                        old_value = old_config.get('pattern_weight', AIConfig.AI_PATTERN_WEIGHT)
                        if log_ai_config_change('pattern_weight', old_value, data['pattern_weight']):
                            changes_count += 1
                        line = f"    AI_PATTERN_WEIGHT = {data['pattern_weight']}\n"

                    # Logging
                    elif 'AI_LOG_PREDICTIONS =' in line and 'log_predictions' in data:
                        if log_ai_config_change('log_predictions', old_config['log_predictions'], data['log_predictions']):
                            changes_count += 1
                        line = f"    AI_LOG_PREDICTIONS = {data['log_predictions']}\n"
                    elif 'AI_LOG_ANOMALIES =' in line and ('log_anomalies' in data or 'anomaly_log_enabled' in data):
                        # Одна настройка — два чекбокса в UI (anomalyLogEnabled и logAnomalies); учитываем оба
                        value = data.get('log_anomalies', False) or data.get('anomaly_log_enabled', False)
                        if log_ai_config_change('log_anomalies', old_config['log_anomalies'], value):
                            changes_count += 1
                        line = f"    AI_LOG_ANOMALIES = {value}\n"
                    elif 'AI_LOG_PATTERNS =' in line and 'log_patterns' in data:
                        old_value = old_config.get('log_patterns', AIConfig.AI_LOG_PATTERNS)
                        if log_ai_config_change('log_patterns', old_value, data['log_patterns']):
                            changes_count += 1
                        line = f"    AI_LOG_PATTERNS = {data['log_patterns']}\n"

                    # Самообучение в реальном времени
                    elif 'AI_SELF_LEARNING_ENABLED =' in line and 'self_learning_enabled' in data:
                        if log_ai_config_change('self_learning_enabled', old_config['self_learning_enabled'], data['self_learning_enabled']):
                            changes_count += 1
                        line = f"    AI_SELF_LEARNING_ENABLED = {data['self_learning_enabled']}\n"

                    # Smart Money Concepts (вкл/выкл — на минутках может мешать)
                    elif 'AI_SMC_ENABLED =' in line and 'smc_enabled' in data:
                        if log_ai_config_change('smc_enabled', old_config.get('smc_enabled', True), data['smc_enabled']):
                            changes_count += 1
                        line = f"    AI_SMC_ENABLED = {data['smc_enabled']}\n"

                # Обновляем значения в RiskConfig
                if in_risk_config and 'AI_OPTIMAL_ENTRY_ENABLED =' in line and 'optimal_entry_enabled' in data:

                    if log_ai_config_change('optimal_entry_enabled', old_config['optimal_entry_enabled'], data['optimal_entry_enabled']):
                        changes_count += 1
                    line = f"    AI_OPTIMAL_ENTRY_ENABLED = {data['optimal_entry_enabled']}\n"

                updated_lines.append(line)

            # Сохраняем изменения
            with open(config_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            from bot_engine.config_loader import reload_config
            reload_config()
            # Выводим итоговое сообщение
            if changes_count > 0:
                logger.info(f"[AI_CONFIG] ✅ AI модули: изменено параметров: {changes_count}, конфигурация сохранена и перезагружена")
            else:
                logger.info("[AI_CONFIG] ℹ️ AI модули: изменений не обнаружено")

            return jsonify({
                'success': True,
                'message': f'AI конфигурация сохранена ({changes_count} изменений)'
            })

        except Exception as e:
            logger.error(f"Ошибка сохранения AI конфигурации: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/ai/anomaly-stats', methods=['GET'])
    def get_anomaly_stats():
        """Получить статистику обнаруженных аномалий"""
        try:
            # TODO: Реализовать сбор статистики аномалий
            # Можно хранить в отдельном файле или БД

            return jsonify({
                'success': True,
                'stats': {
                    'total_analyzed': 0,
                    'anomalies_detected': 0,
                    'blocked_entries': 0,
                    'by_type': {
                        'PUMP': 0,
                        'DUMP': 0,
                        'MANIPULATION': 0
                    }
                }
            })

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/ai/performance', methods=['GET'])
    def get_ai_performance():
        """Получить метрики производительности AI моделей"""
        try:
            from bot_engine.ai.monitoring import get_performance_api_data

            data = get_performance_api_data()
            return jsonify({
                'success': True,
                'performance': data
            })

        except ImportError:
            # Модуль мониторинга не загружен
            return jsonify({
                'success': True,
                'performance': {
                    'available': False,
                    'message': 'AI мониторинг не активирован'
                }
            })
        except Exception as e:
            logger.error(f"Ошибка получения AI performance: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/ai/health', methods=['GET'])
    def get_ai_health():
        """Получить состояние здоровья AI моделей"""
        try:
            from bot_engine.ai.monitoring import get_health_api_data

            data = get_health_api_data()
            return jsonify({
                'success': True,
                'health': data
            })

        except ImportError:
            return jsonify({
                'success': True,
                'health': {
                    'available': False,
                    'message': 'AI мониторинг не активирован'
                }
            })
        except Exception as e:
            logger.error(f"Ошибка получения AI health: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/ai/experiments', methods=['GET'])
    def get_ai_experiments():
        """Получить историю экспериментов обучения"""
        try:
            from bot_engine.ai.auto_trainer import get_experiment_tracker

            limit = request.args.get('limit', 10, type=int)
            tracker = get_experiment_tracker()
            runs = tracker.get_runs_history(limit=limit)

            return jsonify({
                'success': True,
                'experiments': runs,
                'total': len(runs)
            })

        except ImportError:
            return jsonify({
                'success': True,
                'experiments': [],
                'message': 'Трекер экспериментов не активирован'
            })
        except Exception as e:
            logger.error(f"Ошибка получения экспериментов: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/ai/experiments/best', methods=['GET'])
    def get_best_experiment():
        """Получить лучший эксперимент по метрике"""
        try:
            from bot_engine.ai.auto_trainer import get_experiment_tracker

            metric = request.args.get('metric', 'accuracy')
            maximize = request.args.get('maximize', 'true').lower() == 'true'

            tracker = get_experiment_tracker()
            best = tracker.get_best_run(metric=metric, maximize=maximize)

            return jsonify({
                'success': True,
                'best_run': best
            })

        except Exception as e:
            logger.error(f"Ошибка получения лучшего эксперимента: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/ai/smc/signal', methods=['POST'])
    def get_smc_signal():
        """Получить SMC сигнал для символа"""
        try:
            from bot_engine.ai.ai_integration import get_smc_signal as _get_smc_signal

            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'Требуются данные: candles, current_price'
                }), 400

            candles = data.get('candles', [])
            current_price = data.get('current_price', 0)

            if not candles or not current_price:
                return jsonify({
                    'success': False,
                    'error': 'Требуются candles и current_price'
                }), 400

            signal = _get_smc_signal(candles, current_price)

            return jsonify({
                'success': True,
                'signal': signal
            })

        except Exception as e:
            logger.error(f"Ошибка получения SMC сигнала: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    logger.info("[API] ✅ AI endpoints зарегистрированы")
