/**
 * AI Config Manager - управление настройками AI модулей
 */

class AIConfigManager {
    constructor() {
        // Используем тот же способ определения URL что и в bots_manager.js
        this.BOTS_SERVICE_URL = `${window.location.protocol}//${window.location.hostname}:5001`;
        this.aiConfig = null;
        this.licenseInfo = null;
        
        console.log('[AIConfigManager] Инициализация...');
        console.log('[AIConfigManager] BOTS_SERVICE_URL:', this.BOTS_SERVICE_URL);
    }
    
    /**
     * Инициализация менеджера
     */
    async initialize() {
        try {
            console.log('[AIConfigManager] Загрузка AI конфигурации...');
            
            // Загружаем конфигурацию и проверяем лицензию
            await this.loadAIConfig();
            
            // Если лицензия валидна, показываем блок AI
            console.log('[AIConfigManager] 📊 Проверка лицензии:', this.licenseInfo);
            if (this.licenseInfo && this.licenseInfo.valid) {
                console.log('[AIConfigManager] ✅ Лицензия валидна - показываем AI блок');
                this.showAIConfigSection();
                this.bindEvents();
            } else {
                console.log('[AIConfigManager] ❌ AI недоступен (нет лицензии или невалидная лицензия)');
                this.hideAIConfigSection();
            }
        } catch (error) {
            console.error('[AIConfigManager] Ошибка инициализации:', error);
            this.hideAIConfigSection();
        }
    }
    
    /**
     * Загрузка AI конфигурации
     */
    async loadAIConfig() {
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/config`);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.aiConfig = data.config;
                this.licenseInfo = data.license;
                
                console.log('[AIConfigManager] ✅ AI конфигурация загружена');
                console.log('[AIConfigManager] Лицензия:', this.licenseInfo);
                
                // Заполняем форму
                this.populateForm();
                
                // Обновляем badge лицензии
                this.updateLicenseBadge();
            } else {
                console.error('[AIConfigManager] Ошибка загрузки конфигурации:', data.error);
            }
        } catch (error) {
            console.error('[AIConfigManager] Ошибка загрузки AI конфигурации:', error);
            throw error;
        }
    }
    
    /**
     * Заполнение формы значениями из конфигурации
     */
    populateForm() {
        if (!this.aiConfig) return;
        
        const config = this.aiConfig;
        
        // Основные настройки
        this.setCheckbox('aiEnabled', config.ai_enabled);
        
        // Anomaly Detection
        this.setCheckbox('anomalyDetectionEnabled', config.anomaly_detection_enabled);
        this.setValue('anomalyBlockThreshold', config.anomaly_block_threshold);
        this.setCheckbox('anomalyLogEnabled', config.anomaly_log_enabled);
        
        // LSTM Predictor
        this.setCheckbox('lstmEnabled', config.lstm_enabled);
        this.setValue('lstmMinConfidence', config.lstm_min_confidence);
        this.setValue('lstmWeight', config.lstm_weight);
        
        // Pattern Recognition
        this.setCheckbox('patternEnabled', config.pattern_enabled);
        this.setValue('patternMinConfidence', config.pattern_min_confidence);
        this.setValue('patternWeight', config.pattern_weight);
        
        // Risk Management
        this.setCheckbox('riskManagementEnabled', config.risk_management_enabled);
        this.setValue('riskUpdateInterval', config.risk_update_interval);
        
        // Optimal Entry Detection
        this.setCheckbox('optimalEntryEnabled', config.optimal_entry_enabled);
        
        // Auto Training
        this.setCheckbox('autoTrainEnabled', config.auto_train_enabled);
        this.setCheckbox('autoUpdateData', config.auto_update_data);
        this.setValue('dataUpdateInterval', config.data_update_interval);
        this.setCheckbox('autoRetrain', config.auto_retrain);
        this.setValue('retrainInterval', config.retrain_interval);
        this.setValue('retrainHour', config.retrain_hour);
        
        // Logging
        this.setCheckbox('logPredictions', config.log_predictions);
        this.setCheckbox('logAnomalies', config.log_anomalies);
        this.setCheckbox('logPatterns', config.log_patterns);
        
        console.log('[AIConfigManager] ✅ Форма заполнена');
    }
    
    /**
     * Сохранение AI конфигурации
     */
    async saveAIConfig() {
        try {
            console.log('[AIConfigManager] 💾 Сохранение AI конфигурации...');
            
            // Собираем данные из формы
            const configData = {
                // Основные
                ai_enabled: this.getCheckbox('aiEnabled'),
                
                // Anomaly Detection
                anomaly_detection_enabled: this.getCheckbox('anomalyDetectionEnabled'),
                anomaly_block_threshold: parseFloat(this.getValue('anomalyBlockThreshold')),
                anomaly_log_enabled: this.getCheckbox('anomalyLogEnabled'),
                
                // LSTM Predictor
                lstm_enabled: this.getCheckbox('lstmEnabled'),
                lstm_min_confidence: parseFloat(this.getValue('lstmMinConfidence')),
                lstm_weight: parseFloat(this.getValue('lstmWeight')),
                
                // Pattern Recognition
                pattern_enabled: this.getCheckbox('patternEnabled'),
                pattern_min_confidence: parseFloat(this.getValue('patternMinConfidence')),
                pattern_weight: parseFloat(this.getValue('patternWeight')),
                
                // Risk Management
                risk_management_enabled: this.getCheckbox('riskManagementEnabled'),
                risk_update_interval: parseInt(this.getValue('riskUpdateInterval')),
                
                // Optimal Entry Detection
                optimal_entry_enabled: this.getCheckbox('optimalEntryEnabled'),
                
                // Auto Training
                auto_train_enabled: this.getCheckbox('autoTrainEnabled'),
                auto_update_data: this.getCheckbox('autoUpdateData'),
                data_update_interval: parseInt(this.getValue('dataUpdateInterval')),
                auto_retrain: this.getCheckbox('autoRetrain'),
                retrain_interval: parseInt(this.getValue('retrainInterval')),
                retrain_hour: parseInt(this.getValue('retrainHour')),
                
                // Logging
                log_predictions: this.getCheckbox('logPredictions'),
                log_anomalies: this.getCheckbox('logAnomalies'),
                log_patterns: this.getCheckbox('logPatterns')
            };
            
            console.log('[AIConfigManager] Данные для сохранения:', configData);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(configData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                console.log('[AIConfigManager] ✅ AI конфигурация сохранена');
                
                // Показываем уведомление
                if (window.showToast) {
                    window.showToast('✅ AI конфигурация сохранена', 'success');
                }
                
                // Перезагружаем конфигурацию
                await this.loadAIConfig();
                
                return true;
            } else {
                console.error('[AIConfigManager] Ошибка сохранения:', data.error);
                
                if (window.showToast) {
                    window.showToast(`❌ Ошибка: ${data.error}`, 'error');
                }
                
                return false;
            }
        } catch (error) {
            console.error('[AIConfigManager] Ошибка сохранения AI конфигурации:', error);
            
            if (window.showToast) {
                window.showToast(`❌ Ошибка сохранения: ${error.message}`, 'error');
            }
            
            return false;
        }
    }
    
    /**
     * Показать блок AI конфигурации
     */
    showAIConfigSection() {
        const section = document.getElementById('aiConfigSection');
        if (section) {
            section.style.display = 'block';
            console.log('[AIConfigManager] ✅ AI блок показан');
        }
    }
    
    /**
     * Скрыть блок AI конфигурации
     */
    hideAIConfigSection() {
        const section = document.getElementById('aiConfigSection');
        if (section) {
            section.style.display = 'none';
            console.log('[AIConfigManager] ℹ️ AI блок скрыт (нет лицензии)');
        }
    }
    
    /**
     * Обновление badge лицензии
     */
    updateLicenseBadge() {
        const badge = document.getElementById('aiLicenseBadge');
        if (!badge || !this.licenseInfo) return;
        
        const isValid = this.licenseInfo.valid;
        const licenseType = this.licenseInfo.type;
        const expiresAt = this.licenseInfo.expires_at;
        
        if (isValid) {
            badge.innerHTML = `
                <span class="badge badge-success">
                    ✅ <span data-translate="license_active">Лицензия активна</span>: ${licenseType}
                    ${expiresAt !== '9999-12-31' ? ` (до ${expiresAt})` : ''}
                </span>
            `;
        } else {
            badge.innerHTML = `
                <span class="badge badge-danger">
                    ❌ <span data-translate="license_invalid">Лицензия недействительна</span>
                </span>
            `;
        }
    }
    
    /**
     * Привязка событий
     */
    bindEvents() {
        // Кнопка сохранения AI настроек
        const saveBtn = document.querySelector('.config-section-save-btn[data-section="ai"]');
        if (saveBtn) {
            saveBtn.addEventListener('click', async () => {
                await this.saveAIConfig();
            });
            console.log('[AIConfigManager] ✅ События привязаны');
        }
    }
    
    // Утилиты для работы с формой
    setCheckbox(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.checked = Boolean(value);
        }
    }
    
    getCheckbox(id) {
        const element = document.getElementById(id);
        return element ? element.checked : false;
    }
    
    setValue(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.value = value;
        }
    }
    
    getValue(id) {
        const element = document.getElementById(id);
        return element ? element.value : null;
    }
}

// Глобальный экземпляр
window.aiConfigManager = null;

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', async () => {
    try {
        window.aiConfigManager = new AIConfigManager();
        await window.aiConfigManager.initialize();
    } catch (error) {
        console.error('[AIConfigManager] Ошибка инициализации:', error);
    }
});

