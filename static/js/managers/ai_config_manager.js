/**
 * AI Config Manager - управление настройками AI модулей
 */

class AIConfigManager {
    constructor() {
        // Используем тот же способ определения URL что и в bots_manager.js
        this.BOTS_SERVICE_URL = `${window.location.protocol}//${window.location.hostname}:5001`;
        this.aiConfig = null;
        this.licenseInfo = null;
        // Автосохранение при изменении (как на странице конфига Auto Bot)
        this.autoSaveTimer = null;
        this.autoSaveDelay = 2000;
        this.isProgrammaticChange = false;
        
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
            
            // Блок AI всегда виден; при валидной лицензии — включаем сохранение и бейдж
            console.log('[AIConfigManager] 📊 Проверка лицензии:', this.licenseInfo);
            this.showAIConfigSection(); // всегда показываем секцию, чтобы настройки были видны
            this.bindEvents();
            if (this.licenseInfo && this.licenseInfo.valid) {
                console.log('[AIConfigManager] ✅ Лицензия валидна');
            } else {
                console.log('[AIConfigManager] ⚠️ Лицензия не активна — запустите bots.py и активируйте премиум');
            }
        } catch (error) {
            console.error('[AIConfigManager] Ошибка инициализации:', error);
            this.showAIConfigSection(); // при ошибке (например сервис недоступен) всё равно показываем блок
            this.licenseInfo = null;
            this.updateLicenseBadge(); // показать «Запустите bots.py»
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
                
                // Заполняем форму (без срабатывания автосохранения)
                this.isProgrammaticChange = true;
                this.populateForm();
                this.isProgrammaticChange = false;
                
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
        
        const masterOn = Boolean(config.ai_enabled);
        const childCheckboxIds = this.getChildAICheckboxIds();
        
        if (!masterOn) {
            // Мастер выключен — в UI показываем все дочерние как выключенные
            childCheckboxIds.forEach(id => this.setCheckbox(id, false));
            this.setChildAIInputsEnabled(false);
        } else {
            this.setChildAIInputsEnabled(true);
            // Anomaly Detection (логирование аномалий — только в блоке «Логирование AI», без дубликата)
            this.setCheckbox('anomalyDetectionEnabled', config.anomaly_detection_enabled);
            this.setValue('anomalyBlockThreshold', config.anomaly_block_threshold);
            
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
            // Самообучение AI
            this.setCheckbox('selfLearningEnabled', config.self_learning_enabled);
            // Smart Money Concepts
            this.setCheckbox('smcEnabled', config.smc_enabled !== false);
            this.updateSmcStatusText();
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
        }
        
        if (masterOn) {
            this.updateSmcStatusText();
        }
        
        console.log('[AIConfigManager] ✅ Форма заполнена');
    }
    
    /** ID чекбоксов дочерних AI-настроек (все выключаются при выключении мастер-переключателя) */
    getChildAICheckboxIds() {
        return [
            'anomalyDetectionEnabled', 'lstmEnabled', 'patternEnabled',
            'riskManagementEnabled', 'optimalEntryEnabled', 'selfLearningEnabled', 'smcEnabled',
            'autoTrainEnabled', 'autoUpdateData', 'autoRetrain', 'logPredictions', 'logAnomalies', 'logPatterns'
        ];
    }
    
    /** Включить/выключить поля ввода в блоке AI (кроме мастер-переключателя) */
    setChildAIInputsEnabled(enabled) {
        const section = document.getElementById('aiConfigSection');
        if (!section) return;
        const inputs = section.querySelectorAll('input:not(#aiEnabled), select');
        inputs.forEach(el => { el.disabled = !enabled; });
    }
    
    /**
     * Сохранение AI конфигурации
     * @param {boolean} isAutoSave - true при автосохранении (другое уведомление)
     * @param {boolean} skipNotification - true при вызове из saveAllConfiguration (уведомление покажет вызывающий)
     */
    async saveAIConfig(isAutoSave = false, skipNotification = false) {
        try {
            if (!isAutoSave && this.autoSaveTimer) {
                clearTimeout(this.autoSaveTimer);
                this.autoSaveTimer = null;
            }
            console.log('[AIConfigManager] 💾 Сохранение AI конфигурации' + (isAutoSave ? ' (авто)' : '') + '...');
            
            const masterOn = this.getCheckbox('aiEnabled');
            // При выключенном мастер-переключателе все AI-флаги отправляем как false
            const configData = {
                ai_enabled: masterOn,
                anomaly_detection_enabled: masterOn && this.getCheckbox('anomalyDetectionEnabled'),
                anomaly_block_threshold: parseFloat(this.getValue('anomalyBlockThreshold')),
                anomaly_log_enabled: masterOn && this.getCheckbox('logAnomalies'),
                lstm_enabled: masterOn && this.getCheckbox('lstmEnabled'),
                lstm_min_confidence: parseFloat(this.getValue('lstmMinConfidence')),
                lstm_weight: parseFloat(this.getValue('lstmWeight')),
                pattern_enabled: masterOn && this.getCheckbox('patternEnabled'),
                pattern_min_confidence: parseFloat(this.getValue('patternMinConfidence')),
                pattern_weight: parseFloat(this.getValue('patternWeight')),
                risk_management_enabled: masterOn && this.getCheckbox('riskManagementEnabled'),
                risk_update_interval: parseInt(this.getValue('riskUpdateInterval')),
                optimal_entry_enabled: masterOn && this.getCheckbox('optimalEntryEnabled'),
                self_learning_enabled: masterOn && this.getCheckbox('selfLearningEnabled'),
                smc_enabled: masterOn && this.getCheckbox('smcEnabled'),
                auto_train_enabled: masterOn && this.getCheckbox('autoTrainEnabled'),
                auto_update_data: masterOn && this.getCheckbox('autoUpdateData'),
                data_update_interval: parseInt(this.getValue('dataUpdateInterval')),
                auto_retrain: masterOn && this.getCheckbox('autoRetrain'),
                retrain_interval: parseInt(this.getValue('retrainInterval')),
                retrain_hour: parseInt(this.getValue('retrainHour')),
                log_predictions: masterOn && this.getCheckbox('logPredictions'),
                log_anomalies: masterOn && this.getCheckbox('logAnomalies'),
                log_patterns: masterOn && this.getCheckbox('logPatterns')
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
                const text = await response.text();
                if (response.status === 403) {
                    throw new Error('Сохранение AI настроек заблокировано (403). Перезапустите сервис ботов (bots.py) и обновите страницу.');
                }
                let errMsg = `HTTP ${response.status}`;
                try {
                    const j = JSON.parse(text);
                    if (j && j.error) errMsg = j.error;
                } catch (_) {}
                throw new Error(errMsg);
            }
            
            const data = await response.json();
            
            if (data.success) {
                console.log('[AIConfigManager] ✅ AI конфигурация сохранена');
                
                if (!skipNotification) {
                    if (!isAutoSave && window.showToast) {
                        window.showToast('✅ AI конфигурация сохранена', 'success');
                    }
                    if (isAutoSave && window.toastManager) {
                        if (!window.toastManager.container) window.toastManager.init();
                        window.toastManager.success('✅ AI настройки автоматически сохранены', 3000);
                    } else if (isAutoSave && window.showToast) {
                        window.showToast('✅ AI настройки автоматически сохранены', 'success');
                    }
                }
                
                if (!isAutoSave) {
                    this.isProgrammaticChange = true;
                    await this.loadAIConfig();
                    this.isProgrammaticChange = false;
                }
                
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
        const blockTitle = document.getElementById('aiConfigBlockTitle');
        if (section) section.style.display = 'block';
        if (blockTitle) blockTitle.style.display = 'block';
        console.log('[AIConfigManager] ✅ AI блок показан');
        this.loadSelfLearningOnShow();
    }
    
    /**
     * Скрыть блок AI конфигурации
     */
    hideAIConfigSection() {
        const section = document.getElementById('aiConfigSection');
        const blockTitle = document.getElementById('aiConfigBlockTitle');
        if (section) section.style.display = 'none';
        if (blockTitle) blockTitle.style.display = 'none';
        console.log('[AIConfigManager] ℹ️ AI блок скрыт (нет лицензии)');
    }
    
    /**
     * Обновление badge лицензии
     */
    updateLicenseBadge() {
        const badge = document.getElementById('aiLicenseBadge');
        if (!badge) return;
        if (!this.licenseInfo) {
            badge.innerHTML = `<span class="badge badge-warning">⚠️ Запустите bots.py для проверки лицензии</span>`;
            return;
        }
        
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
            const reason = this.licenseInfo.reason || '';
            badge.innerHTML = `
                <span class="badge badge-danger">
                    ❌ <span data-translate="license_invalid">Лицензия недействительна</span>
                    ${reason ? `<br><small style="display:block;margin-top:6px;opacity:0.9;">${reason}</small>` : ''}
                </span>
            `;
        }
    }
    
    /**
     * Обновить подпись статуса SMC (активен/выключен)
     */
    updateSmcStatusText() {
        const statusText = document.getElementById('smcStatusText');
        const indicator = document.querySelector('#smcStatus .status-indicator');
        const checkbox = document.getElementById('smcEnabled');
        if (!statusText || !checkbox) return;
        const enabled = checkbox.checked;
        statusText.textContent = enabled ? 'SMC модуль активен' : 'SMC модуль выключен';
        if (indicator) indicator.textContent = enabled ? '✅' : '❌';
    }

    /**
     * Планирует автосохранение AI конфигурации (debounce), как на странице конфига Auto Bot
     */
    scheduleAutoSave() {
        if (this.isProgrammaticChange) return;
        const self = this;
        if (this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
            this.autoSaveTimer = null;
        }
        this.autoSaveTimer = setTimeout(async () => {
            try {
                await self.saveAIConfig(true);
                self.autoSaveTimer = null;
            } catch (e) {
                console.error('[AIConfigManager] Ошибка автосохранения:', e);
                if (window.toastManager) {
                    window.toastManager.error('Ошибка автосохранения: ' + e.message, 5000);
                }
                self.autoSaveTimer = null;
            }
        }, this.autoSaveDelay);
    }

    /**
     * Привязка событий
     */
    bindEvents() {
        // Мастер-переключатель «Включить AI модули»: при выключении — сброс всех дочерних и блокировка полей
        const masterToggle = document.getElementById('aiEnabled');
        if (masterToggle) {
            masterToggle.addEventListener('change', () => {
                if (this.isProgrammaticChange) return;
                const masterOn = masterToggle.checked;
                this.setChildAIInputsEnabled(masterOn);
                this.isProgrammaticChange = true;
                if (!masterOn) {
                    this.getChildAICheckboxIds().forEach(id => this.setCheckbox(id, false));
                } else {
                    // Мастер включён — включаем все дочерние ползунки
                    this.getChildAICheckboxIds().forEach(id => this.setCheckbox(id, true));
                }
                this.updateSmcStatusText();
                this.isProgrammaticChange = false;
                if (window.botsManager) {
                    window.botsManager.scheduleToggleAutoSave(masterToggle);
                }
            });
        }

        // Кнопка сохранения AI настроек
        const saveBtn = document.querySelector('.config-section-save-btn[data-section="ai"]');
        if (saveBtn) {
            saveBtn.addEventListener('click', async () => {
                await this.saveAIConfig(false);
            });
            console.log('[AIConfigManager] ✅ События привязаны');
        }

        // SMC: обновлять подпись при переключении
        const smcCheckbox = document.getElementById('smcEnabled');
        if (smcCheckbox) {
            smcCheckbox.addEventListener('change', () => {
                this.updateSmcStatusText();
                if (window.botsManager) {
                    window.botsManager.scheduleToggleAutoSave(smcCheckbox);
                }
            });
        }

        // Автосохранение при изменении любых полей в блоке AI конфигурации
        const section = document.getElementById('aiConfigSection');
        if (section) {
            const inputs = section.querySelectorAll('input, select');
            inputs.forEach(el => {
                if (el.id === 'smcEnabled') return; // уже обработан выше
                if (el.getAttribute('data-autosave-bound')) return;
                el.setAttribute('data-autosave-bound', 'true');
                el.addEventListener('change', () => {
                    if (!this.isProgrammaticChange && window.botsManager) {
                        if (el.type === 'checkbox' || el.tagName === 'SELECT') {
                            window.botsManager.scheduleToggleAutoSave(el);
                        } else {
                            window.botsManager.aiConfigDirty = true;
                            window.botsManager.updateFloatingSaveButtonVisibility();
                        }
                    }
                });
                el.addEventListener('input', () => {
                    if (!this.isProgrammaticChange && window.botsManager) {
                        window.botsManager.aiConfigDirty = true;
                        window.botsManager.updateFloatingSaveButtonVisibility();
                    }
                });
            });
            console.log('[AIConfigManager] ✅ Автосохранение при изменении полей включено');
        }

        // События для самообучения AI
        this.bindSelfLearningEvents();
        
        // События для AI Performance
        this.bindPerformanceRefreshEvent();
    }

    /**
     * Привязка событий для самообучения AI
     */
    bindSelfLearningEvents() {
        const refreshBtn = document.getElementById('refreshSelfLearningBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadSelfLearningResults();
            });
            console.log('[AIConfigManager] ✅ События самообучения привязаны');
        }
    }

    /**
     * Загрузка результатов самообучения AI
     */
    async loadSelfLearningResults() {
        try {
            console.log('[AIConfigManager] 📊 Загрузка результатов самообучения...');

            const resultsContent = document.getElementById('selfLearningResultsContent');
            if (!resultsContent) return;

            // Показываем загрузку
            resultsContent.innerHTML = `
                <div class="loading-results">
                    <div class="spinner-border spinner-border-sm" role="status">
                        <span class="sr-only">Загрузка...</span>
                    </div>
                    <span>Загрузка результатов...</span>
                </div>
            `;

            // Загружаем статистику
            const statsResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/self-learning/stats`);
            const statsData = await statsResponse.json();

            // Загружаем метрики производительности
            const perfResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/self-learning/performance`);
            const perfData = await perfResponse.json();

            if (statsData.success && perfData.success) {
                if (statsData.license_required || perfData.license_required) {
                    this.displaySelfLearningPremiumRequired(statsData.message || perfData.message || 'Доступно с премиум лицензией');
                } else {
                    this.displaySelfLearningResults(statsData.stats, perfData.performance, perfData.trends);
                }
                console.log('[AIConfigManager] ✅ Результаты самообучения загружены');
            } else {
                const errorMsg = statsData.error || perfData.error || 'Неизвестная ошибка';
                this.displaySelfLearningError(errorMsg);
            }

        } catch (error) {
            console.error('[AIConfigManager] Ошибка загрузки результатов самообучения:', error);
            this.displaySelfLearningError('Ошибка загрузки данных');
        }
    }

    /**
     * Отображение результатов самообучения
     */
    displaySelfLearningResults(stats, performance, trends) {
        const resultsContent = document.getElementById('selfLearningResultsContent');
        if (!resultsContent) return;

        const statsData = stats.stats || {};

        let html = '';

        // Метрики производительности
        if (performance && !performance.error) {
            const aiWinRate = performance.ai_win_rate ?? 0;
            const nonAiWinRate = performance.non_ai_win_rate ?? null;
            const winRateDiff = performance.win_rate_difference ?? 0;
            const aiAvgPnl = performance.ai_avg_pnl ?? 0;
            const nonAiAvgPnl = performance.non_ai_avg_pnl ?? null;
            const avgPnlDiff = performance.avg_pnl_difference ?? 0;

            html += `
                <div class="self-learning-metrics">
                    <div class="metric-card">
                        <h6>Win Rate AI</h6>
                        <div class="metric-value ${aiWinRate > 0.6 ? 'positive' : aiWinRate > 0.5 ? '' : 'negative'}">
                            ${(aiWinRate * 100).toFixed(1)}%
                        </div>
                        ${nonAiWinRate !== null ? `
                            <div class="metric-trend ${winRateDiff > 0 ? 'positive' : 'negative'}">
                                vs ${(nonAiWinRate * 100).toFixed(1)}% (без AI)
                            </div>
                        ` : ''}
                    </div>

                    <div class="metric-card">
                        <h6>Avg PnL AI</h6>
                        <div class="metric-value ${aiAvgPnl > 0 ? 'positive' : 'negative'}">
                            $${aiAvgPnl.toFixed(2)}
                        </div>
                        ${nonAiAvgPnl !== null ? `
                            <div class="metric-trend ${avgPnlDiff > 0 ? 'positive' : 'negative'}">
                                vs $${nonAiAvgPnl.toFixed(2)} (без AI)
                            </div>
                        ` : ''}
                    </div>

                    <div class="metric-card">
                        <h6>Рейтинг AI</h6>
                        <div class="metric-value">
                            ${performance.ai_performance_score || 0}/3
                        </div>
                        <div class="metric-trend">
                            ${performance.ai_performance_rating || 'Не оценено'}
                        </div>
                    </div>

                    <div class="metric-card">
                        <h6>Обработано сделок</h6>
                        <div class="metric-value">
                            ${statsData.total_trades_processed || 0}
                        </div>
                        <div class="metric-trend">
                            Онлайн обновлений: ${statsData.online_updates || 0}
                        </div>
                    </div>
                </div>
            `;
        }

        // Статистика самообучения
        html += `
            <div class="self-learning-stats">
                <h6>📈 Статистика самообучения</h6>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-label">Онлайн обучение:</span>
                        <span class="stat-value">${stats.online_learning_enabled ? '✅ Включено' : '❌ Выключено'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Адаптация к рынку:</span>
                        <span class="stat-value">${stats.adaptive_learning_enabled ? '✅ Включено' : '❌ Выключено'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Успешных адаптаций:</span>
                        <span class="stat-value">${statsData.successful_adaptations || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Размер буфера:</span>
                        <span class="stat-value">${stats.buffer_size || 0} сделок</span>
                    </div>
                </div>
            </div>
        `;

        // Тренды производительности
        if (trends && !trends.error) {
            html += `
                <div class="performance-trends">
                    <h6>📊 Тренды производительности</h6>
                    <div class="trend-info">
                        <div class="trend-item">
                            <span class="trend-label">Тренд Win Rate:</span>
                            <span class="trend-value ${trends.win_rate_trend > 0 ? 'positive' : trends.win_rate_trend < 0 ? 'negative' : ''}">
                                ${trends.win_rate_trend > 0 ? '↗️ Растет' : trends.win_rate_trend < 0 ? '↘️ Падает' : '➡️ Стабильный'}
                            </span>
                        </div>
                        <div class="trend-item">
                            <span class="trend-label">Тренд Avg PnL:</span>
                            <span class="trend-value ${trends.avg_pnl_trend > 0 ? 'positive' : trends.avg_pnl_trend < 0 ? 'negative' : ''}">
                                ${trends.avg_pnl_trend > 0 ? '↗️ Растет' : trends.avg_pnl_trend < 0 ? '↘️ Падает' : '➡️ Стабильный'}
                            </span>
                        </div>
                        <div class="trend-item">
                            <span class="trend-label">Общий тренд:</span>
                            <span class="trend-value ${trends.ai_improving ? 'positive' : 'negative'}">
                                ${trends.ai_improving ? '🚀 AI улучшается' : '⚠️ AI стабильен или ухудшается'}
                            </span>
                        </div>
                    </div>
                </div>
            `;
        }

        // Сообщение если данных недостаточно
        if (performance && performance.error) {
            html += `
                <div class="no-results">
                    <div class="no-results-icon">📊</div>
                    <p>${performance.error}</p>
                    <small>Накопите больше сделок для анализа производительности AI</small>
                </div>
            `;
        }

        resultsContent.innerHTML = html;
    }

    /**
     * Отображение ошибки загрузки результатов
     */
    displaySelfLearningError(errorMsg) {
        const resultsContent = document.getElementById('selfLearningResultsContent');
        if (!resultsContent) return;

        resultsContent.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">⚠️</div>
                <p>Ошибка загрузки данных</p>
                <small>${errorMsg}</small>
            </div>
        `;
    }

    /**
     * Отображение блока «доступно с премиум лицензией» (без ошибки)
     */
    displaySelfLearningPremiumRequired(message) {
        const resultsContent = document.getElementById('selfLearningResultsContent');
        if (!resultsContent) return;

        resultsContent.innerHTML = `
            <div class="no-results" style="border-color: var(--border-color, #444);">
                <div class="no-results-icon">🔒</div>
                <p>Результаты самообучения</p>
                <small>${message}</small>
            </div>
        `;
    }

    /**
     * Автоматическая загрузка результатов при открытии AI секции
     */
    loadSelfLearningOnShow() {
        // Автоматически загружаем результаты при открытии секции AI
        setTimeout(() => {
            this.loadSelfLearningResults();
            this.loadAIPerformance(); // Также загружаем AI Performance
        }, 500); // Небольшая задержка для завершения анимации
    }
    
    /**
     * Загрузка AI Performance данных
     */
    async loadAIPerformance() {
        try {
            console.log('[AIConfigManager] Загрузка AI Performance...');
            
            // Загружаем performance
            const perfResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/performance`);
            const perfData = await perfResponse.json();
            
            // Загружаем health
            const healthResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/health`);
            const healthData = await healthResponse.json();
            
            this.updatePerformanceCards(perfData, healthData);
            
        } catch (error) {
            console.error('[AIConfigManager] Ошибка загрузки AI Performance:', error);
            this.updatePerformanceCards(null, null);
        }
    }
    
    /**
     * Обновление карточек AI Performance
     */
    updatePerformanceCards(perfData, healthData) {
        // Accuracy
        const accuracyEl = document.getElementById('aiAccuracyValue');
        if (accuracyEl) {
            if (perfData && perfData.success && perfData.performance.daily_metrics) {
                const accuracy = perfData.performance.daily_metrics.direction_accuracy;
                if (accuracy !== null && accuracy !== undefined) {
                    const pct = (accuracy * 100).toFixed(1);
                    accuracyEl.textContent = `${pct}%`;
                    accuracyEl.className = 'perf-card-value ' + (accuracy >= 0.6 ? 'good' : accuracy >= 0.4 ? 'warning' : 'danger');
                } else {
                    accuracyEl.textContent = 'N/A';
                    accuracyEl.className = 'perf-card-value';
                }
            } else {
                accuracyEl.textContent = '--';
                accuracyEl.className = 'perf-card-value';
            }
        }
        
        // Predictions count
        const predictionsEl = document.getElementById('aiPredictionsValue');
        if (predictionsEl) {
            if (perfData && perfData.success && perfData.performance.daily_metrics) {
                const count = perfData.performance.daily_metrics.total_predictions || 0;
                predictionsEl.textContent = count.toLocaleString();
            } else {
                predictionsEl.textContent = '--';
            }
        }
        
        // Confidence
        const confidenceEl = document.getElementById('aiConfidenceValue');
        if (confidenceEl) {
            if (perfData && perfData.success && perfData.performance.daily_metrics) {
                const conf = perfData.performance.daily_metrics.avg_confidence;
                if (conf !== null && conf !== undefined) {
                    confidenceEl.textContent = `${conf.toFixed(1)}%`;
                } else {
                    confidenceEl.textContent = 'N/A';
                }
            } else {
                confidenceEl.textContent = '--';
            }
        }
        
        // Health
        const healthEl = document.getElementById('aiHealthValue');
        if (healthEl) {
            if (healthData && healthData.success && healthData.health) {
                const status = healthData.health.overall_status || 'unknown';
                const statusMap = {
                    'healthy': { text: 'OK', class: 'good' },
                    'warning': { text: '⚠️', class: 'warning' },
                    'critical': { text: '❌', class: 'danger' },
                    'unknown': { text: '?', class: '' }
                };
                const s = statusMap[status] || statusMap['unknown'];
                healthEl.textContent = s.text;
                healthEl.className = 'perf-card-value ' + s.class;
            } else {
                healthEl.textContent = '--';
                healthEl.className = 'perf-card-value';
            }
        }
        
        // Recommendations
        const recsContainer = document.getElementById('aiRecommendations');
        const recsList = document.getElementById('aiRecommendationsList');
        if (recsContainer && recsList) {
            if (perfData && perfData.success && perfData.performance.recommendations && perfData.performance.recommendations.length > 0) {
                recsList.innerHTML = perfData.performance.recommendations
                    .slice(0, 5)
                    .map(rec => `<li>${rec}</li>`)
                    .join('');
                recsContainer.style.display = 'block';
            } else {
                recsContainer.style.display = 'none';
            }
        }
    }
    
    /**
     * Привязка события обновления AI Performance
     */
    bindPerformanceRefreshEvent() {
        const refreshBtn = document.getElementById('refreshAiPerformanceBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadAIPerformance();
            });
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

