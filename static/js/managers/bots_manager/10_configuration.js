/**
 * BotsManager - 10_configuration
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            initializeScopeButtons() {
        const scopeButtons = document.querySelectorAll('.scope-btn');
        const scopeInput = document.getElementById('autoBotScope');
        
        if (!scopeButtons.length || !scopeInput) return;
        
        scopeButtons.forEach(button => {
            button.addEventListener('click', async () => {
                // Убираем активность со всех кнопок
                scopeButtons.forEach(btn => btn.classList.remove('active'));
                
                // Добавляем активность на нажатую кнопку
                button.classList.add('active');
                
                // Обновляем скрытое поле
                const value = button.getAttribute('data-value');
                const oldValue = scopeInput.value;
                scopeInput.value = value;
                
                console.log('[BotsManager] 🎯 Область действия изменена на:', value, '(было:', oldValue + ')');
                console.log('[BotsManager] 🔍 Проверка: autoBotScope.value =', scopeInput.value);
                
                if (oldValue !== value) this.scheduleToggleAutoSave(scopeInput);
            });
        });
        
        console.log('[BotsManager] ✅ Кнопки области действия инициализированы');
    }
    // ==========================================
    // ЗАГРУЗКА КОНФИГУРАЦИИ
    // ==========================================,
            async loadConfigurationData() {
        this.logDebug('[BotsManager] 📋 Загрузка конфигурации...');
        
        try {
            this.logDebug('[BotsManager] 🌐 Запрос Auto Bot конфигурации...');
            // Загружаем конфигурацию Auto Bot
            const autoBotResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`);
            this.logDebug('[BotsManager] 📡 Auto Bot response status:', autoBotResponse.status);
            const autoBotData = await autoBotResponse.json();
            this.logDebug('[BotsManager] 🤖 Auto Bot data:', autoBotData);
            
            this.logDebug('[BotsManager] 🌐 Запрос системных настроек...');
            // Загружаем системные настройки
            const systemResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`);
            this.logDebug('[BotsManager] 📡 System response status:', systemResponse.status);
            const systemData = await systemResponse.json();
            this.logDebug('[BotsManager] ⚙️ System data:', systemData);,
            populateConfigurationForm(config) {
        // Устанавливаем флаг, чтобы предотвратить автосохранение при программном изменении
        this.isProgrammaticChange = true;
        
        this.logDebug('[BotsManager] 🔧 Заполнение формы конфигурации:', config);
        this.logDebug('[BotsManager] 🔍 DOM готовность:', document.readyState);
        this.logDebug('[BotsManager] 🔍 Элемент positionSyncInterval существует:', !!document.getElementById('positionSyncInterval'));
        this.logDebug('[BotsManager] 🔍 Детали конфигурации:');
        this.logDebug('   autoBot:', config.autoBot);
        this.logDebug('   system:', config.system);
        
        const autoBotConfig = config.autoBot || config;,
            showConfigurationLoading(show) {
        // ✅ БЕЗ БЛОКИРОВКИ: Просто логируем, но не блокируем элементы
        const configContainer = document.getElementById('configTab');
        if (!configContainer) return;,
            async saveDefaultConfiguration(defaultConfig) {
        console.log('[BotsManager] 💾 Сохранение конфигурации по умолчанию...');
        
        try {
            // ✅ Проверяем, что есть данные для отправки
            if (!defaultConfig.autoBot || Object.keys(defaultConfig.autoBot).length === 0) {
                console.log('[BotsManager] ⚠️ Auto Bot конфигурация пуста, пропускаем сохранение');
            } else {
                // Сохраняем Auto Bot настройки
                const autoBotResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(defaultConfig.autoBot)
                });
                
                const autoBotData = await autoBotResponse.json();,
            camelToSnake(str) {
        return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
    }
    
    /**
     * Автоматически маппит ID элемента на ключ конфигурации
     */,
            mapElementIdToConfigKey(elementId) {
        // Прямые маппинги для элементов с нестандартными ID
        const directMappings = {
            'globalAutoBotToggle': 'enabled',
            'autoBotMaxConcurrent': 'max_concurrent',
            'autoBotRiskCap': 'risk_cap_percent',
            'autoBotScope': 'scope',  // ✅ КРИТИЧЕСКИ ВАЖНО: маппинг для scope
            'aiEnabled': 'ai_enabled',  // мастер-переключатель AI
            'aiMinConfidence': 'ai_min_confidence',
            'aiOverrideOriginal': 'ai_override_original',
            'fullAiControlToggle': 'full_ai_control',  // полный режим ИИ на вкладке Управление
            'fullAiControlToggleConfig': 'full_ai_control',  // дубль на вкладке Конфигурация
            'rsiLongThreshold': 'rsi_long_threshold',
            'rsiShortThreshold': 'rsi_short_threshold',
            'rsiExitLongWithTrendGlobal': 'rsi_exit_long_with_trend',
            'rsiExitLongAgainstTrendGlobal': 'rsi_exit_long_against_trend',
            'rsiExitShortWithTrendGlobal': 'rsi_exit_short_with_trend',
            'rsiExitShortAgainstTrendGlobal': 'rsi_exit_short_against_trend',
            'rsiExitMinCandlesGlobal': 'rsi_exit_min_candles',
            'rsiExitMinMinutesGlobal': 'rsi_exit_min_minutes',
            'rsiExitMinMovePercentGlobal': 'rsi_exit_min_move_percent',
            'exitWaitBreakevenWhenLoss': 'exit_wait_breakeven_when_loss',
            'rsiLimitEntryEnabled': 'rsi_limit_entry_enabled',
            'rsiLimitExitEnabled': 'rsi_limit_exit_enabled',
            'rsiLimitExitOffsetPercentGlobal': 'rsi_limit_exit_offset_percent',
            'rsiLimitOffsetPercentGlobal': 'rsi_limit_offset_percent',
            'defaultPositionSize': 'default_position_size',
            'defaultPositionMode': 'default_position_mode',
            'leverage': 'leverage',
            'checkInterval': 'check_interval',
            'maxLossPercent': 'max_loss_percent',
            'takeProfitPercent': 'take_profit_percent',
            'closeAtProfitEnabled': 'close_at_profit_enabled',
            'trailingStopActivation': 'trailing_stop_activation',
            'trailingStopDistance': 'trailing_stop_distance',
            'trailingTakeDistance': 'trailing_take_distance',
            'trailingUpdateInterval': 'trailing_update_interval',
            'maxPositionHours': 'max_position_hours',
            'breakEvenProtection': 'break_even_protection',
            'breakEvenTrigger': 'break_even_trigger_percent',
            'lossReentryProtection': 'loss_reentry_protection',
            'lossReentryCount': 'loss_reentry_count',
            'lossReentryCandles': 'loss_reentry_candles',
            'avoidDownTrend': 'avoid_down_trend',
            'avoidUpTrend': 'avoid_up_trend',
            'trendDetectionEnabled': 'trend_detection_enabled',
            'trendAnalysisPeriod': 'trend_analysis_period',
            'trendPriceChangeThreshold': 'trend_price_change_threshold',
            'trendCandlesThreshold': 'trend_candles_threshold',
            'enableMaturityCheck': 'enable_maturity_check',
            'minCandlesForMaturity': 'min_candles_for_maturity',
            'minRsiLow': 'min_rsi_low',
            'maxRsiHigh': 'max_rsi_high',
            'minVolatilityThreshold': 'min_volatility_threshold',
            'rsiTimeFilterEnabled': 'rsi_time_filter_enabled',
            'rsiTimeFilterCandles': 'rsi_time_filter_candles',
            'rsiTimeFilterUpper': 'rsi_time_filter_upper',
            'rsiTimeFilterLower': 'rsi_time_filter_lower',
            'exitScamEnabled': 'exit_scam_enabled',
            'exitScamCandles': 'exit_scam_candles',
            'exitScamSingleCandlePercent': 'exit_scam_single_candle_percent',
            'exitScamMultiCandleCount': 'exit_scam_multi_candle_count',
            'exitScamMultiCandlePercent': 'exit_scam_multi_candle_percent',
            'exitScamTimeframe': 'exit_scam_timeframe',
            'exitScamAutoLearnEnabled': 'exit_scam_auto_learn_enabled',
            'tradingEnabled': 'trading_enabled',
            'useTestServer': 'use_test_server',
            'enhancedRsiEnabled': 'enhanced_rsi_enabled',
            'enhancedRsiVolumeConfirm': 'enhanced_rsi_require_volume_confirmation',
            'enhancedRsiDivergenceConfirm': 'enhanced_rsi_require_divergence_confirmation',
            'enhancedRsiUseStochRsi': 'enhanced_rsi_use_stoch_rsi',
            'rsiExtremeZoneTimeout': 'rsi_extreme_zone_timeout',
            'rsiExtremeOversold': 'rsi_extreme_oversold',
            'rsiExtremeOverbought': 'rsi_extreme_overbought',
            'rsiVolumeMultiplier': 'rsi_volume_confirmation_multiplier',
            'rsiDivergenceLookback': 'rsi_divergence_lookback',
            'limitOrdersEntryEnabled': 'limit_orders_entry_enabled',
            'optimalEntryEnabled': 'ai_optimal_entry_enabled',
            'rsiUpdateInterval': 'rsi_update_interval',
            'autoSaveInterval': 'auto_save_interval',
            'debugMode': 'debug_mode',
            'positionSyncInterval': 'position_sync_interval',
            'inactiveBotCleanupInterval': 'inactive_bot_cleanup_interval',
            'inactiveBotTimeout': 'inactive_bot_timeout',
            'stopLossSetupInterval': 'stop_loss_setup_interval',
            'bybitMarginMode': 'bybit_margin_mode'
        };
        
        // Используем прямое маппинг если есть,
            collectConfigurationData() {
        console.log('[BotsManager] 📋 Сбор данных конфигурации (автоматический режим)...');
        
        // ✅ РАБОТАЕМ НАПРЯМУЮ С КЭШИРОВАННОЙ КОНФИГУРАЦИЕЙ ИЗ БЭКЕНДА
        // Это гарантирует, что мы используем реальные значения из файла конфига, а не дефолтные из HTML,
            collectFieldsFromElements(elements, config) {
        elements.forEach(element => {
            // Пропускаем кнопки и элементы управления
            if (element.type === 'button' || element.type === 'submit' || element.closest('button')) {
                return;
            }
            
            // Пропускаем элементы без ID (динамические поля лимитных ордеров обрабатываются отдельно)
            if (!element.id || element.classList.contains('limit-order-percent') || element.classList.contains('limit-order-margin')) {
                return;
            }
            
            const configKey = this.mapElementIdToConfigKey(element.id);,
            async saveBasicSettings() {
        console.log('[BotsManager] 💾 Сохранение основных настроек...');
        try {
            // ✅ КРИТИЧЕСКИ ВАЖНО: Сначала получаем scope напрямую из UI
            const scopeInput = document.getElementById('autoBotScope');
            const scopeFromUI = scopeInput ? scopeInput.value : null;
            console.log('[BotsManager] 🔍 scope из UI (autoBotScope):', scopeFromUI);
            
            const config = this.collectConfigurationData();
            console.log('[BotsManager] 🔍 scope из collectConfigurationData():', config.autoBot.scope);
            
            // Полный Режим ИИ: тумблер на Управлении или дубль на Конфигурации
            const fullAiControlEl = document.getElementById('fullAiControlToggle');
            const fullAiControlConfigEl = document.getElementById('fullAiControlToggleConfig');
            const fullAiControl = (fullAiControlEl?.checked ?? fullAiControlConfigEl?.checked ?? config.autoBot.full_ai_control) === true;
            const basicSettings = {
                enabled: config.autoBot.enabled,
                max_concurrent: config.autoBot.max_concurrent,
                risk_cap_percent: config.autoBot.risk_cap_percent,
                scope: scopeFromUI || config.autoBot.scope || 'all',  // ✅ Приоритет UI значению
                ai_enabled: config.autoBot.ai_enabled,
                ai_min_confidence: config.autoBot.ai_min_confidence,
                ai_override_original: config.autoBot.ai_override_original,
                full_ai_control: fullAiControl
            };
            
            console.log('[BotsManager] 🔍 Основные настройки для сохранения:', basicSettings);
            console.log('[BotsManager] 🔍 originalConfig.autoBot.scope:', this.originalConfig?.autoBot?.scope);
            console.log('[BotsManager] 🔍 Сравнение scope: UI=' + basicSettings.scope + ', original=' + (this.originalConfig?.autoBot?.scope || 'undefined'));
            
            await this.sendConfigUpdate('auto-bot', basicSettings, 'Основные настройки');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения основных настроек:', error);
            this.showNotification('❌ Ошибка сохранения основных настроек: ' + error.message, 'error');
        }
    },
            _updateFullaiAdaptiveDependentFields() {
        const el = (id) => document.getElementById(id);
        const virtualSuccess = parseInt(el('fullaiAdaptiveVirtualSuccess')?.value, 10);
        const disabled = !Number.isFinite(virtualSuccess) || virtualSuccess <= 0;
        const ids = ['fullaiAdaptiveRealLoss', 'fullaiAdaptiveRoundSize', 'fullaiAdaptiveMaxFailures'];
        const groupIds = ['fullaiAdaptiveDependentGroup', 'fullaiAdaptiveDependentGroup2', 'fullaiAdaptiveDependentGroup3'];
        ids.forEach(id => { const i = el(id); if (i) i.disabled = disabled; });
        groupIds.forEach(id => { const g = el(id); if (g) g.style.opacity = disabled ? '0.6' : '1'; });
    },
            async loadFullaiAdaptiveConfig() {
        try {
            const res = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/fullai-config`, { method: 'GET' });
            const data = await res.json();
            if (!data.success || !data.config) return;
            const c = data.config;
            const el = (id) => document.getElementById(id);
            if (el('fullaiAdaptiveDeadCandles')) el('fullaiAdaptiveDeadCandles').value = c.fullai_adaptive_dead_candles ?? 100;
            if (el('fullaiAdaptiveVirtualSuccess')) el('fullaiAdaptiveVirtualSuccess').value = c.fullai_adaptive_virtual_success_count ?? 3;
            if (el('fullaiAdaptiveRealLoss')) el('fullaiAdaptiveRealLoss').value = c.fullai_adaptive_real_loss_to_retry ?? 1;
            if (el('fullaiAdaptiveRoundSize')) el('fullaiAdaptiveRoundSize').value = c.fullai_adaptive_virtual_round_size ?? 3;
            if (el('fullaiAdaptiveMaxFailures')) el('fullaiAdaptiveMaxFailures').value = c.fullai_adaptive_virtual_max_failures ?? 0;
            this._updateFullaiAdaptiveDependentFields();
        } catch (e) {
            console.warn('[BotsManager] loadFullaiAdaptiveConfig:', e);
        }
    },
            async saveFullaiAdaptiveConfig() {
        try {
            const el = (id) => document.getElementById(id);
            // Один переключатель: Full AI вкл → Adaptive вкл (второй выключатель убран)
            const fullAiOn = el('fullAiControlToggleConfig')?.checked ?? el('fullAiControlToggle')?.checked ?? false;
            const vs = parseInt(el('fullaiAdaptiveVirtualSuccess')?.value, 10);
            const payload = {
                fullai_adaptive_enabled: fullAiOn,
                fullai_adaptive_dead_candles: parseInt(el('fullaiAdaptiveDeadCandles')?.value, 10) || 100,
                fullai_adaptive_virtual_success_count: Number.isFinite(vs) ? vs : 3,
                fullai_adaptive_real_loss_to_retry: parseInt(el('fullaiAdaptiveRealLoss')?.value, 10) || 1,
                fullai_adaptive_virtual_round_size: parseInt(el('fullaiAdaptiveRoundSize')?.value, 10) || 3,
                fullai_adaptive_virtual_max_failures: parseInt(el('fullaiAdaptiveMaxFailures')?.value, 10) || 0
            };
            const res = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/fullai-config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();,
            async saveSystemSettings() {
        console.log('[BotsManager] 💾 Сохранение системных настроек...');
        try {
            const config = this.collectConfigurationData();
            const systemSettings = { ...config.system };
            const bybitMarginEl = document.getElementById('bybitMarginMode');,
            async saveTradingAndRsiExits() {
        console.log('[BotsManager] 💾 Сохранение торговых параметров и RSI выходов...');
        try {
            const config = this.collectConfigurationData();
            const params = {
                rsi_long_threshold: config.autoBot.rsi_long_threshold,
                rsi_short_threshold: config.autoBot.rsi_short_threshold,
                rsi_exit_long_with_trend: config.autoBot.rsi_exit_long_with_trend,
                rsi_exit_long_against_trend: config.autoBot.rsi_exit_long_against_trend,
                rsi_exit_short_with_trend: config.autoBot.rsi_exit_short_with_trend,
                rsi_exit_short_against_trend: config.autoBot.rsi_exit_short_against_trend,
                rsi_exit_min_candles: parseInt(config.autoBot.rsi_exit_min_candles, 10) || 0,
                rsi_exit_min_minutes: parseInt(config.autoBot.rsi_exit_min_minutes, 10) || 0,
                rsi_exit_min_move_percent: parseFloat(config.autoBot.rsi_exit_min_move_percent) || 0,
                exit_wait_breakeven_when_loss: (() => {
                    const el = document.getElementById('exitWaitBreakevenWhenLoss');
                    return el ? el.checked : (config.autoBot.exit_wait_breakeven_when_loss === true);
                })(),
                rsi_limit_entry_enabled: (() => {
                    const el = document.getElementById('rsiLimitEntryEnabled');
                    return el ? el.checked : (config.autoBot.rsi_limit_entry_enabled === true);
                })(),
                rsi_limit_offset_percent: (() => {
                    const el = document.getElementById('rsiLimitOffsetPercentGlobal');,
            async saveRsiTimeFilter() {
        console.log('[BotsManager] 💾 Сохранение RSI временного фильтра...');
        try {
            const config = this.collectConfigurationData();
            const rsiTimeFilter = {
                rsi_time_filter_enabled: config.autoBot.rsi_time_filter_enabled,
                rsi_time_filter_candles: config.autoBot.rsi_time_filter_candles || 6,
                rsi_time_filter_upper: config.autoBot.rsi_time_filter_upper,
                rsi_time_filter_lower: config.autoBot.rsi_time_filter_lower
            };
            
            await this.sendConfigUpdate('auto-bot', rsiTimeFilter, 'RSI временной фильтр');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения RSI временного фильтра:', error);
            this.showNotification('❌ Ошибка сохранения RSI временного фильтра', 'error');
        }
    },
            async saveExitScamFilter() {
        console.log('[BotsManager] 💾 Сохранение ExitScam фильтра...');
        try {
            // ✅ Читаем значения напрямую из DOM, чтобы после перезапуска сервера сохранялось то, что в UI
            const exitScamEnabledEl = document.getElementById('exitScamEnabled');
            const exitScamCandlesEl = document.getElementById('exitScamCandles');
            const exitScamSingleEl = document.getElementById('exitScamSingleCandlePercent');
            const exitScamMultiCountEl = document.getElementById('exitScamMultiCandleCount');
            const exitScamMultiPercentEl = document.getElementById('exitScamMultiCandlePercent');
            const exitScamTimeframeEl = document.getElementById('exitScamTimeframe');
            const config = this.collectConfigurationData();
            const exitScamAutoLearnEl = document.getElementById('exitScamAutoLearnEnabled');
            const exitScamFilter = {
                exit_scam_enabled: exitScamEnabledEl ? exitScamEnabledEl.checked : (config.autoBot.exit_scam_enabled !== false),
                exit_scam_auto_learn_enabled: exitScamAutoLearnEl ? exitScamAutoLearnEl.checked : (config.autoBot.exit_scam_auto_learn_enabled === true),
                exit_scam_candles: exitScamCandlesEl && exitScamCandlesEl.value !== '' ? parseInt(exitScamCandlesEl.value, 10) : (config.autoBot.exit_scam_candles ?? 8),
                exit_scam_single_candle_percent: exitScamSingleEl && exitScamSingleEl.value !== '' ? parseFloat(exitScamSingleEl.value) : (config.autoBot.exit_scam_single_candle_percent ?? 15),
                exit_scam_multi_candle_count: exitScamMultiCountEl && exitScamMultiCountEl.value !== '' ? parseInt(exitScamMultiCountEl.value, 10) : (config.autoBot.exit_scam_multi_candle_count ?? 4),
                exit_scam_multi_candle_percent: exitScamMultiPercentEl && exitScamMultiPercentEl.value !== '' ? parseFloat(exitScamMultiPercentEl.value) : (config.autoBot.exit_scam_multi_candle_percent ?? 50),
                exit_scam_timeframe: exitScamTimeframeEl && exitScamTimeframeEl.value ? exitScamTimeframeEl.value : (config.autoBot.exit_scam_timeframe || '1m')
            };
            console.log('[BotsManager] 🔍 ExitScam из UI:', exitScamFilter.exit_scam_enabled, exitScamFilter.exit_scam_candles);
            await this.sendConfigUpdate('auto-bot', exitScamFilter, 'ExitScam фильтр');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения ExitScam фильтра:', error);
            this.showNotification('❌ Ошибка сохранения ExitScam фильтра', 'error');
        }
    },
            async saveEnhancedRsi() {
        console.log('[BotsManager] 💾 Сохранение Enhanced RSI...');
        try {
            // ✅ Сначала проверяем значения из UI напрямую
            const enhancedRsiEnabledEl = document.getElementById('enhancedRsiEnabled');
            const enhancedRsiVolumeConfirmEl = document.getElementById('enhancedRsiVolumeConfirm');
            const enhancedRsiDivergenceConfirmEl = document.getElementById('enhancedRsiDivergenceConfirm');
            const enhancedRsiUseStochRsiEl = document.getElementById('enhancedRsiUseStochRsi');
            
            console.log('[BotsManager] 🔍 Значения из UI напрямую:');
            console.log('  enhancedRsiEnabled:', enhancedRsiEnabledEl?.checked);
            console.log('  enhancedRsiVolumeConfirm:', enhancedRsiVolumeConfirmEl?.checked);
            console.log('  enhancedRsiDivergenceConfirm:', enhancedRsiDivergenceConfirmEl?.checked);
            console.log('  enhancedRsiUseStochRsi:', enhancedRsiUseStochRsiEl?.checked);
            
            const config = this.collectConfigurationData();
            console.log('[BotsManager] 🔍 Значения из collectConfigurationData():');
            console.log('  config.system:', config.system);
            
            const enhancedRsi = {
                enhanced_rsi_enabled: config.system.enhanced_rsi_enabled,
                enhanced_rsi_require_volume_confirmation: config.system.enhanced_rsi_require_volume_confirmation,
                enhanced_rsi_require_divergence_confirmation: config.system.enhanced_rsi_require_divergence_confirmation,
                enhanced_rsi_use_stoch_rsi: config.system.enhanced_rsi_use_stoch_rsi,
                rsi_extreme_zone_timeout: config.system.rsi_extreme_zone_timeout,
                rsi_extreme_oversold: config.system.rsi_extreme_oversold,
                rsi_extreme_overbought: config.system.rsi_extreme_overbought,
                rsi_volume_confirmation_multiplier: config.system.rsi_volume_confirmation_multiplier,
                rsi_divergence_lookback: config.system.rsi_divergence_lookback
            };
            
            console.log('[BotsManager] 📤 Отправляемые Enhanced RSI настройки:', enhancedRsi);
            
            await this.sendConfigUpdate('system-config', enhancedRsi, 'Enhanced RSI');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения Enhanced RSI:', error);
            this.showNotification('❌ Ошибка сохранения Enhanced RSI', 'error');
        }
    },
            async saveProtectiveMechanisms() {
        console.log('[BotsManager] 💾 Сохранение защитных механизмов...');
        try {
            const config = this.collectConfigurationData();
            
            const protectiveMechanisms = {
                max_loss_percent: config.autoBot.max_loss_percent,
                take_profit_percent: config.autoBot.take_profit_percent,
                close_at_profit_enabled: config.autoBot.close_at_profit_enabled !== false,
                trailing_stop_activation: config.autoBot.trailing_stop_activation,
                trailing_stop_distance: config.autoBot.trailing_stop_distance,
                trailing_take_distance: config.autoBot.trailing_take_distance,
                trailing_update_interval: config.autoBot.trailing_update_interval,
                max_position_hours: config.autoBot.max_position_hours,
                break_even_protection: config.autoBot.break_even_protection,
                break_even_trigger: config.autoBot.break_even_trigger,
                break_even_trigger_percent: config.autoBot.break_even_trigger_percent,
                loss_reentry_protection: config.autoBot.loss_reentry_protection !== false,
                loss_reentry_count: parseInt(config.autoBot.loss_reentry_count || 1),
                loss_reentry_candles: parseInt(config.autoBot.loss_reentry_candles || 3),
                avoid_down_trend: config.autoBot.avoid_down_trend,
                avoid_up_trend: config.autoBot.avoid_up_trend,
                // ✅ ПАРАМЕТРЫ АНАЛИЗА ТРЕНДА
                trend_detection_enabled: config.autoBot.trend_detection_enabled,
                trend_analysis_period: config.autoBot.trend_analysis_period,
                trend_price_change_threshold: config.autoBot.trend_price_change_threshold,
                trend_candles_threshold: config.autoBot.trend_candles_threshold
            };
            
            // sendConfigUpdate автоматически отфильтрует только измененные параметры
            await this.sendConfigUpdate('auto-bot', protectiveMechanisms, 'Защитные механизмы');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения защитных механизмов:', error);
            this.showNotification('❌ Ошибка сохранения защитных механизмов', 'error');
        }
    },
            async saveMaturitySettings() {
        console.log('[BotsManager] 💾 Сохранение настроек зрелости...');
        try {
            const config = this.collectConfigurationData();
            const maturitySettings = {
                enable_maturity_check: config.autoBot.enable_maturity_check,
                min_candles_for_maturity: config.autoBot.min_candles_for_maturity,
                min_rsi_low: config.autoBot.min_rsi_low,
                max_rsi_high: config.autoBot.max_rsi_high,
                min_volatility_threshold: config.autoBot.min_volatility_threshold
            };
            
            await this.sendConfigUpdate('auto-bot', maturitySettings, 'Настройки зрелости');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения настроек зрелости:', error);
            this.showNotification('❌ Ошибка сохранения настроек зрелости', 'error');
        }
    },
            async saveEmaParameters() {
        console.log('[BotsManager] 💾 Сохранение EMA параметров...');
        try {
            const config = this.collectConfigurationData();
            const emaParameters = {
                ema_fast: config.system.ema_fast,
                ema_slow: config.system.ema_slow,
                trend_confirmation_bars: config.system.trend_confirmation_bars
            };
            
            await this.sendConfigUpdate('system-config', emaParameters, 'EMA параметры');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения EMA параметров:', error);
            this.showNotification('❌ Ошибка сохранения EMA параметров', 'error');
        }
    },
            async saveTrendParameters() {
        console.log('[BotsManager] 💾 Сохранение параметров определения тренда...');
        // ❌ УСТАРЕВШИЕ НАСТРОЙКИ EMA - УБРАНЫ (больше не используются)
        // Тренд теперь определяется простым анализом цены - настройки не требуются
        this.showNotification('ℹ️ Настройки тренда больше не используются (тренд определяется автоматически по цене)', 'info');
    },
            hasUnsavedConfigChanges() {
        if (!this.originalConfig) return false;
        try {
            const config = this.collectConfigurationData();
            const autoBotChanges = this.filterChangedParams(config.autoBot || {}, 'autoBot');
            const systemChanges = this.filterChangedParams(config.system || {}, 'system');
            return Object.keys(autoBotChanges).length > 0 || Object.keys(systemChanges).length > 0 || this.aiConfigDirty;
        } catch (e) {
            return false;
        }
    },
            createFloatingSaveButton() {
        if (document.getElementById('floatingSaveConfigBtn')) return;
        const btn = document.createElement('button');
        btn.id = 'floatingSaveConfigBtn';
        btn.className = 'floating-save-config-btn';
        btn.innerHTML = '💾 ' + (this.getTranslation('save_all_config_btn') || 'Сохранить все настройки');
        btn.addEventListener('click', async () => {
            try {
                btn.disabled = true;
                await this.saveAllConfiguration();
            } finally {
                btn.disabled = false;
            }
        });
        document.body.appendChild(btn);
    },
            async saveAllConfiguration() {
        try {
            await this.saveConfiguration(false, true);,
            hideFloatingSaveButton() {
        const btn = document.getElementById('floatingSaveConfigBtn');
        if (btn) btn.classList.remove('visible');
    },
            updateFloatingSaveButtonVisibility() {
        const btn = document.getElementById('floatingSaveConfigBtn');
        if (!btn) return;
        const configTab = document.getElementById('configTab');
        const isConfigTabActive = configTab && configTab.classList.contains('active');
        const botsContainer = document.getElementById('botsContainer');
        const isBotsPageVisible = botsContainer && botsContainer.style.display !== 'none';
        const hasChanges = this.hasUnsavedConfigChanges();,
            filterChangedParams(data, configType = 'autoBot') {
        const originalGroup = configType === 'system'
            ? (this.originalConfig?.system)
            : (this.originalConfig?.autoBot);,
            async sendConfigUpdate(endpoint, data, sectionName, options = {}) {
        // БЕЗ БЛОКИРОВКИ - элементы остаются активными!
        
        try {
            const configType = endpoint === 'system-config' ? 'system' : 'autoBot';
            const filteredData = options.forceSend ? data : this.filterChangedParams(data, configType);
            
            // Если нет изменений, не отправляем запрос (кроме forceSend)
            if (Object.keys(filteredData).length === 0) {
                console.log(`[BotsManager] ℹ️ Нет изменений в ${sectionName}, пропускаем отправку`);
                this.showNotification(`ℹ️ Нет изменений в ${sectionName}`, 'info');
                return;
            }
            
            console.log(`[BotsManager] 📤 Отправка ${sectionName}:`, filteredData);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(filteredData)
            });,
            async saveConfiguration(isAutoSave = false, skipNotification = false) {
        // Отменяем запланированное автосохранение при ручном сохранении,
            async resetConfiguration() {
        console.log('[BotsManager] 🔄 Сброс конфигурации к умолчаниям...');
        
        if (!confirm('Вы уверены, что хотите сбросить конфигурацию к умолчаниям?')) {
            return;
        }
        
        try {
            // Загружаем конфигурацию по умолчанию
            const defaultConfig = {
                autoBot: {
                    enabled: false,
                    max_concurrent: 5,
                    risk_cap_percent: 10,
                    scope: 'all',
                    rsi_long_threshold: 29,
                    rsi_short_threshold: 71,
                    // ✅ Новые параметры RSI выхода с учетом тренда
                    rsi_exit_long_with_trend: 65,
                    rsi_exit_long_against_trend: 60,
                    rsi_exit_short_with_trend: 35,
                    rsi_exit_short_against_trend: 40,
                    rsi_exit_min_candles: 0,
                    rsi_exit_min_minutes: 0,
                    rsi_exit_min_move_percent: 0,
                    exit_wait_breakeven_when_loss: true,
                    default_position_size: 10,
                    default_position_mode: 'usdt',
                    check_interval: 180,
                    max_loss_percent: 15.0,
                    take_profit_percent: 5.0,
                    close_at_profit_enabled: true,
                    trailing_stop_activation: 20.0,
                    trailing_stop_distance: 5.0,
                    trailing_take_distance: 0.5,
                    trailing_update_interval: 3.0,
                    max_position_hours: 0,
                    break_even_protection: true,
                    loss_reentry_protection: true,
                    loss_reentry_count: 1,
                    loss_reentry_candles: 3,
                    avoid_down_trend: true,
                    avoid_up_trend: true,
                    // Параметры анализа тренда
                    trend_detection_enabled: true,
                    trend_analysis_period: 30,
                    trend_price_change_threshold: 7,
                    trend_candles_threshold: 70,
                    break_even_trigger: 20.0,
                    enable_maturity_check: true,
                    min_candles_for_maturity: 200,
                    min_rsi_low: 35,
                    max_rsi_high: 65,
                    trading_enabled: true,
                    use_test_server: false,
                    enhanced_rsi_enabled: true,
                    enhanced_rsi_require_volume_confirmation: true,
                    enhanced_rsi_require_divergence_confirmation: false,
                    enhanced_rsi_use_stoch_rsi: true,
                    rsi_extreme_zone_timeout: 3,
                    rsi_extreme_oversold: 20,
                    rsi_extreme_overbought: 80,
                    rsi_volume_confirmation_multiplier: 1.2,
                    rsi_divergence_lookback: 10
                },
                system: {
                    rsi_update_interval: 1800,
                    auto_save_interval: 30,
                    debug_mode: false,
                    auto_refresh_ui: true,
                    refresh_interval: 3,
                    position_sync_interval: 600,
                    inactive_bot_cleanup_interval: 600,
                    inactive_bot_timeout: 600,
                    stop_loss_setup_interval: 300
                }
            };
            
            await this.saveDefaultConfiguration(defaultConfig);
            this.showNotification('✅ Конфигурация сброшена к умолчаниям!', 'success');
            
            // Перезагружаем конфигурацию
            await this.loadConfigurationData();
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сброса конфигурации:', error);
            this.showNotification('❌ Ошибка сброса конфигурации: ' + error.message, 'error');
        }
    }

    /**
     * Экспорт полного конфига в InfoBot_Config_<TF>.json (Auto Bot + System + AI с сервера).
     * Имя файла по выбранному таймфрейму: InfoBot_Config_1m.json, InfoBot_Config_5m.json, InfoBot_Config_15m.json и т.д.
     */,
            async exportConfig() {
        try {
            const res = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/export-config`);
            if (!res.ok) throw new Error('Не удалось загрузить конфигурацию');
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Ошибка API');
            const tf = (data.timeframe || '1m').replace(/\s/g, '');
            const payload = {
                ...(data.config || {}),
                exportedAt: new Date().toISOString(),
                timeframe: tf,
                version: 1
            };
            const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `InfoBot_Config_${tf}.json`;
            a.click();
            URL.revokeObjectURL(url);
            this.showNotification(`✅ Конфигурация экспортирована в InfoBot_Config_${tf}.json`, 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка экспорта:', error);
            this.showNotification('❌ Ошибка экспорта: ' + error.message, 'error');
        }
    }

    /**
     * Импорт конфигурации из InfoBot_Config_<TF>.json (файл, выгруженный через «Экспорт»).
     * Поддерживает форматы: { autoBot, system, ai } и { config: { autoBot, system, ai } }.
     * Один запрос POST /api/bots/import-config — все блоки применяются и сохраняются в файл и БД.
     */,
            async importConfig(file) {
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            if (!data || typeof data !== 'object') throw new Error('Неверный формат JSON');
            const config = data.config && typeof data.config === 'object' ? data.config : data;
            const hasAutoBot = config.autoBot && typeof config.autoBot === 'object';
            const hasSystem = config.system && typeof config.system === 'object';
            const hasAi = config.ai && typeof config.ai === 'object';
            if (!hasAutoBot && !hasSystem && !hasAi) throw new Error('В файле должны быть autoBot, system и/или ai');
            if (!confirm('Применить загруженную конфигурацию? Настройки будут сохранены в файл и БД.')) return;

            const baseUrl = this.BOTS_SERVICE_URL;
            const res = await fetch(`${baseUrl}/api/bots/import-config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config })
            });
            const result = await res.json();
            if (!result.success) throw new Error(result.error || 'Ошибка импорта');

            await this.loadConfigurationData();
            this.showNotification('✅ Конфигурация импортирована и сохранена в файл', 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка импорта:', error);
            this.showNotification('❌ Ошибка импорта: ' + error.message, 'error');
        }
    },
            testConfiguration() {
        console.log('[BotsManager] 🧪 Тестирование конфигурации...');
        const config = this.collectConfigurationData();
        
        // Простая валидация
        let errors = [];,
            syncDuplicateSettings(config) {
        console.log('[BotsManager] 🔄 Синхронизация дублированных настроек...');
        
        // КРИТИЧЕСКИ ВАЖНО: Синхронизируем переключатель Auto Bot на главной странице
        const globalAutoBotToggleEl = document.getElementById('globalAutoBotToggle');,
            async loadDuplicateSettings() {
        console.log('[BotsManager] 📋 Загрузка дублированных настроек...');
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`);
            const data = await response.json();,
            async initializeGlobalAutoBotToggle() {
        const globalAutoBotToggleEl = document.getElementById('globalAutoBotToggle');
        console.log('[BotsManager] 🔍 initializeGlobalAutoBotToggle вызван');
        console.log('[BotsManager] 🔍 Элемент найден:', !!globalAutoBotToggleEl);
        console.log('[BotsManager] 🔍 data-initialized:', globalAutoBotToggleEl?.getAttribute('data-initialized'));
        
        if (globalAutoBotToggleEl && !globalAutoBotToggleEl.hasAttribute('data-initialized')) {
            console.log('[BotsManager] 🔧 Устанавливаем обработчик события...');
            globalAutoBotToggleEl.setAttribute('data-initialized', 'true');
            
            // КРИТИЧЕСКИ ВАЖНО: Загружаем текущее состояние Auto Bot с сервера
            try {
                console.log('[BotsManager] 🔄 Загрузка текущего состояния Auto Bot...');
                const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`);
                const data = await response.json();,
            initializeMobileAutoBotToggle() {
        const mobileAutoBotToggleEl = document.getElementById('mobileAutobotToggle');
        console.log('[BotsManager] 🔍 initializeMobileAutoBotToggle вызван');
        console.log('[BotsManager] 🔍 Мобильный элемент найден:', !!mobileAutoBotToggleEl);
        console.log('[BotsManager] 🔍 data-initialized:', mobileAutoBotToggleEl?.getAttribute('data-initialized'));
        
        if (mobileAutoBotToggleEl && !mobileAutoBotToggleEl.hasAttribute('data-initialized')) {
            console.log('[BotsManager] 🔧 Устанавливаем обработчик события для мобильного переключателя...');
            mobileAutoBotToggleEl.setAttribute('data-initialized', 'true');
            
            mobileAutoBotToggleEl.addEventListener('change', async (e) => {
                const isEnabled = e.target.checked;
                console.log(`[BotsManager] 🤖 ИЗМЕНЕНИЕ МОБИЛЬНОГО ПЕРЕКЛЮЧАТЕЛЯ: ${isEnabled}`);
                
                // Помечаем, что пользователь изменил переключатель
                mobileAutoBotToggleEl.setAttribute('data-user-changed', 'true');
                console.log('[BotsManager] 🔒 Флаг data-user-changed установлен для мобильного');
                
                // Обновляем визуальное состояние сразу
                const statusText = document.getElementById('mobileAutobotStatusText');,
            async loadAccountInfo() {
        this.logDebug('[BotsManager] 💰 Загрузка информации о едином торговом счете...');
        
        try {
            // Используем account-info сервиса ботов (баланс + флаг недостатка средств)
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/account-info`);
            const data = await response.json();
            
            if (data.success && (data.total_wallet_balance !== undefined || data.total_available_balance !== undefined)) {
                const accountData = {
                    success: true,
                    total_wallet_balance: data.total_wallet_balance,
                    total_available_balance: data.total_available_balance,
                    total_unrealized_pnl: data.total_unrealized_pnl,
                    active_positions: data.active_positions ?? 0,
                    active_bots: data.active_bots ?? this.activeBots?.length ?? 0,
                    insufficient_funds: !!data.insufficient_funds
                };
                this.updateAccountDisplay(accountData);
                this.logDebug('[BotsManager] ✅ Информация о счете загружена:', accountData);
            } else if (data.wallet_data) {
                // Fallback: ответ в формате /api/positions
                const accountData = {
                    success: true,
                    total_wallet_balance: data.wallet_data.total_balance,
                    total_available_balance: data.wallet_data.available_balance,
                    total_unrealized_pnl: data.wallet_data.realized_pnl,
                    active_positions: data.stats?.total_trades || 0,
                    active_bots: this.activeBots?.length || 0,
                    insufficient_funds: !!data.insufficient_funds
                };
                this.updateAccountDisplay(accountData);
            } else {
                console.warn('[BotsManager] ⚠️ Нет данных аккаунта в ответе');
                this.updateAccountDisplay(null);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка запроса информации о счете:', error);
            this.updateAccountDisplay(null);
        }
    },
            updateAccountDisplay(accountData) {
        const balance = accountData && accountData.success ? parseFloat(accountData.total_wallet_balance || 0) : null;
        const available = accountData && accountData.success ? parseFloat(accountData.total_available_balance || 0) : null;
        const pnl = accountData && accountData.success ? parseFloat(accountData.total_unrealized_pnl || 0) : null;
        const positions = accountData && accountData.success ? parseInt(accountData.active_positions || 0) : null;
        const insufficient_funds = !!(accountData && accountData.insufficient_funds);
        const key = [balance, available, pnl, positions, insufficient_funds].join('|');,
            updateBulkControlsVisibility(bots) {
        const bulkControlsEl = document.getElementById('bulkBotControls');
        const countEl = document.getElementById('bulkControlsCount');,
            initializeBulkControls(bots) {
        const startAllBtn = document.getElementById('startAllBotsBtn');
        const stopAllBtn = document.getElementById('stopAllBotsBtn');
        const deleteAllBtn = document.getElementById('deleteAllBotsBtn');
        
        if (startAllBtn && !startAllBtn.hasAttribute('data-initialized')) {
            startAllBtn.setAttribute('data-initialized', 'true');
            startAllBtn.addEventListener('click', () => this.startAllBots());
        }
        
        if (stopAllBtn && !stopAllBtn.hasAttribute('data-initialized')) {
            stopAllBtn.setAttribute('data-initialized', 'true');
            stopAllBtn.addEventListener('click', () => this.stopAllBots());
        }
        
        if (deleteAllBtn && !deleteAllBtn.hasAttribute('data-initialized')) {
            deleteAllBtn.setAttribute('data-initialized', 'true');
            deleteAllBtn.addEventListener('click', () => this.deleteAllBots());
        }
    }

    /** Применить сохранённый вид настроек (Карточки / Списком) */,
            applyConfigViewMode() {
        const wrapper = document.getElementById('configViewWrapper');
        const mode = (typeof localStorage !== 'undefined' && localStorage.getItem('configViewMode')) || 'cards';
        if (!wrapper) return;
        wrapper.classList.remove('config-view-cards', 'config-view-list');
        wrapper.classList.add(mode === 'list' ? 'config-view-list' : 'config-view-cards');
        document.querySelectorAll('.config-view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === mode);
        });
    }

    /** Инициализация переключателя вида настроек (Карточки / Списком) */,
            _initConfigViewSwitcher() {
        const wrapper = document.getElementById('configViewWrapper');
        const btns = document.querySelectorAll('.config-view-btn');
        if (!wrapper || !btns.length) return;
        this.applyConfigViewMode();
        btns.forEach(btn => {
            if (btn.hasAttribute('data-initialized')) return;
            btn.setAttribute('data-initialized', 'true');
            btn.addEventListener('click', () => {
                const view = btn.dataset.view;
                if (typeof localStorage !== 'undefined') localStorage.setItem('configViewMode', view);
                wrapper.classList.remove('config-view-cards', 'config-view-list');
                wrapper.classList.add(view === 'list' ? 'config-view-list' : 'config-view-cards');
                btns.forEach(b => b.classList.toggle('active', b.dataset.view === view));
            });
        });
    },
            initializeConfigurationButtons() {
        console.log('[BotsManager] ⚙️ Инициализация кнопок конфигурации...');
        
        // Обработчик кнопки сохранения конфигурации
        const saveConfigBtn = document.getElementById('saveConfigBtn');
        if (saveConfigBtn && !saveConfigBtn.hasAttribute('data-initialized')) {
            saveConfigBtn.setAttribute('data-initialized', 'true');
            saveConfigBtn.addEventListener('click', () => this.saveConfiguration());
            console.log('[BotsManager] ✅ Кнопка "Сохранить конфигурацию" инициализирована');
        }
        
        // Обработчик кнопки сброса конфигурации
        const resetConfigBtn = document.getElementById('resetConfigBtn');
        if (resetConfigBtn && !resetConfigBtn.hasAttribute('data-initialized')) {
            resetConfigBtn.setAttribute('data-initialized', 'true');
            resetConfigBtn.addEventListener('click', () => this.resetConfiguration());
            console.log('[BotsManager] ✅ Кнопка "Сбросить к умолчаниям" инициализирована');
        }
        
        // Обработчик кнопки тестирования конфигурации
        const testConfigBtn = document.getElementById('testConfigBtn');
        if (testConfigBtn && !testConfigBtn.hasAttribute('data-initialized')) {
            testConfigBtn.setAttribute('data-initialized', 'true');
            testConfigBtn.addEventListener('click', () => this.testConfiguration());
            console.log('[BotsManager] ✅ Кнопка "Тестировать конфигурацию" инициализирована');
        }

        // Экспорт конфигурации в config.json
        const exportConfigBtn = document.getElementById('exportConfigBtn');
        if (exportConfigBtn && !exportConfigBtn.hasAttribute('data-initialized')) {
            exportConfigBtn.setAttribute('data-initialized', 'true');
            exportConfigBtn.addEventListener('click', () => this.exportConfig());
        }

        // Импорт конфигурации из config.json
        const importConfigBtn = document.getElementById('importConfigBtn');
        const importConfigFileInput = document.getElementById('importConfigFileInput');
        if (importConfigBtn && importConfigFileInput && !importConfigBtn.hasAttribute('data-initialized')) {
            importConfigBtn.setAttribute('data-initialized', 'true');
            importConfigBtn.addEventListener('click', () => importConfigFileInput.click());
            importConfigFileInput.addEventListener('change', (e) => {
                const file = e.target.files?.[0];
                if (file) this.importConfig(file);
                e.target.value = '';
            });
        }

        // Переключатель вида настроек (Карточки / Списком)
        this._initConfigViewSwitcher();
        
        // ✅ ОБРАБОТЧИКИ ДЛЯ КНОПОК СОХРАНЕНИЯ ОТДЕЛЬНЫХ БЛОКОВ
        
        // Основные настройки
        const saveBasicBtn = document.querySelector('.config-section-save-btn[data-section="basic"]');
        if (saveBasicBtn && !saveBasicBtn.hasAttribute('data-initialized')) {
            saveBasicBtn.setAttribute('data-initialized', 'true');
            saveBasicBtn.addEventListener('click', () => this.saveBasicSettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить основные настройки" инициализирована');
        }
        
        const applyFullAiControl = async (value) => {
            try {
                await this.sendConfigUpdate('auto-bot', { full_ai_control: value }, value ? 'Полный Режим ИИ включён' : 'Полный Режим ИИ выключен', { forceSend: true });
                const autoBot = this.collectConfigurationData().autoBot || {};
                this.syncDuplicateSettings({ ...autoBot, full_ai_control: value });
                // Один переключатель управляет и Adaptive: синхронизируем fullai_config
                await this.saveFullaiAdaptiveConfig();
            } catch (e) {
                console.error('[BotsManager] Ошибка сохранения FullAI:', e);
                this.showNotification('Ошибка сохранения переключателя FullAI', 'error');
            }
        };
        // Тумблер «Полный Режим ИИ» на вкладке Управление и дубль на Конфигурации — синхронизируем при изменении любого
        const fullAiToggleEl = document.getElementById('fullAiControlToggle');
        const fullAiToggleConfigEl = document.getElementById('fullAiControlToggleConfig');
        const syncFullAiToggles = (sourceEl, value) => {
            if (fullAiToggleEl && fullAiToggleEl !== sourceEl) fullAiToggleEl.checked = value;
            if (fullAiToggleConfigEl && fullAiToggleConfigEl !== sourceEl) fullAiToggleConfigEl.checked = value;
        };
        if (fullAiToggleEl && !fullAiToggleEl.hasAttribute('data-fullai-listener')) {
            fullAiToggleEl.setAttribute('data-fullai-listener', 'true');
            fullAiToggleEl.addEventListener('change', () => {
                const value = fullAiToggleEl.checked;
                syncFullAiToggles(fullAiToggleEl, value);
                applyFullAiControl(value);
            });
        }
        if (fullAiToggleConfigEl && !fullAiToggleConfigEl.hasAttribute('data-fullai-listener')) {
            fullAiToggleConfigEl.setAttribute('data-fullai-listener', 'true');
            fullAiToggleConfigEl.addEventListener('change', () => {
                const value = fullAiToggleConfigEl.checked;
                syncFullAiToggles(fullAiToggleConfigEl, value);
                applyFullAiControl(value);
            });
        }
        
        let fullaiAdaptiveSaveTimer = null;
        const scheduleFullaiAdaptiveSave = () => {
            if (fullaiAdaptiveSaveTimer) clearTimeout(fullaiAdaptiveSaveTimer);
            fullaiAdaptiveSaveTimer = setTimeout(() => this.saveFullaiAdaptiveConfig(), 800);
        };
        const fullaiAdaptiveIds = ['fullaiAdaptiveDeadCandles', 'fullaiAdaptiveVirtualSuccess', 'fullaiAdaptiveRealLoss', 'fullaiAdaptiveRoundSize', 'fullaiAdaptiveMaxFailures'];
        fullaiAdaptiveIds.forEach(id => {
            const el = document.getElementById(id);
            if (el && !el.hasAttribute('data-fullai-adaptive-listener')) {
                el.setAttribute('data-fullai-adaptive-listener', 'true');
                el.addEventListener('change', () => {
                    if (id === 'fullaiAdaptiveVirtualSuccess') this._updateFullaiAdaptiveDependentFields();
                    scheduleFullaiAdaptiveSave();
                });
                el.addEventListener('input', () => {
                    if (id === 'fullaiAdaptiveVirtualSuccess') this._updateFullaiAdaptiveDependentFields();
                    scheduleFullaiAdaptiveSave();
                });
            }
        });
        this._updateFullaiAdaptiveDependentFields();
        
        // Кнопка сброса всех монет к глобальным настройкам
        const resetAllCoinsBtn = document.getElementById('resetAllCoinsToGlobalBtn');
        if (resetAllCoinsBtn && !resetAllCoinsBtn.hasAttribute('data-initialized')) {
            resetAllCoinsBtn.setAttribute('data-initialized', 'true');
            resetAllCoinsBtn.addEventListener('click', () => this.resetAllCoinsToGlobalSettings());
            console.log('[BotsManager] ✅ Кнопка "Сбросить все монеты к глобальным настройкам" инициализирована');
        }
        
        // Системные настройки
        const saveSystemBtn = document.querySelector('.config-section-save-btn[data-section="system"]');
        if (saveSystemBtn && !saveSystemBtn.hasAttribute('data-initialized')) {
            saveSystemBtn.setAttribute('data-initialized', 'true');
            saveSystemBtn.addEventListener('click', () => this.saveSystemSettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить системные настройки" инициализирована');
        }
        
        // Торговые параметры и RSI выходы (объединённая кнопка)
        const saveTradingRsiBtn = document.querySelector('.config-section-save-btn[data-section="trading-rsi"]');
        if (saveTradingRsiBtn && !saveTradingRsiBtn.hasAttribute('data-initialized')) {
            saveTradingRsiBtn.setAttribute('data-initialized', 'true');
            saveTradingRsiBtn.addEventListener('click', () => this.saveTradingAndRsiExits());
        }
        
        // RSI временной фильтр
        const saveRsiTimeBtn = document.querySelector('.config-section-save-btn[data-section="rsi-time-filter"]');
        if (saveRsiTimeBtn && !saveRsiTimeBtn.hasAttribute('data-initialized')) {
            saveRsiTimeBtn.setAttribute('data-initialized', 'true');
            saveRsiTimeBtn.addEventListener('click', () => this.saveRsiTimeFilter());
            console.log('[BotsManager] ✅ Кнопка "Сохранить RSI временной фильтр" инициализирована');
        }
        
        // ExitScam фильтр — сохраняется по общим правилам (авто при переключении чекбоксов/select, числа — через общее сохранение)
        
        // Enhanced RSI
        const saveEnhancedRsiBtn = document.querySelector('.config-section-save-btn[data-section="enhanced-rsi"]');
        if (saveEnhancedRsiBtn && !saveEnhancedRsiBtn.hasAttribute('data-initialized')) {
            saveEnhancedRsiBtn.setAttribute('data-initialized', 'true');
            saveEnhancedRsiBtn.addEventListener('click', () => this.saveEnhancedRsi());
            console.log('[BotsManager] ✅ Кнопка "Сохранить Enhanced RSI" инициализирована');
        }
        
        // Защитные механизмы
        const saveProtectiveBtn = document.querySelector('.config-section-save-btn[data-section="protective"]');
        if (saveProtectiveBtn && !saveProtectiveBtn.hasAttribute('data-initialized')) {
            saveProtectiveBtn.setAttribute('data-initialized', 'true');
            saveProtectiveBtn.addEventListener('click', () => this.saveProtectiveMechanisms());
            console.log('[BotsManager] ✅ Кнопка "Сохранить защитные механизмы" инициализирована');
        }
        
        // Настройки зрелости
        const saveMaturityBtn = document.querySelector('.config-section-save-btn[data-section="maturity"]');
        if (saveMaturityBtn && !saveMaturityBtn.hasAttribute('data-initialized')) {
            saveMaturityBtn.setAttribute('data-initialized', 'true');
            saveMaturityBtn.addEventListener('click', () => this.saveMaturitySettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить настройки зрелости" инициализирована');
        }
        
        // EMA параметры
        const saveEmaBtn = document.querySelector('.config-section-save-btn[data-section="ema"]');
        if (saveEmaBtn && !saveEmaBtn.hasAttribute('data-initialized')) {
            saveEmaBtn.setAttribute('data-initialized', 'true');
            saveEmaBtn.addEventListener('click', () => this.saveEmaParameters());
            console.log('[BotsManager] ✅ Кнопка "Сохранить EMA параметры" инициализирована');
        }
        
        // Параметры тренда
        const saveTrendBtn = document.querySelector('.config-section-save-btn[data-section="trend"]');
        if (saveTrendBtn && !saveTrendBtn.hasAttribute('data-initialized')) {
            saveTrendBtn.setAttribute('data-initialized', 'true');
            saveTrendBtn.addEventListener('click', () => this.saveTrendParameters());
            console.log('[BotsManager] ✅ Кнопка "Сохранить параметры тренда" инициализирована');
        }
        
        // Набор позиций лимитными ордерами
        const saveLimitOrdersBtn = document.querySelector('.config-section-save-btn[data-section="limit-orders"]');
        if (saveLimitOrdersBtn && !saveLimitOrdersBtn.hasAttribute('data-initialized')) {
            saveLimitOrdersBtn.setAttribute('data-initialized', 'true');
            saveLimitOrdersBtn.addEventListener('click', () => this.saveLimitOrdersSettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить настройки набора позиций" инициализирована');
        }
        
        // Кнопка "По умолчанию" для лимитных ордеров
        const resetLimitOrdersBtn = document.getElementById('resetLimitOrdersBtn');
        if (resetLimitOrdersBtn && !resetLimitOrdersBtn.hasAttribute('data-initialized')) {
            resetLimitOrdersBtn.setAttribute('data-initialized', 'true');
            resetLimitOrdersBtn.addEventListener('click', () => this.resetLimitOrdersToDefault());
            console.log('[BotsManager] ✅ Кнопка "По умолчанию" для лимитных ордеров инициализирована');
        }
        
        // Hot Reload кнопка
        const reloadModulesBtn = document.getElementById('reloadModulesBtn');
        if (reloadModulesBtn && !reloadModulesBtn.hasAttribute('data-initialized')) {
            reloadModulesBtn.setAttribute('data-initialized', 'true');
            reloadModulesBtn.addEventListener('click', () => this.reloadModules());
            console.log('[BotsManager] ✅ Кнопка "Hot Reload" инициализирована');
        }
        
        console.log('[BotsManager] ✅ Все кнопки конфигурации инициализированы');
    }
    
    /**
     * Инициализация автосохранения конфигурации
     * Автоматически сохраняет изменения через 2 секунды после внесения в поле
     */,
            initializeAutoSave() {
        console.log('[BotsManager] ⚙️ Инициализация автосохранения конфигурации...');
        
        // Находим контейнер конфигурации
        const configTab = document.getElementById('configTab');,
            async saveSingleToggleToBackend(input) {
        if (!input || !input.id) return false;
        const configKey = this.mapElementIdToConfigKey(input.id);
        if (!configKey) return false;

        const systemConfigKeys = [
            'enhanced_rsi_enabled', 'enhanced_rsi_require_volume_confirmation', 'enhanced_rsi_require_divergence_confirmation',
            'enhanced_rsi_use_stoch_rsi', 'rsi_extreme_zone_timeout', 'rsi_extreme_oversold', 'rsi_extreme_overbought',
            'rsi_volume_confirmation_multiplier', 'rsi_divergence_lookback', 'rsi_update_interval', 'auto_save_interval',
            'debug_mode', 'refresh_interval', 'position_sync_interval',
            'inactive_bot_cleanup_interval', 'inactive_bot_timeout', 'stop_loss_setup_interval',
            'bybit_margin_mode'
        ];
        const isSystem = configKey.startsWith('system_') || systemConfigKeys.includes(configKey);

        let value;,
            scheduleToggleAutoSave(input) {
        if (this.toggleAutoSaveTimer) clearTimeout(this.toggleAutoSaveTimer);
        const self = this;
        this.toggleAutoSaveTimer = setTimeout(async () => {
            self.toggleAutoSaveTimer = null;
            try {
                if (input && input.closest('#aiConfigSection')) {,
            addAutoSaveHandlers(inputs) {
        // Добавляем обработчики для каждого поля
        inputs.forEach((input, index) => {
            // Пропускаем кнопки и элементы управления
            if (input.type === 'button' || input.type === 'submit' || input.closest('button')) {
                return;
            }
            
            // Проверяем, не добавлен ли уже обработчик
            if (input.hasAttribute('data-autosave-initialized')) {
                return;
            }
            
            input.setAttribute('data-autosave-initialized', 'true');
            
            // Числа и текст: сохраняем только при blur (уход с поля) или Enter — не при каждом нажатии клавиши,
            addStepperButtons() {
        try {
            const configTab = document.getElementById('configTab');
            const aiSection = document.getElementById('aiConfigSection');
            const containers = [configTab, aiSection].filter(Boolean);
            let added = 0;
            containers.forEach(container => {
                if (!container || !container.querySelectorAll) return;
                const inputs = container.querySelectorAll('.config-input-with-unit input[type="number"].config-input');
                inputs.forEach((input) => {
                    try {
                        const parent = input.closest('.config-input-with-unit');
                        if (!parent || parent.hasAttribute('data-stepper-initialized')) return;
                        parent.setAttribute('data-stepper-initialized', 'true');
                        parent.classList.add('config-input-stepper');
                        const step = parseFloat(input.getAttribute('step')) || 1;
                        const min = input.hasAttribute('min') ? parseFloat(input.getAttribute('min')) : null;
                        const max = input.hasAttribute('max') ? parseFloat(input.getAttribute('max')) : null;
                        const self = this;
                        const applyValue = (val) => {
                            if (min != null && val < min) val = min;
                            if (max != null && val > max) val = max;
                            input.value = val;
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                            input.dispatchEvent(new Event('change', { bubbles: true }));
                            if (!self.isProgrammaticChange) self.updateFloatingSaveButtonVisibility();
                        };
                        const minusBtn = document.createElement('button');
                        minusBtn.type = 'button';
                        minusBtn.className = 'config-step-btn config-step-minus';
                        minusBtn.setAttribute('aria-label', '-');
                        minusBtn.textContent = '−';
                        minusBtn.addEventListener('click', () => {
                            const v = parseFloat(input.value) || 0;
                            applyValue(v - step);
                        });
                        const plusBtn = document.createElement('button');
                        plusBtn.type = 'button';
                        plusBtn.className = 'config-step-btn config-step-plus';
                        plusBtn.setAttribute('aria-label', '+');
                        plusBtn.textContent = '+';
                        plusBtn.addEventListener('click', () => {
                            const v = parseFloat(input.value) || 0;
                            applyValue(v + step);
                        });
                        parent.insertBefore(minusBtn, input);
                        parent.insertBefore(plusBtn, input.nextSibling);
                        added++;
                    } catch (err) {
                        console.warn('[BotsManager] addStepperButtons: ошибка для поля', input?.id, err);
                    }
                });
            });
            if (added > 0) console.log('[BotsManager] ✅ Кнопки +/- добавлены для', added, 'полей');
        } catch (err) {
            console.warn('[BotsManager] addStepperButtons:', err);
        }
    }
    
    /**
     * Планирует автоматическое сохранение конфигурации с задержкой
     */,
            scheduleAutoSave() {
        // ✅ Сохраняем контекст this для использования в setTimeout
        const self = this;
        
        // Очищаем предыдущий таймер,
            async reloadModules() {
        console.log('[BotsManager] 🔄 Перезагрузка модулей...');
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/system/reload-modules`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();,
            async startAllBots() {,
            async stopAllBots() {,
            async deleteAllBots() {,
            showConfigNotification(title, message, type = 'success', changes = null) {
        // Удаляем предыдущее уведомление если есть
        const existingNotification = document.querySelector('.config-save-notification');,
            detectConfigChanges(oldAutoBot, oldSystem, newAutoBot, newSystem) {
        const changes = [];
        
        // Словарь с человеко-читаемыми названиями настроек
        const configLabels = {
            // Auto Bot настройки
            'enabled': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Auto Bot enabled' : 'Auto Bot включен',
            'max_concurrent': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Max concurrent bots' : 'Макс. одновременных ботов',
            'risk_cap_percent': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Risk cap (% of deposit)' : 'Рискованность (% от депозита)',
            'scope': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Action scope' : 'Область действия',
            'rsi_long_threshold': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI for LONG positions' : 'RSI для LONG позиций',
            'rsi_short_threshold': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI for SHORT positions' : 'RSI для SHORT позиций',
            'rsi_exit_long': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI exit from LONG' : 'RSI выход из LONG',
            'rsi_exit_short': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI exit from SHORT' : 'RSI выход из SHORT',
            'default_position_size': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Default position size' : 'Размер позиции по умолчанию',
            'check_interval': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Check interval (min)' : 'Интервал проверки (мин)',
            'max_loss_percent': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Max loss (%)' : 'Макс. убыток (%)',
            'trailing_stop_activation': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Trailing stop activation (%)' : 'Активация трейлинг-стопа (%)',
            'trailing_stop_distance': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Trailing stop distance (%)' : 'Расстояние трейлинг-стопа (%)',
            'max_position_hours': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Max time in position (sec)' : 'Макс. время в позиции (сек)',
            'break_even_protection': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Break-even protection' : 'Защита безубыточности',
            'break_even_trigger': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Break-even trigger (%)' : 'Триггер безубыточности (%)',
            'avoid_down_trend': window.languageUtils?.getCurrentLanguage() === 'en' ? '🔻 Avoid downtrend (LONG)' : '🔻 Избегать нисходящий тренд (LONG)',
            'avoid_up_trend': window.languageUtils?.getCurrentLanguage() === 'en' ? '📈 Avoid uptrend (SHORT)' : '📈 Избегать восходящий тренд (SHORT)',
            
            // Системные настройки
            'rsi_update_interval': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI update interval' : 'Интервал обновления RSI',
            'auto_save_interval': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Auto-save interval' : 'Интервал автосохранения',
            'mini_chart_update_interval': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Mini charts update interval' : 'Интервал обновления миниграфиков',
            'debug_mode': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Debug mode' : 'Режим отладки',
            'auto_refresh_ui': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Auto-refresh UI' : 'Автообновление UI'
        };
        
        // Функция для форматирования значений
        const formatValue = (key, value) => {
            const isEnglish = window.languageUtils?.getCurrentLanguage() === 'en';
    });
})();
