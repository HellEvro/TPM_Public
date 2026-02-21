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
    },
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
            this.logDebug('[BotsManager] ⚙️ System data:', systemData);
            
            if (autoBotData.success && systemData.success) {
                this.populateConfigurationForm({
                    autoBot: autoBotData.config,
                    system: systemData.config
                });
                
                // Обновляем RSI пороги из конфигурации
                this.updateRsiThresholds(autoBotData.config);
                
                console.log('[BotsManager] ✅ Конфигурация загружена');
                console.log('[BotsManager] Auto Bot config:', autoBotData.config);
                console.log('[BotsManager] System config:', systemData.config);
            } else {
                const errorMsg = !autoBotData.success ? autoBotData.message : systemData.message;
                console.error('[BotsManager] ❌ Ошибка загрузки конфигурации:', errorMsg);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка запроса конфигурации:', error);
        }
    },
            populateConfigurationForm(config) {
        // Устанавливаем флаг, чтобы предотвратить автосохранение при программном изменении
        this.isProgrammaticChange = true;
        
        this.logDebug('[BotsManager] 🔧 Заполнение формы конфигурации:', config);
        this.logDebug('[BotsManager] 🔍 DOM готовность:', document.readyState);
        this.logDebug('[BotsManager] 🔍 Элемент positionSyncInterval существует:', !!document.getElementById('positionSyncInterval'));
        this.logDebug('[BotsManager] 🔍 Детали конфигурации:');
        this.logDebug('   autoBot:', config.autoBot);
        this.logDebug('   system:', config.system);
        
        const autoBotConfig = config.autoBot || config;
        if (!autoBotConfig.default_position_mode) {
            autoBotConfig.default_position_mode = 'usdt';
        }
        
        // ✅ Кэшируем конфигурацию Auto Bot для быстрого доступа (для updateCoinInfo и др.)
        this.cachedAutoBotConfig = autoBotConfig;
        
        // ✅ ВСЕГДА обновляем originalConfig при загрузке конфигурации из бэкенда
        // Это гарантирует, что после сохранения и перезагрузки конфигурации originalConfig синхронизирован
        this.originalConfig = {
            autoBot: JSON.parse(JSON.stringify(autoBotConfig)), // Глубокое копирование
            system: JSON.parse(JSON.stringify(config.system || {}))
        };
        console.log(`[BotsManager] 💾 originalConfig обновлен из бэкенда для отслеживания изменений`);
        console.log(`[BotsManager] 🔍 originalConfig ключи:`, Object.keys(this.originalConfig.autoBot));
        console.log(`[BotsManager] 🔍 trailing_stop_activation в originalConfig:`, this.originalConfig.autoBot.trailing_stop_activation);
        console.log(`[BotsManager] 🔍 trailing_stop_distance в originalConfig:`, this.originalConfig.autoBot.trailing_stop_distance);
        console.log(`[BotsManager] 🔍 break_even_trigger в originalConfig:`, this.originalConfig.autoBot.break_even_trigger_percent ?? this.originalConfig.autoBot.break_even_trigger);
            
            // Защита от повторных входов после убытка
            const lossReentryProtectionEl = document.getElementById('lossReentryProtection');
            if (lossReentryProtectionEl) {
                lossReentryProtectionEl.checked = autoBotConfig.loss_reentry_protection !== false;
                console.log('[BotsManager] 🛡️ Защита от повторных входов:', lossReentryProtectionEl.checked);
            }

            const lossReentryCountEl = document.getElementById('lossReentryCount');
            if (lossReentryCountEl) {
                lossReentryCountEl.value = autoBotConfig.loss_reentry_count || 1;
                console.log('[BotsManager] 🔢 Количество убыточных позиций (N):', lossReentryCountEl.value);
            }

            const lossReentryCandlesEl = document.getElementById('lossReentryCandles');
            if (lossReentryCandlesEl) {
                lossReentryCandlesEl.value = autoBotConfig.loss_reentry_candles || 3;
                console.log('[BotsManager] 🕯️ ВХОД ЧЕРЕЗ X СВЕЧЕЙ:', lossReentryCandlesEl.value);
            }
        console.log(`[BotsManager] 🔍 avoid_down_trend в originalConfig:`, this.originalConfig.autoBot.avoid_down_trend);
        console.log(`[BotsManager] 🔍 avoid_up_trend в originalConfig:`, this.originalConfig.autoBot.avoid_up_trend);
        
        // ==========================================
        // КОНФИГУРАЦИЯ AUTO BOT
        // ==========================================
        
        // Основные настройки Auto Bot (включение/выключение управляется на основной вкладке)
        
        const maxConcurrentEl = document.getElementById('autoBotMaxConcurrent');
        if (maxConcurrentEl) {
            maxConcurrentEl.value = autoBotConfig.max_concurrent || 5;
            console.log('[BotsManager] 👥 Макс. одновременных ботов:', maxConcurrentEl.value);
        }
        
        const riskCapEl = document.getElementById('autoBotRiskCap');
        if (riskCapEl) {
            riskCapEl.value = autoBotConfig.risk_cap_percent || 10;
            console.log('[BotsManager] ⚠️ Лимит риска:', riskCapEl.value);
        }
        
        // Область действия
        const scopeEl = document.getElementById('autoBotScope');
        if (scopeEl) {
            const scopeValue = autoBotConfig.scope;
            if (scopeValue !== undefined) {
                scopeEl.value = scopeValue;
                console.log('[BotsManager] 🎯 Область действия:', scopeValue);
                
                const scopeButtons = document.querySelectorAll('.scope-btn');
                console.log('[BotsManager] 🔍 Найдено кнопок области:', scopeButtons.length);
                
                scopeButtons.forEach(btn => {
                    btn.classList.remove('active');
                    const btnValue = btn.getAttribute('data-value');
                    if (btnValue === scopeEl.value) {
                        btn.classList.add('active');
                        console.log('[BotsManager] ✅ Активирована кнопка:', btnValue);
                    }
                });
            } else {
                console.warn('[BotsManager] ⚠️ Область действия не найдена в API, оставляем поле пустым');
            }
        } else {
            console.error('[BotsManager] ❌ Элемент autoBotScope не найден!');
        }
        
        // ai_enabled в auto-bot конфиге задаётся мастер-переключателем aiEnabled (см. mapElementIdToConfigKey)
        const aiConfidenceEl = document.getElementById('aiMinConfidence');
        if (aiConfidenceEl) {
            const value = Number.parseFloat(autoBotConfig.ai_min_confidence);
            aiConfidenceEl.value = Number.isFinite(value) ? value : 0.7;
        }
        
        const aiOverrideEl = document.getElementById('aiOverrideOriginal');
        if (aiOverrideEl) {
            const overrideValue = autoBotConfig.ai_override_original;
            aiOverrideEl.checked = overrideValue !== false;
        }
        
        // ✅ AI оптимальный вход (может быть в AI секции, но сохраняется в auto-bot конфиге)
        const optimalEntryEl = document.getElementById('optimalEntryEnabled');
        if (optimalEntryEl) {
            optimalEntryEl.checked = Boolean(autoBotConfig.ai_optimal_entry_enabled);
            console.log('[BotsManager] 🎯 AI оптимальный вход:', optimalEntryEl.checked);
        }
        
        // ✅ FullAI адаптивные параметры (из auto-bot ответа; GET auto-bot уже подмешивает значения из AutoBotConfig)
        const deadCandles = autoBotConfig.fullai_adaptive_dead_candles;
        if (deadCandles !== undefined && document.getElementById('fullaiAdaptiveDeadCandles')) {
            document.getElementById('fullaiAdaptiveDeadCandles').value = parseInt(deadCandles, 10) || 10;
        }
        const virtualSuccess = autoBotConfig.fullai_adaptive_virtual_success_count ?? autoBotConfig.fullai_adaptive_virtual_success;
        if (virtualSuccess !== undefined && document.getElementById('fullaiAdaptiveVirtualSuccess')) {
            document.getElementById('fullaiAdaptiveVirtualSuccess').value = parseInt(virtualSuccess, 10) || 3;
        }
        const realLoss = autoBotConfig.fullai_adaptive_real_loss_to_retry ?? autoBotConfig.fullai_adaptive_real_loss;
        if (realLoss !== undefined && document.getElementById('fullaiAdaptiveRealLoss')) {
            document.getElementById('fullaiAdaptiveRealLoss').value = parseInt(realLoss, 10) || 1;
        }
        const roundSize = autoBotConfig.fullai_adaptive_virtual_round_size ?? autoBotConfig.fullai_adaptive_round_size;
        if (roundSize !== undefined && document.getElementById('fullaiAdaptiveRoundSize')) {
            document.getElementById('fullaiAdaptiveRoundSize').value = parseInt(roundSize, 10) || 3;
        }
        const maxFailures = autoBotConfig.fullai_adaptive_virtual_max_failures ?? autoBotConfig.fullai_adaptive_max_failures;
        if (maxFailures !== undefined && document.getElementById('fullaiAdaptiveMaxFailures')) {
            document.getElementById('fullaiAdaptiveMaxFailures').value = parseInt(maxFailures, 10) || 0;
        }
        
        // Торговые параметры
        const rsiLongEl = document.getElementById('rsiLongThreshold');
        if (rsiLongEl) {
            rsiLongEl.value = autoBotConfig.rsi_long_threshold || 29;
            console.log('[BotsManager] 📈 RSI LONG порог:', rsiLongEl.value);
        }
        
        const rsiShortEl = document.getElementById('rsiShortThreshold');
        if (rsiShortEl) {
            rsiShortEl.value = autoBotConfig.rsi_short_threshold || 71;
            console.log('[BotsManager] 📈 RSI SHORT порог:', rsiShortEl.value);
        }
        
        const rsiLimitEntryEl = document.getElementById('rsiLimitEntryEnabled');
        if (rsiLimitEntryEl) {
            rsiLimitEntryEl.checked = autoBotConfig.rsi_limit_entry_enabled === true;
        }
        const rsiLimitOffsetEl = document.getElementById('rsiLimitOffsetPercentGlobal');
        if (rsiLimitOffsetEl) {
            const v = parseFloat(autoBotConfig.rsi_limit_offset_percent);
            rsiLimitOffsetEl.value = (!isNaN(v) && v >= 0) ? v : 0.2;
        }
        const rsiLimitExitEl = document.getElementById('rsiLimitExitEnabled');
        if (rsiLimitExitEl) {
            rsiLimitExitEl.checked = autoBotConfig.rsi_limit_exit_enabled === true;
        }
        const rsiLimitExitOffsetEl = document.getElementById('rsiLimitExitOffsetPercentGlobal');
        if (rsiLimitExitOffsetEl) {
            const v = parseFloat(autoBotConfig.rsi_limit_exit_offset_percent);
            rsiLimitExitOffsetEl.value = (!isNaN(v) && v >= 0) ? v : 0.2;
        }
        
        const positionSizeEl = document.getElementById('defaultPositionSize');
        if (positionSizeEl) {
            positionSizeEl.value = autoBotConfig.default_position_size || 10;
            console.log('[BotsManager] 💰 Размер позиции:', positionSizeEl.value);
        }
        const positionModeEl = document.getElementById('defaultPositionMode');
        if (positionModeEl) {
            positionModeEl.value = autoBotConfig.default_position_mode || 'usdt';
            console.log('[BotsManager] 🔄 Режим размера позиции:', positionModeEl.value);
        }
        
        const leverageEl = document.getElementById('leverage');
        if (leverageEl) {
            leverageEl.value = autoBotConfig.leverage || 10;
            console.log('[BotsManager] ⚡ Кредитное плечо:', leverageEl.value);
        }
        
        const checkIntervalEl = document.getElementById('checkInterval');
        if (checkIntervalEl && autoBotConfig.check_interval !== undefined) {
            checkIntervalEl.value = autoBotConfig.check_interval;
            console.log('[BotsManager] ⏱️ Интервал проверки установлен:', autoBotConfig.check_interval, '(из API)');
        } else if (checkIntervalEl) {
            console.warn('[BotsManager] ⚠️ Интервал проверки не найден в API, оставляем поле пустым');
        }
        

        
        // ✅ Новые параметры RSI выхода с учетом тренда
        const rsiExitLongWithTrendEl = document.getElementById('rsiExitLongWithTrendGlobal');
        if (rsiExitLongWithTrendEl && rsiExitLongWithTrendEl.value) {
            rsiExitLongWithTrendEl.value = autoBotConfig.rsi_exit_long_with_trend || 65;
            console.log('[BotsManager] 🟢📈 RSI выход LONG (по тренду):', rsiExitLongWithTrendEl.value);
        }
        
        const rsiExitLongAgainstTrendEl = document.getElementById('rsiExitLongAgainstTrendGlobal');
        if (rsiExitLongAgainstTrendEl) {
            rsiExitLongAgainstTrendEl.value = autoBotConfig.rsi_exit_long_against_trend || 60;
            console.log('[BotsManager] 🟢📉 RSI выход LONG (против тренда):', rsiExitLongAgainstTrendEl.value);
        }
        
        const rsiExitShortWithTrendEl = document.getElementById('rsiExitShortWithTrendGlobal');
        if (rsiExitShortWithTrendEl) {
            rsiExitShortWithTrendEl.value = autoBotConfig.rsi_exit_short_with_trend || 35;
            console.log('[BotsManager] 🔴📉 RSI выход SHORT (по тренду):', rsiExitShortWithTrendEl.value);
        }
        
        const rsiExitShortAgainstTrendEl = document.getElementById('rsiExitShortAgainstTrendGlobal');
        if (rsiExitShortAgainstTrendEl) {
            rsiExitShortAgainstTrendEl.value = autoBotConfig.rsi_exit_short_against_trend || 40;
            console.log('[BotsManager] 🔴📈 RSI выход SHORT (против тренда):', rsiExitShortAgainstTrendEl.value);
        }
        
        const rsiExitMinCandlesEl = document.getElementById('rsiExitMinCandlesGlobal');
        if (rsiExitMinCandlesEl) {
            const v = parseInt(autoBotConfig.rsi_exit_min_candles, 10);
            rsiExitMinCandlesEl.value = (!isNaN(v) && v >= 0) ? v : 0;
            console.log('[BotsManager] ⏱️ Мин. свечей до выхода по RSI:', rsiExitMinCandlesEl.value);
        }
        const rsiExitMinMinutesEl = document.getElementById('rsiExitMinMinutesGlobal');
        if (rsiExitMinMinutesEl) {
            const v = parseInt(autoBotConfig.rsi_exit_min_minutes, 10);
            rsiExitMinMinutesEl.value = (!isNaN(v) && v >= 0) ? v : 0;
        }
        const rsiExitMinMovePercentEl = document.getElementById('rsiExitMinMovePercentGlobal');
        if (rsiExitMinMovePercentEl) {
            const v = parseFloat(autoBotConfig.rsi_exit_min_move_percent);
            rsiExitMinMovePercentEl.value = (v !== undefined && !isNaN(v) && v >= 0) ? v : 0;
        }
        const exitWaitBreakevenEl = document.getElementById('exitWaitBreakevenWhenLoss');
        if (exitWaitBreakevenEl) {
            exitWaitBreakevenEl.checked = autoBotConfig.exit_wait_breakeven_when_loss === true;
        }
        
        // Торговые настройки (перенесены в блок Торговые параметры)
        const tradingEnabledEl = document.getElementById('tradingEnabled');
        if (tradingEnabledEl) {
            tradingEnabledEl.checked = autoBotConfig.trading_enabled !== false;
            console.log('[BotsManager] 🎛️ Реальная торговля:', tradingEnabledEl.checked);
        }
        
        const useTestServerEl1 = document.getElementById('useTestServer');
        if (useTestServerEl1) {
            useTestServerEl1.checked = autoBotConfig.use_test_server || false;
            console.log('[BotsManager] 🧪 Тестовый сервер:', useTestServerEl1.checked);
        }
        
        // ==========================================
        // ЗАЩИТНЫЕ МЕХАНИЗМЫ
        // ==========================================
        
        const maxLossPercentEl = document.getElementById('maxLossPercent');
        if (maxLossPercentEl) {
            maxLossPercentEl.value = autoBotConfig.max_loss_percent || 15.0;
            console.log('[BotsManager] 🛡️ Макс. убыток (стоп-лосс):', maxLossPercentEl.value);
        }
        
        const takeProfitPercentEl = document.getElementById('takeProfitPercent');
        if (takeProfitPercentEl) {
            takeProfitPercentEl.value = autoBotConfig.take_profit_percent ?? 5.0;
            console.log('[BotsManager] 🎯 Защитный TP (%):', takeProfitPercentEl.value);
        }
        
        const closeAtProfitEnabledEl = document.getElementById('closeAtProfitEnabled');
        if (closeAtProfitEnabledEl) {
            closeAtProfitEnabledEl.checked = autoBotConfig.close_at_profit_enabled !== false;
            console.log('[BotsManager] 🎯 Закрывать по % прибыли:', closeAtProfitEnabledEl.checked);
        }
        
        const trailingStopActivationEl = document.getElementById('trailingStopActivation');
        if (trailingStopActivationEl) {
            const value = Number.parseFloat(autoBotConfig.trailing_stop_activation);
            trailingStopActivationEl.value = Number.isFinite(value) ? value : 20.0;
            console.log('[BotsManager] 📈 Активация trailing stop:', trailingStopActivationEl.value);
        }
        
        const trailingStopDistanceEl = document.getElementById('trailingStopDistance');
        if (trailingStopDistanceEl) {
            const value = Number.parseFloat(autoBotConfig.trailing_stop_distance);
            trailingStopDistanceEl.value = Number.isFinite(value) ? value : 5.0;
            console.log('[BotsManager] 📉 Расстояние trailing stop:', trailingStopDistanceEl.value);
        }

        const trailingTakeDistanceEl = document.getElementById('trailingTakeDistance');
        if (trailingTakeDistanceEl) {
            const value = autoBotConfig.trailing_take_distance;
            trailingTakeDistanceEl.value = (value !== undefined && value !== null) ? value : 0.5;
            console.log('[BotsManager] 🎯 Резервный trailing take:', trailingTakeDistanceEl.value);
        }

        const trailingUpdateIntervalEl = document.getElementById('trailingUpdateInterval');
        if (trailingUpdateIntervalEl) {
            const value = autoBotConfig.trailing_update_interval;
            trailingUpdateIntervalEl.value = (value !== undefined && value !== null) ? value : 3.0;
            console.log('[BotsManager] ⏱️ Интервал обновления трейлинга:', trailingUpdateIntervalEl.value);
        }
        
        const maxPositionHoursEl = document.getElementById('maxPositionHours');
        if (maxPositionHoursEl) {
            const hours = autoBotConfig.max_position_hours || 0;
            maxPositionHoursEl.value = Math.round(hours * 3600);
            console.log('[BotsManager] ⏰ Макс. время позиции (сек):', maxPositionHoursEl.value);
        }
        
        const breakEvenProtectionEl = document.getElementById('breakEvenProtection');
        if (breakEvenProtectionEl) {
            breakEvenProtectionEl.checked = autoBotConfig.break_even_protection !== false;
            console.log('[BotsManager] 🛡️ Защита безубыточности:', breakEvenProtectionEl.checked);
        }
        
        const breakEvenTriggerEl = document.getElementById('breakEvenTrigger');
        if (breakEvenTriggerEl) {
            // ✅ ИСПОЛЬЗУЕМ РЕАЛЬНОЕ ЗНАЧЕНИЕ ИЗ КОНФИГА, А НЕ ДЕФОЛТНОЕ
            const triggerValue = autoBotConfig.break_even_trigger_percent ?? autoBotConfig.break_even_trigger ?? 20.0;
            breakEvenTriggerEl.value = triggerValue;
            console.log('[BotsManager] 🎯 Триггер безубыточности:', breakEvenTriggerEl.value, '(из конфига:', autoBotConfig.break_even_trigger_percent ?? autoBotConfig.break_even_trigger, ')');
        }
        
        // ==========================================
        // ФИЛЬТРЫ ПО ТРЕНДУ
        // ==========================================
        
        const avoidDownTrendEl = document.getElementById('avoidDownTrend');
        if (avoidDownTrendEl) {
            // ✅ ИСПОЛЬЗУЕМ РЕАЛЬНОЕ ЗНАЧЕНИЕ ИЗ КОНФИГА, А НЕ ДЕФОЛТНОЕ
            const configValue = autoBotConfig.avoid_down_trend;
            avoidDownTrendEl.checked = configValue === true;
            console.log('[BotsManager] 📉 Избегать DOWN тренд:', avoidDownTrendEl.checked, '(из конфига:', configValue, ')');
        }
        
        const avoidUpTrendEl = document.getElementById('avoidUpTrend');
        if (avoidUpTrendEl) {
            // ✅ ИСПОЛЬЗУЕМ РЕАЛЬНОЕ ЗНАЧЕНИЕ ИЗ КОНФИГА, А НЕ ДЕФОЛТНОЕ
            const configValue = autoBotConfig.avoid_up_trend;
            avoidUpTrendEl.checked = configValue === true;
            console.log('[BotsManager] 📈 Избегать UP тренд:', avoidUpTrendEl.checked, '(из конфига:', configValue, ')');
        }
        
        // ==========================================
        // ПАРАМЕТРЫ АНАЛИЗА ТРЕНДА
        // ==========================================
        
        const trendDetectionEnabledEl = document.getElementById('trendDetectionEnabled');
        if (trendDetectionEnabledEl) {
            // ✅ ИСПОЛЬЗУЕМ РЕАЛЬНОЕ ЗНАЧЕНИЕ ИЗ КОНФИГА, А НЕ ДЕФОЛТНОЕ
            const configValue = autoBotConfig.trend_detection_enabled;
            trendDetectionEnabledEl.checked = configValue === true;
            console.log('[BotsManager] 🔍 Анализ трендов включен:', trendDetectionEnabledEl.checked, '(из конфига:', configValue, ')');
        }
        
        const trendAnalysisPeriodEl = document.getElementById('trendAnalysisPeriod');
        if (trendAnalysisPeriodEl && autoBotConfig.trend_analysis_period !== undefined) {
            trendAnalysisPeriodEl.value = autoBotConfig.trend_analysis_period;
            console.log('[BotsManager] 📊 Период анализа тренда:', trendAnalysisPeriodEl.value);
        }
        
        const trendPriceChangeThresholdEl = document.getElementById('trendPriceChangeThreshold');
        if (trendPriceChangeThresholdEl && autoBotConfig.trend_price_change_threshold !== undefined) {
            trendPriceChangeThresholdEl.value = autoBotConfig.trend_price_change_threshold;
            console.log('[BotsManager] 📈 Порог изменения цены:', trendPriceChangeThresholdEl.value);
        }
        
        const trendCandlesThresholdEl = document.getElementById('trendCandlesThreshold');
        if (trendCandlesThresholdEl && autoBotConfig.trend_candles_threshold !== undefined) {
            trendCandlesThresholdEl.value = autoBotConfig.trend_candles_threshold;
            console.log('[BotsManager] 🕯️ Порог свечей:', trendCandlesThresholdEl.value);
        }
        
        // ==========================================
        // СИСТЕМНЫЕ НАСТРОЙКИ
        // ==========================================
        const systemConfig = config.system || {};
        
        // Загружаем таймфрейм в select
        const timeframeSelect = document.getElementById('systemTimeframe');
        if (timeframeSelect && systemConfig.timeframe) {
            timeframeSelect.value = systemConfig.timeframe;
            const applyBtn = document.getElementById('applyTimeframeBtn');
            if (applyBtn) {
                applyBtn.dataset.currentTimeframe = systemConfig.timeframe;
            }
            console.log('[BotsManager] ⏱️ Таймфрейм загружен:', systemConfig.timeframe);
        }
        
        // Интервалы обновления - ТОЛЬКО из API, без значений по умолчанию
        const rsiUpdateIntervalEl = document.getElementById('rsiUpdateInterval');
        if (rsiUpdateIntervalEl && systemConfig.rsi_update_interval !== undefined) {
            rsiUpdateIntervalEl.value = systemConfig.rsi_update_interval;
            console.log('[BotsManager] 🔄 RSI интервал установлен:', systemConfig.rsi_update_interval, '(из API)');
        } else if (rsiUpdateIntervalEl) {
            console.warn('[BotsManager] ⚠️ RSI интервал не найден в API, оставляем поле пустым');
        } else {
            console.error('[BotsManager] ❌ Элемент rsiUpdateInterval не найден!');
        }
        
        const autoSaveIntervalEl = document.getElementById('autoSaveInterval');
        if (autoSaveIntervalEl && systemConfig.auto_save_interval !== undefined) {
            autoSaveIntervalEl.value = systemConfig.auto_save_interval;
            console.log('[BotsManager] 💾 Автосохранение интервал установлен:', systemConfig.auto_save_interval, '(из API)');
        } else if (autoSaveIntervalEl) {
            console.warn('[BotsManager] ⚠️ Автосохранение интервал не найден в API, оставляем поле пустым');
        } else {
            console.error('[BotsManager] ❌ Элемент autoSaveInterval не найден!');
        }
        
        // Миниграфики = интервал «Синхронизация позиций» (настройка убрана из UI). Автообновление UI всегда вкл.
        
        // Режим отладки
        const debugModeEl = document.getElementById('debugMode');
        if (debugModeEl) {
            debugModeEl.checked = systemConfig.debug_mode || false;
            console.log('[BotsManager] 🐛 Режим отладки:', debugModeEl.checked);
        }
        
        // Режим маржи Bybit (auto / cross / isolated)
        const bybitMarginModeEl = document.getElementById('bybitMarginMode');
        if (bybitMarginModeEl && systemConfig.bybit_margin_mode !== undefined) {
            const val = (systemConfig.bybit_margin_mode || 'auto').toLowerCase();
            bybitMarginModeEl.value = ['auto', 'cross', 'isolated'].includes(val) ? val : 'auto';
            console.log('[BotsManager] 📊 Режим маржи Bybit:', bybitMarginModeEl.value);
        } else if (bybitMarginModeEl) {
            bybitMarginModeEl.value = 'auto';
        }
        
        // ==========================================
        // ИНТЕРВАЛЫ СИНХРОНИЗАЦИИ И ОЧИСТКИ
        // ==========================================
        // Единый период обновления для всех RSI-зависимых данных в UI (боты, списки, фильтры, мониторинг) = position_sync_interval
        const positionSyncIntervalEl = document.getElementById('positionSyncInterval');
        console.log('[BotsManager] 🔍 Поиск элемента positionSyncInterval:', positionSyncIntervalEl);
        console.log('[BotsManager] 🔍 systemConfig.position_sync_interval:', systemConfig.position_sync_interval);
        if (positionSyncIntervalEl && systemConfig.position_sync_interval !== undefined) {
            positionSyncIntervalEl.value = systemConfig.position_sync_interval;
            // Минимум 5 сек — иначе интерфейс мигает как стробоскоп
            this.refreshInterval = Math.max(5000, systemConfig.position_sync_interval * 1000);
            console.log('[BotsManager] 🔄 Синхронизация позиций и период обновления UI (RSI, боты, мониторинг):', systemConfig.position_sync_interval, 'сек');
        } else if (positionSyncIntervalEl) {
            positionSyncIntervalEl.value = 600;
            this.refreshInterval = 600 * 1000;
            console.log('[BotsManager] 🔄 Position Sync и обновление UI по умолчанию: 600 сек');
        } else {
            console.error('[BotsManager] ❌ Элемент positionSyncInterval не найден!');
            this.refreshInterval = 600 * 1000;
        }
        
        // Интервал очистки неактивных ботов
        const inactiveBotCleanupIntervalEl = document.getElementById('inactiveBotCleanupInterval');
        if (inactiveBotCleanupIntervalEl && systemConfig.inactive_bot_cleanup_interval !== undefined) {
            inactiveBotCleanupIntervalEl.value = systemConfig.inactive_bot_cleanup_interval;
            console.log('[BotsManager] 🧹 Inactive Bot Cleanup интервал установлен:', systemConfig.inactive_bot_cleanup_interval, 'сек (из API)');
        } else if (inactiveBotCleanupIntervalEl) {
            inactiveBotCleanupIntervalEl.value = 600; // 10 минут по умолчанию
            console.log('[BotsManager] 🧹 Inactive Bot Cleanup интервал установлен по умолчанию: 600 сек');
        }
        
        // Таймаут неактивных ботов
        const inactiveBotTimeoutEl = document.getElementById('inactiveBotTimeout');
        if (inactiveBotTimeoutEl && systemConfig.inactive_bot_timeout !== undefined) {
            inactiveBotTimeoutEl.value = systemConfig.inactive_bot_timeout;
            console.log('[BotsManager] ⏰ Inactive Bot Timeout установлен:', systemConfig.inactive_bot_timeout, 'сек (из API)');
        } else if (inactiveBotTimeoutEl) {
            inactiveBotTimeoutEl.value = 600; // 10 минут по умолчанию
            console.log('[BotsManager] ⏰ Inactive Bot Timeout установлен по умолчанию: 600 сек');
        }
        
        // Интервал настройки стоп-лоссов
        const stopLossSetupIntervalEl = document.getElementById('stopLossSetupInterval');
        if (stopLossSetupIntervalEl && systemConfig.stop_loss_setup_interval !== undefined) {
            stopLossSetupIntervalEl.value = systemConfig.stop_loss_setup_interval;
            console.log('[BotsManager] 🛡️ Stop Loss Setup интервал установлен:', systemConfig.stop_loss_setup_interval, 'сек (из API)');
        } else if (stopLossSetupIntervalEl) {
            stopLossSetupIntervalEl.value = 300; // 5 минут по умолчанию
            console.log('[BotsManager] 🛡️ Stop Loss Setup интервал установлен по умолчанию: 300 сек');
        }
        
        // ==========================================
        // RSI ВРЕМЕННОЙ ФИЛЬТР
        // ==========================================
        
        const rsiTimeFilterEnabledEl = document.getElementById('rsiTimeFilterEnabled');
        if (rsiTimeFilterEnabledEl) {
            rsiTimeFilterEnabledEl.checked = autoBotConfig.rsi_time_filter_enabled !== false;
            console.log('[BotsManager] ⏰ RSI временной фильтр:', rsiTimeFilterEnabledEl.checked);
        }
        
        const rsiTimeFilterCandlesEl = document.getElementById('rsiTimeFilterCandles');
        if (rsiTimeFilterCandlesEl) {
            rsiTimeFilterCandlesEl.value = autoBotConfig.rsi_time_filter_candles || 8;
            console.log('[BotsManager] 🕐 RSI временной фильтр (свечей):', rsiTimeFilterCandlesEl.value);
        }
        
        const rsiTimeFilterUpperEl = document.getElementById('rsiTimeFilterUpper');
        if (rsiTimeFilterUpperEl) {
            rsiTimeFilterUpperEl.value = autoBotConfig.rsi_time_filter_upper || 65;
            console.log('[BotsManager] 📈 RSI временной фильтр (верхняя граница):', rsiTimeFilterUpperEl.value);
        }
        
        const rsiTimeFilterLowerEl = document.getElementById('rsiTimeFilterLower');
        if (rsiTimeFilterLowerEl) {
            rsiTimeFilterLowerEl.value = autoBotConfig.rsi_time_filter_lower || 35;
            console.log('[BotsManager] 📉 RSI временной фильтр (нижняя граница):', rsiTimeFilterLowerEl.value);
        }
        
        // ==========================================
        // EXITSCAM ФИЛЬТР
        // ==========================================
        
        const exitScamEnabledEl = document.getElementById('exitScamEnabled');
        if (exitScamEnabledEl) {
            exitScamEnabledEl.checked = autoBotConfig.exit_scam_enabled !== false;
            console.log('[BotsManager] 🛡️ ExitScam фильтр:', exitScamEnabledEl.checked);
        }
        const exitScamAutoLearnEl = document.getElementById('exitScamAutoLearnEnabled');
        if (exitScamAutoLearnEl) {
            exitScamAutoLearnEl.checked = autoBotConfig.exit_scam_auto_learn_enabled === true;
        }
        
        const exitScamCandlesEl = document.getElementById('exitScamCandles');
        if (exitScamCandlesEl) {
            exitScamCandlesEl.value = autoBotConfig.exit_scam_candles || 10;
            console.log('[BotsManager] 📊 ExitScam анализ свечей:', exitScamCandlesEl.value);
        }
        
        const exitScamSingleCandlePercentEl = document.getElementById('exitScamSingleCandlePercent');
        if (exitScamSingleCandlePercentEl) {
            exitScamSingleCandlePercentEl.value = autoBotConfig.exit_scam_single_candle_percent || 15.0;
            console.log('[BotsManager] ⚡ ExitScam лимит одной свечи:', exitScamSingleCandlePercentEl.value);
        }
        
        const exitScamMultiCandleCountEl = document.getElementById('exitScamMultiCandleCount');
        if (exitScamMultiCandleCountEl) {
            exitScamMultiCandleCountEl.value = autoBotConfig.exit_scam_multi_candle_count || 4;
            console.log('[BotsManager] 📈 ExitScam свечей для анализа:', exitScamMultiCandleCountEl.value);
        }
        
        const exitScamMultiCandlePercentEl = document.getElementById('exitScamMultiCandlePercent');
        if (exitScamMultiCandlePercentEl) {
            exitScamMultiCandlePercentEl.value = autoBotConfig.exit_scam_multi_candle_percent || 50.0;
            console.log('[BotsManager] 📊 ExitScam суммарный лимит:', exitScamMultiCandlePercentEl.value);
        }
        const exitScamTimeframeEl = document.getElementById('exitScamTimeframe');
        if (exitScamTimeframeEl) {
            const tf = autoBotConfig.exit_scam_timeframe || '1m';
            exitScamTimeframeEl.value = tf;
        }
        const exitScamEffectiveScaleEl = document.getElementById('exitScamEffectiveScale');
        if (exitScamEffectiveScaleEl) {
            const single = autoBotConfig.exit_scam_effective_single_pct ?? autoBotConfig.exit_scam_single_candle_percent ?? 15;
            const multi = autoBotConfig.exit_scam_effective_multi_pct ?? autoBotConfig.exit_scam_multi_candle_percent ?? 50;
            const n = autoBotConfig.exit_scam_multi_candle_count || 4;
            exitScamEffectiveScaleEl.textContent = `Одна свеча: ${Number(single)}% | суммарно за ${n} св.: ${Number(multi)}% (как в конфиге)`;
        }
        // ==========================================
        // НАСТРОЙКИ ЗРЕЛОСТИ МОНЕТ
        // ==========================================
        
        const enableMaturityCheckEl = document.getElementById('enableMaturityCheck');
        if (enableMaturityCheckEl) {
            enableMaturityCheckEl.checked = autoBotConfig.enable_maturity_check !== false;
            console.log('[BotsManager] 🔍 Проверка зрелости:', enableMaturityCheckEl.checked);
        }
        
        const minCandlesForMaturityEl = document.getElementById('minCandlesForMaturity');
        if (minCandlesForMaturityEl) {
            minCandlesForMaturityEl.value = autoBotConfig.min_candles_for_maturity || 200;
            console.log('[BotsManager] 📊 Мин. свечей для зрелости:', minCandlesForMaturityEl.value);
        }
        
        const minRsiLowEl = document.getElementById('minRsiLow');
        if (minRsiLowEl) {
            minRsiLowEl.value = autoBotConfig.min_rsi_low || 35;
            console.log('[BotsManager] 📉 Мин. RSI low:', minRsiLowEl.value);
        }
        
        const maxRsiHighEl = document.getElementById('maxRsiHigh');
        if (maxRsiHighEl) {
            maxRsiHighEl.value = autoBotConfig.max_rsi_high || 65;
            console.log('[BotsManager] 📈 Макс. RSI high:', maxRsiHighEl.value);
        }
        
        // ==========================================
        // ENHANCED RSI (УЛУЧШЕННАЯ СИСТЕМА RSI)
        // ==========================================
        
        const enhancedRsiEnabledEl = document.getElementById('enhancedRsiEnabled');
        if (enhancedRsiEnabledEl) {
            enhancedRsiEnabledEl.checked = systemConfig.enhanced_rsi_enabled || false;
            console.log('[BotsManager] 🧠 Enhanced RSI включен:', enhancedRsiEnabledEl.checked);
        }
        
        const enhancedRsiVolumeConfirmEl = document.getElementById('enhancedRsiVolumeConfirm');
        if (enhancedRsiVolumeConfirmEl) {
            enhancedRsiVolumeConfirmEl.checked = systemConfig.enhanced_rsi_require_volume_confirmation || false;
            console.log('[BotsManager] 📊 Enhanced RSI требует подтверждение объёмом:', enhancedRsiVolumeConfirmEl.checked);
        }
        
        const enhancedRsiDivergenceConfirmEl = document.getElementById('enhancedRsiDivergenceConfirm');
        if (enhancedRsiDivergenceConfirmEl) {
            enhancedRsiDivergenceConfirmEl.checked = systemConfig.enhanced_rsi_require_divergence_confirmation || false;
            console.log('[BotsManager] 📈 Enhanced RSI требует дивергенцию:', enhancedRsiDivergenceConfirmEl.checked);
        }
        
        const enhancedRsiUseStochRsiEl = document.getElementById('enhancedRsiUseStochRsi');
        if (enhancedRsiUseStochRsiEl) {
            enhancedRsiUseStochRsiEl.checked = systemConfig.enhanced_rsi_use_stoch_rsi || false;
            console.log('[BotsManager] 📊 Enhanced RSI использует Stoch RSI:', enhancedRsiUseStochRsiEl.checked);
        }
        
        const rsiExtremeZoneTimeoutEl = document.getElementById('rsiExtremeZoneTimeout');
        if (rsiExtremeZoneTimeoutEl) {
            rsiExtremeZoneTimeoutEl.value = systemConfig.rsi_extreme_zone_timeout || 3;
            console.log('[BotsManager] ⏰ RSI экстремальная зона таймаут:', rsiExtremeZoneTimeoutEl.value);
        }
        
        const rsiExtremeOversoldEl = document.getElementById('rsiExtremeOversold');
        if (rsiExtremeOversoldEl) {
            rsiExtremeOversoldEl.value = systemConfig.rsi_extreme_oversold || 20;
            console.log('[BotsManager] 📉 RSI экстремальный oversold:', rsiExtremeOversoldEl.value);
        }
        
        const rsiExtremeOverboughtEl = document.getElementById('rsiExtremeOverbought');
        if (rsiExtremeOverboughtEl) {
            rsiExtremeOverboughtEl.value = systemConfig.rsi_extreme_overbought || 80;
            console.log('[BotsManager] 📈 RSI экстремальный overbought:', rsiExtremeOverboughtEl.value);
        }
        const rsiVolumeMultiplierEl = document.getElementById('rsiVolumeMultiplier');
        if (rsiVolumeMultiplierEl) {
            rsiVolumeMultiplierEl.value = systemConfig.rsi_volume_confirmation_multiplier || 1.2;
            console.log('[BotsManager] 📊 RSI множитель объёма:', rsiVolumeMultiplierEl.value);
        }
        
        const rsiDivergenceLookbackEl = document.getElementById('rsiDivergenceLookback');
        if (rsiDivergenceLookbackEl) {
            rsiDivergenceLookbackEl.value = systemConfig.rsi_divergence_lookback || 10;
            console.log('[BotsManager] 🔍 RSI период поиска дивергенций:', rsiDivergenceLookbackEl.value);
        }
        
        // ==========================================
        // НАБОР ПОЗИЦИЙ ЛИМИТНЫМИ ОРДЕРАМИ
        // ==========================================
        
        const limitOrdersEnabledEl = document.getElementById('limitOrdersEntryEnabled');
        // Используем уже объявленные переменные positionSizeEl и positionModeEl из блока торговых параметров
        const limitPositionSizeEl = document.getElementById('defaultPositionSize');
        const limitPositionModeEl = document.getElementById('defaultPositionMode');
        
        if (limitOrdersEnabledEl) {
            const isEnabled = autoBotConfig.limit_orders_entry_enabled || false;
            // ✅ Устанавливаем значение БЕЗ триггера события change (чтобы не сработало автосохранение)
            // Используем прямую установку свойства, а не событие
            limitOrdersEnabledEl.checked = isEnabled;
            
            // ✅ Вручную обновляем UI без триггера события change
            const configDiv = document.getElementById('limitOrdersConfig');
            if (configDiv) {
                configDiv.style.display = isEnabled ? 'block' : 'none';
            }
            
            // Деактивируем настройку "Размер позиции" при включении лимитных ордеров
            if (limitPositionSizeEl) {
                limitPositionSizeEl.disabled = isEnabled;
                limitPositionSizeEl.style.opacity = isEnabled ? '0.5' : '1';
                limitPositionSizeEl.style.cursor = isEnabled ? 'not-allowed' : 'text';
            }
            if (limitPositionModeEl) {
                limitPositionModeEl.disabled = isEnabled;
                limitPositionModeEl.style.opacity = isEnabled ? '0.5' : '1';
                limitPositionModeEl.style.cursor = isEnabled ? 'not-allowed' : 'pointer';
            }
            
            // ✅ Обновляем состояние кнопки "По умолчанию"
            const resetBtn = document.getElementById('resetLimitOrdersBtn');
            if (resetBtn) {
                resetBtn.disabled = !isEnabled;
                resetBtn.style.opacity = isEnabled ? '1' : '0.5';
                resetBtn.style.cursor = isEnabled ? 'pointer' : 'not-allowed';
            }
            
            console.log('[BotsManager] 📊 Набор позиций лимитными ордерами:', isEnabled);
        }
        
        // Загружаем настройки лимитных ордеров
        const percentSteps = autoBotConfig.limit_orders_percent_steps || [1, 2, 3, 4, 5];
        const marginAmounts = autoBotConfig.limit_orders_margin_amounts || [5, 5, 5, 5, 5];
        const listEl = document.getElementById('limitOrdersList');
        if (listEl) {
            // ✅ Инициализируем UI ПЕРЕД загрузкой данных, но ПОСЛЕ установки значения toggle
            // Это гарантирует, что обработчики установлены, но не перезаписывают значение
            try {
                this.initializeLimitOrdersUI();
            } catch (e) {
                console.warn('[BotsManager] ⚠️ Ошибка инициализации UI лимитных ордеров:', e);
            }
            
            // ✅ Убеждаемся, что значение toggle не изменилось после инициализации UI
            if (limitOrdersEnabledEl) {
                const currentEnabled = limitOrdersEnabledEl.checked;
                const shouldBeEnabled = autoBotConfig.limit_orders_entry_enabled || false;
                if (currentEnabled !== shouldBeEnabled) {
                    // Если значение изменилось, восстанавливаем его
                    limitOrdersEnabledEl.checked = shouldBeEnabled;
                    const configDiv = document.getElementById('limitOrdersConfig');
                    if (configDiv) {
                        configDiv.style.display = shouldBeEnabled ? 'block' : 'none';
                    }
                }
            }
            
            listEl.innerHTML = ''; // Очищаем список
            for (let i = 0; i < Math.max(percentSteps.length, marginAmounts.length); i++) {
                try {
                    this.addLimitOrderRow(
                        percentSteps[i] || 0,
                        marginAmounts[i] || 0
                    );
                } catch (e) {
                    console.warn('[BotsManager] ⚠️ Ошибка добавления строки лимитного ордера:', e);
                }
            }
        }
        
        // ==========================================
        // ПАРАМЕТРЫ ОПРЕДЕЛЕНИЯ ТРЕНДА
        // ==========================================
        
        // ❌ УСТАРЕВШИЕ НАСТРОЙКИ EMA - УБРАНЫ (больше не используются)
        // Тренд теперь определяется простым анализом цены (% изменения и растущие/падающие свечи)
        
        // Сбрасываем флаг программного изменения после заполнения формы
        // Используем setTimeout чтобы гарантировать, что все события завершились
        setTimeout(() => {
            this.isProgrammaticChange = false;
        }, 100);
        
        console.log('[BotsManager] ✅ Форма заполнена данными из API');
    },
            showConfigurationLoading(show) {
        // ✅ БЕЗ БЛОКИРОВКИ: Просто логируем, но не блокируем элементы
        const configContainer = document.getElementById('configTab');
        if (!configContainer) return;
        
        if (show) {
            // Добавляем класс загрузки для визуального индикатора
            configContainer.classList.add('loading');
            console.log('[BotsManager] ⏳ Конфигурация загружается...');
        } else {
            // Убираем класс загрузки
            configContainer.classList.remove('loading');
            console.log('[BotsManager] ✅ Конфигурация загружена');
            
            // ✅ КРИТИЧЕСКИ ВАЖНО: Убеждаемся что все элементы разблокированы
            const allInputs = configContainer.querySelectorAll('input, select, textarea, button');
            allInputs.forEach(el => {
                el.removeAttribute('disabled');
                el.disabled = false;
                el.style.pointerEvents = 'auto';
                el.style.opacity = '1';
                el.style.cursor = 'pointer';
            });
        }
    },
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
                
                const autoBotData = await autoBotResponse.json();
                if (autoBotData.success) {
                    console.log('[BotsManager] ✅ Auto Bot конфигурация сохранена');
                }
            }
            
            // ✅ Проверяем, что есть данные для отправки
            if (!defaultConfig.system || Object.keys(defaultConfig.system).length === 0) {
                console.log('[BotsManager] ⚠️ System конфигурация пуста, пропускаем сохранение');
            } else {
                // Сохраняем системные настройки
                const systemResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(defaultConfig.system)
                });
                
                const systemData = await systemResponse.json();
                if (systemData.success) {
                    console.log('[BotsManager] ✅ System конфигурация сохранена');
                }
            }
            
            console.log('[BotsManager] ✅ Конфигурация по умолчанию обработана');
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения конфигурации по умолчанию:', error);
            throw error;
        }
    }
    /**
     * Конвертирует camelCase в snake_case для маппинга ID элементов на ключи конфигурации
     */,
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
        
        // Используем прямое маппинг если есть
        if (directMappings[elementId]) {
            return directMappings[elementId];
        }
        
        // Иначе конвертируем camelCase в snake_case
        return this.camelToSnake(elementId);
    },
            collectConfigurationData() {
        console.log('[BotsManager] 📋 Сбор данных конфигурации (автоматический режим)...');
        
        // ✅ РАБОТАЕМ НАПРЯМУЮ С КЭШИРОВАННОЙ КОНФИГУРАЦИЕЙ ИЗ БЭКЕНДА
        // Это гарантирует, что мы используем реальные значения из файла конфига, а не дефолтные из HTML
        if (!this.cachedAutoBotConfig) {
            console.warn('[BotsManager] ⚠️ cachedAutoBotConfig не загружен, используем пустой объект');
            return {
                autoBot: {},
                system: {}
            };
        }
        
        // ✅ ГЛУБОКОЕ КОПИРОВАНИЕ КЭШИРОВАННОЙ КОНФИГУРАЦИИ
        const autoBotConfig = JSON.parse(JSON.stringify(this.cachedAutoBotConfig));
        if (!autoBotConfig.default_position_mode) {
            autoBotConfig.default_position_mode = 'usdt';
        }
        
        // ✅ АВТОМАТИЧЕСКИЙ СБОР ВСЕХ ПОЛЕЙ КОНФИГУРАЦИИ
        const configTab = document.getElementById('configTab');
        if (!configTab) {
            console.warn('[BotsManager] ⚠️ configTab не найден');
            return { autoBot: autoBotConfig, system: {} };
        }
        
        // Находим ВСЕ поля конфигурации: input, select, checkbox
        // ✅ КРИТИЧЕСКИ ВАЖНО: Включаем скрытые input (hidden) для scope
        const autoBotInputs = configTab.querySelectorAll('input[type="number"], input[type="text"], input[type="hidden"], input[type="checkbox"], select');
        
        // Также добавляем поля из секции AI, если она существует
        const aiConfigSection = document.getElementById('aiConfigSection');
        if (aiConfigSection) {
            const aiInputs = aiConfigSection.querySelectorAll('input[type="number"], input[type="text"], input[type="hidden"], input[type="checkbox"], select');
            const uniqueInputs = new Set([...autoBotInputs, ...aiInputs]);
            this.collectFieldsFromElements(Array.from(uniqueInputs), autoBotConfig);
        } else {
            this.collectFieldsFromElements(Array.from(autoBotInputs), autoBotConfig);
        }
        
        // ✅ ОБРАБОТКА ДИНАМИЧЕСКИХ ПОЛЕЙ ЛИМИТНЫХ ОРДЕРОВ
        // Сначала обрабатываем toggle для limit_orders_entry_enabled
        const limitOrdersEntryEnabledEl = document.getElementById('limitOrdersEntryEnabled');
        if (limitOrdersEntryEnabledEl) {
            const enabled = limitOrdersEntryEnabledEl.checked;
            // Всегда обновляем значение, чтобы оно сохранялось при обычном сохранении конфигурации
            autoBotConfig.limit_orders_entry_enabled = enabled;
            console.log('[BotsManager] 🔄 Обновлен limit_orders_entry_enabled:', enabled);
        }
        // ✅ ExitScam: всегда берём из DOM, чтобы после перезапуска сервера сохранялось отключение
        const exitScamEnabledEl = document.getElementById('exitScamEnabled');
        const exitScamCandlesEl = document.getElementById('exitScamCandles');
        const exitScamSingleEl = document.getElementById('exitScamSingleCandlePercent');
        const exitScamMultiCountEl = document.getElementById('exitScamMultiCandleCount');
        const exitScamMultiPercentEl = document.getElementById('exitScamMultiCandlePercent');
        if (exitScamEnabledEl) {
            autoBotConfig.exit_scam_enabled = exitScamEnabledEl.checked;
        }
        if (exitScamCandlesEl && exitScamCandlesEl.value !== '') {
            const v = parseInt(exitScamCandlesEl.value, 10);
            if (!isNaN(v)) autoBotConfig.exit_scam_candles = v;
        }
        if (exitScamSingleEl && exitScamSingleEl.value !== '') {
            const v = parseFloat(exitScamSingleEl.value);
            if (!isNaN(v)) autoBotConfig.exit_scam_single_candle_percent = v;
        }
        if (exitScamMultiCountEl && exitScamMultiCountEl.value !== '') {
            const v = parseInt(exitScamMultiCountEl.value, 10);
            if (!isNaN(v)) autoBotConfig.exit_scam_multi_candle_count = v;
        }
        if (exitScamMultiPercentEl && exitScamMultiPercentEl.value !== '') {
            const v = parseFloat(exitScamMultiPercentEl.value);
            if (!isNaN(v)) autoBotConfig.exit_scam_multi_candle_percent = v;
        }
        const exitScamAutoLearnEl = document.getElementById('exitScamAutoLearnEnabled');
        if (exitScamAutoLearnEl) {
            autoBotConfig.exit_scam_auto_learn_enabled = exitScamAutoLearnEl.checked;
        }
        // ✅ КРИТИЧНО: exit_wait_breakeven_when_loss — всегда из DOM (иначе не сохраняется при переключении view)
        const exitWaitBreakevenEl = document.getElementById('exitWaitBreakevenWhenLoss');
        if (exitWaitBreakevenEl) {
            autoBotConfig.exit_wait_breakeven_when_loss = exitWaitBreakevenEl.checked;
        }
        const rsiLimitEntryEl = document.getElementById('rsiLimitEntryEnabled');
        if (rsiLimitEntryEl) {
            autoBotConfig.rsi_limit_entry_enabled = rsiLimitEntryEl.checked;
        }
        const rsiLimitOffsetEl = document.getElementById('rsiLimitOffsetPercentGlobal');
        if (rsiLimitOffsetEl && rsiLimitOffsetEl.value !== '') {
            const v = parseFloat(rsiLimitOffsetEl.value);
            if (!isNaN(v) && v >= 0) autoBotConfig.rsi_limit_offset_percent = v;
        }
        const rsiLimitExitEl = document.getElementById('rsiLimitExitEnabled');
        if (rsiLimitExitEl) {
            autoBotConfig.rsi_limit_exit_enabled = rsiLimitExitEl.checked;
        }
        const rsiLimitExitOffsetEl = document.getElementById('rsiLimitExitOffsetPercentGlobal');
        if (rsiLimitExitOffsetEl && rsiLimitExitOffsetEl.value !== '') {
            const v = parseFloat(rsiLimitExitOffsetEl.value);
            if (!isNaN(v) && v >= 0) autoBotConfig.rsi_limit_exit_offset_percent = v;
        }
        
        const limitOrderRows = document.querySelectorAll('.limit-order-row');
        if (limitOrderRows.length > 0) {
            const percentSteps = [];
            const marginAmounts = [];
            
            limitOrderRows.forEach(row => {
                const percentEl = row.querySelector('.limit-order-percent');
                const marginEl = row.querySelector('.limit-order-margin');
                
                if (percentEl) {
                    const percent = parseFloat(percentEl.value);
                    if (!isNaN(percent)) {
                        percentSteps.push(percent);
                    } else {
                        percentSteps.push(0); // Добавляем 0 если значение невалидно
                    }
                }
                
                if (marginEl) {
                    const margin = parseFloat(marginEl.value);
                    if (!isNaN(margin)) {
                        marginAmounts.push(margin);
                    } else {
                        marginAmounts.push(0); // Добавляем 0 если значение невалидно
                    }
                }
            });
            
            // ✅ ВСЕГДА обновляем значения лимитных ордеров (для автосохранения)
            // Это гарантирует, что изменения сохраняются даже если originalConfig не обновлен
            if (percentSteps.length > 0 || marginAmounts.length > 0) {
                autoBotConfig.limit_orders_percent_steps = percentSteps;
                autoBotConfig.limit_orders_margin_amounts = marginAmounts;
                console.log('[BotsManager] 🔄 Обновлены настройки лимитных ордеров:', { percentSteps, marginAmounts });
            }
        }
        
        // ✅ СБОР СИСТЕМНЫХ НАСТРОЕК (автоматически из системных полей)
        const systemConfig = {};
        
        // ✅ Список системных настроек Enhanced RSI и других системных настроек
        const systemConfigKeys = [
            'enhanced_rsi_enabled',
            'enhanced_rsi_require_volume_confirmation',
            'enhanced_rsi_require_divergence_confirmation',
            'enhanced_rsi_use_stoch_rsi',
            'rsi_extreme_zone_timeout',
            'rsi_extreme_oversold',
            'rsi_extreme_overbought',
            'rsi_volume_confirmation_multiplier',
            'rsi_divergence_lookback',
            'rsi_update_interval',
            'auto_save_interval',
            'debug_mode',
            'refresh_interval',
            'position_sync_interval',
            'inactive_bot_cleanup_interval',
            'inactive_bot_timeout',
            'stop_loss_setup_interval'
        ];
        
        // ✅ Находим все системные поля в configTab (используем более надежный подход)
        // Сначала собираем все поля Enhanced RSI по конкретным ID
        const enhancedRsiFields = [
            'enhancedRsiEnabled',
            'enhancedRsiVolumeConfirm',
            'enhancedRsiDivergenceConfirm',
            'enhancedRsiUseStochRsi',
            'rsiExtremeZoneTimeout',
            'rsiExtremeOversold',
            'rsiExtremeOverbought',
            'rsiVolumeMultiplier',
            'rsiDivergenceLookback'
        ];
        
        enhancedRsiFields.forEach(fieldId => {
            const element = document.getElementById(fieldId);
            if (element && !element.closest('#limitOrdersList') && !element.closest('.limit-order-row')) {
                const configKey = this.mapElementIdToConfigKey(fieldId);
                if (configKey && systemConfigKeys.includes(configKey)) {
                    let value;
                    if (element.type === 'checkbox') {
                        value = element.checked;
                    } else if (element.type === 'number') {
                        const numValue = parseFloat(element.value);
                        value = isNaN(numValue) ? undefined : numValue;
                    } else {
                        value = element.value;
                    }
                    
                    if (value !== undefined && value !== null) {
                        systemConfig[configKey] = value;
                        console.log(`[BotsManager] ✅ Собрана Enhanced RSI настройка ${configKey}:`, value);
                    }
                }
            }
        });
        
        // ✅ Находим остальные системные поля (интервалы, режимы и т.д.)
        // Используем селектор, который ищет по ID (нечувствительный к регистру через проверку)
        const allInputs = configTab.querySelectorAll('input, select');
        allInputs.forEach(element => {
            if (!element.id || element.closest('#limitOrdersList') || element.closest('.limit-order-row')) {
                return; // Пропускаем динамические поля лимитных ордеров
            }
            
            // Пропускаем поля Enhanced RSI, которые уже обработаны выше
            if (enhancedRsiFields.includes(element.id)) {
                return;
            }
            
            const configKey = this.mapElementIdToConfigKey(element.id);
            if (!configKey) {
                return;
            }
            
            // ✅ Проверяем, что это системная настройка (либо начинается с system_, либо в списке системных настроек)
            const isSystemConfig = configKey.startsWith('system_') || systemConfigKeys.includes(configKey);
            
            if (isSystemConfig) {
                const systemKey = configKey.startsWith('system_') ? configKey.replace('system_', '') : configKey;
                let value;
                if (element.type === 'checkbox') {
                    value = element.checked;
                } else if (element.type === 'number') {
                    const numValue = parseFloat(element.value);
                    value = isNaN(numValue) ? undefined : numValue;
                } else {
                    value = element.value;
                }
                
                if (value !== undefined && value !== null) {
                    systemConfig[systemKey] = value;
                    console.log(`[BotsManager] ✅ Собрана системная настройка ${systemKey}:`, value);
                }
            }
        });
        
        // Период обновления RSI/UI везде берётся из «Синхронизация позиций»
        if (systemConfig.position_sync_interval != null) {
            systemConfig.refresh_interval = systemConfig.position_sync_interval;
        }
        
        return {
            autoBot: autoBotConfig,
            system: systemConfig
        };
    }
    
    /**
     * Собирает значения из элементов формы и обновляет конфигурацию
     */,
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
            
            const configKey = this.mapElementIdToConfigKey(element.id);
            if (!configKey) {
                return;
            }
            
            // Определяем значение в зависимости от типа элемента
            let value;
            if (element.type === 'checkbox') {
                value = element.checked;
            } else if (element.type === 'number') {
                const numValue = parseFloat(element.value);
                value = isNaN(numValue) ? undefined : numValue;
            } else if (element.tagName === 'SELECT') {
                value = element.value;
            } else {
                value = element.value;
            }
            
            // Применяем значение только если оно изменилось
            const originalValue = this.originalConfig?.autoBot?.[configKey];
            
            // ✅ Макс. время позиции: в UI в секундах, в конфиге — в часах
            if (configKey === 'max_position_hours' && typeof value === 'number') {
                value = value / 3600;
            }
            // ✅ КРИТИЧЕСКИ ВАЖНО: Специальная обработка для scope - всегда обновляем если значение изменилось
            if (configKey === 'scope') {
                if (value !== undefined && value !== null) {
                    config[configKey] = value;
                    console.log(`[BotsManager] 🔄 scope собран из UI: ${value} (было в originalConfig: ${originalValue || 'undefined'})`);
                }
                return; // Пропускаем остальную логику для scope
            }
            
            if (value !== undefined && value !== null) {
                // Если originalValue undefined (новое поле), всегда устанавливаем значение
                if (originalValue === undefined) {
                    config[configKey] = value;
                    console.log(`[BotsManager] 🔄 Авто-применено (новое поле): ${configKey} = ${value}`);
                }
                // Для булевых значений
                else if (typeof value === 'boolean') {
                    const normalizedOriginal = originalValue === true ? true : false;
                    if (value !== normalizedOriginal) {
                        config[configKey] = value;
                        console.log(`[BotsManager] 🔄 Авто-применено: ${configKey} = ${value} (было ${normalizedOriginal})`);
                    }
                }
                // Для чисел: сравниваем с точностью 0.01
                else if (typeof value === 'number' && typeof originalValue === 'number') {
                    if (Math.abs(value - originalValue) > 0.01) {
                        config[configKey] = value;
                        console.log(`[BotsManager] 🔄 Авто-применено: ${configKey} = ${value} (было ${originalValue})`);
                    }
                }
                // Для остальных типов: точное сравнение
                else if (value !== originalValue) {
                    config[configKey] = value;
                    console.log(`[BotsManager] 🔄 Авто-применено: ${configKey} = ${value} (было ${originalValue})`);
                }
            }
        });
    },
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
            const data = await res.json();
            if (data.success) {
                this.showNotification('Параметры Full AI сохранены', 'success');
            } else {
                this.showNotification('Ошибка сохранения параметров Full AI: ' + (data.error || res.status), 'error');
            }
        } catch (e) {
            console.error('[BotsManager] saveFullaiAdaptiveConfig:', e);
            this.showNotification('Ошибка сохранения параметров Full AI', 'error');
        }
    },
            async saveSystemSettings() {
        console.log('[BotsManager] 💾 Сохранение системных настроек...');
        try {
            const config = this.collectConfigurationData();
            const systemSettings = { ...config.system };
            const bybitMarginEl = document.getElementById('bybitMarginMode');
            if (bybitMarginEl) {
                const v = (bybitMarginEl.value || 'auto').toLowerCase();
                systemSettings.bybit_margin_mode = ['auto', 'cross', 'isolated'].includes(v) ? v : 'auto';
            }
            
            await this.sendConfigUpdate('system-config', systemSettings, 'Системные настройки');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения системных настроек:', error);
            this.showNotification('❌ Ошибка сохранения системных настроек', 'error');
        }
    }
    
    /**
     * Сохраняет весь блок: торговые параметры и RSI выходы (объединённая кнопка)
     */,
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
                    const el = document.getElementById('rsiLimitOffsetPercentGlobal');
                    if (el && el.value !== '') {
                        const v = parseFloat(el.value);
                        return !isNaN(v) && v >= 0 ? v : 0.2;
                    }
                    return parseFloat(config.autoBot.rsi_limit_offset_percent) || 0.2;
                })(),
                rsi_limit_exit_enabled: (() => {
                    const el = document.getElementById('rsiLimitExitEnabled');
                    return el ? el.checked : (config.autoBot.rsi_limit_exit_enabled === true);
                })(),
                rsi_limit_exit_offset_percent: (() => {
                    const el = document.getElementById('rsiLimitExitOffsetPercentGlobal');
                    if (el && el.value !== '') {
                        const v = parseFloat(el.value);
                        return !isNaN(v) && v >= 0 ? v : 0.2;
                    }
                    return parseFloat(config.autoBot.rsi_limit_exit_offset_percent) || 0.2;
                })(),
                default_position_size: config.autoBot.default_position_size,
                default_position_mode: config.autoBot.default_position_mode,
                leverage: config.autoBot.leverage,
                check_interval: config.autoBot.check_interval,
                trading_enabled: config.autoBot.trading_enabled,
                use_test_server: config.autoBot.use_test_server
            };
            await this.sendConfigUpdate('auto-bot', params, 'Торговые параметры и RSI выходы');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения:', error);
            this.showNotification('❌ Ошибка сохранения торговых параметров и RSI выходов', 'error');
        }
    },
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
            await this.saveConfiguration(false, true);
            if (window.aiConfigManager && typeof window.aiConfigManager.saveAIConfig === 'function') {
                await window.aiConfigManager.saveAIConfig(false, true);
            }
            this.aiConfigDirty = false;
            this.updateFloatingSaveButtonVisibility();
            this.showConfigNotification('✅ Сохранено', 'Все настройки сохранены', 'success');
        } catch (error) {
            console.error('[BotsManager] Ошибка при сохранении:', error);
            this.showConfigNotification('❌ Ошибка', 'Ошибка сохранения настроек: ' + error.message, 'error');
        }
    },
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
        const hasChanges = this.hasUnsavedConfigChanges();
        if (isBotsPageVisible && isConfigTabActive) {
            btn.classList.add('visible');
            btn.disabled = !hasChanges;
        } else {
            btn.classList.remove('visible');
            btn.disabled = false;
        }
    },
            filterChangedParams(data, configType = 'autoBot') {
        const originalGroup = configType === 'system'
            ? (this.originalConfig?.system)
            : (this.originalConfig?.autoBot);

        if (!originalGroup) {
            // Если нет исходной конфигурации, отправляем все данные
            console.log('[BotsManager] ⚠️ originalConfig не инициализирован, отправляем все параметры');
            return data;
        }
        
        const original = originalGroup;
        const filtered = {};
        let changedCount = 0;
        
        console.log(`[BotsManager] 🔍 filterChangedParams: сравниваем ${Object.keys(data).length} параметров`);
        // ✅ КРИТИЧЕСКИ ВАЖНО: Логируем scope для отладки
        if (data.scope !== undefined) {
            console.log(`[BotsManager] 🔍 SCOPE в data: "${data.scope}" (тип: ${typeof data.scope})`);
            console.log(`[BotsManager] 🔍 SCOPE в original: "${original.scope}" (тип: ${typeof original.scope})`);
            console.log(`[BotsManager] 🔍 SCOPE сравнение: ${data.scope} !== ${original.scope} = ${data.scope !== original.scope}`);
        }
        
        for (const [key, value] of Object.entries(data)) {
            const originalValue = original[key];
            
            // ✅ ОСОБАЯ ОБРАБОТКА ДЛЯ break_even_trigger_percent
            if (key === 'break_even_trigger_percent' && originalValue === undefined) {
                // Если в originalConfig нет break_even_trigger_percent, проверяем break_even_trigger
                const altOriginalValue = original['break_even_trigger'];
                if (altOriginalValue !== undefined) {
                    if (typeof value === 'number' && typeof altOriginalValue === 'number') {
                        if (Math.abs(value - altOriginalValue) > 0.01) {
                            filtered[key] = value;
                            changedCount++;
                            console.log(`[BotsManager] 🔄 Изменен ${key}: ${altOriginalValue} → ${value} (из break_even_trigger)`);
                        }
                    }
                } else {
                    // Если и break_even_trigger нет, считаем что значение изменилось
                    filtered[key] = value;
                    changedCount++;
                    console.log(`[BotsManager] 🔄 Изменен ${key}: undefined → ${value} (новый параметр)`);
                }
                continue;
            }
            
            // Для чисел: сравниваем с точностью 0.01
            if (typeof value === 'number' && typeof originalValue === 'number') {
                if (Math.abs(value - originalValue) > 0.01) {
                    filtered[key] = value;
                    changedCount++;
                    console.log(`[BotsManager] 🔄 Изменен ${key}: ${originalValue} → ${value}`);
                } else {
                    console.log(`[BotsManager] ⏭️ Пропущен ${key}: ${originalValue} == ${value} (не изменился)`);
                }
            }
            // Для булевых значений: точное сравнение
            else if (typeof value === 'boolean' && typeof originalValue === 'boolean') {
                if (value !== originalValue) {
                    filtered[key] = value;
                    changedCount++;
                    console.log(`[BotsManager] 🔄 Изменен ${key}: ${originalValue} → ${value}`);
                } else {
                    console.log(`[BotsManager] ⏭️ Пропущен ${key}: ${originalValue} == ${value} (не изменился)`);
                }
            }
            // ✅ ОСОБАЯ ОБРАБОТКА ДЛЯ scope - ВСЕГДА проверяем первым!
            else if (key === 'scope') {
                console.log(`[BotsManager] 🔍 [SCOPE] Сравнение scope: текущее="${value}" (тип: ${typeof value}), оригинальное="${originalValue}" (тип: ${typeof originalValue})`);
                console.log(`[BotsManager] 🔍 [SCOPE] Строгое сравнение: ${value} !== ${originalValue} = ${value !== originalValue}`);
                // ✅ КРИТИЧЕСКИ ВАЖНО: Для scope всегда проверяем изменение, даже если originalValue undefined
                if (originalValue === undefined || value !== originalValue) {
                    filtered[key] = value;
                    changedCount++;
                    console.log(`[BotsManager] ✅ [SCOPE] Изменен scope: ${originalValue || 'undefined'} → ${value} (ДОБАВЛЕН В ИЗМЕНЕННЫЕ!)`);
                } else {
                    console.log(`[BotsManager] ⏭️ [SCOPE] Пропущен scope: ${originalValue} == ${value} (не изменился)`);
                }
            }
            // Для остальных типов: точное сравнение
            else if (value !== originalValue) {
                filtered[key] = value;
                changedCount++;
                console.log(`[BotsManager] 🔄 Изменен ${key}: ${originalValue} → ${value}`);
            } else {
                console.log(`[BotsManager] ⏭️ Пропущен ${key}: ${originalValue} == ${value} (не изменился)`);
            }
        }
        
        console.log(`[BotsManager] 📊 Отфильтровано: ${changedCount} из ${Object.keys(data).length} параметров изменены`);
        // ✅ КРИТИЧЕСКИ ВАЖНО: Логируем scope в результате
        if (data.scope !== undefined) {
            if (filtered.scope !== undefined) {
                console.log(`[BotsManager] ✅ [SCOPE] scope ПОПАЛ В ОТПРАВЛЯЕМЫЕ ПАРАМЕТРЫ: "${filtered.scope}"`);
            } else {
                console.log(`[BotsManager] ❌ [SCOPE] scope НЕ ПОПАЛ В ОТПРАВЛЯЕМЫЕ ПАРАМЕТРЫ! data.scope="${data.scope}", original.scope="${original.scope}"`);
            }
        }
        if (changedCount > 0) {
            console.log(`[BotsManager] 📤 Отправляемые параметры:`, filtered);
        } else {
            console.log(`[BotsManager] ⚠️ НЕТ ИЗМЕНЕННЫХ ПАРАМЕТРОВ! Все ${Object.keys(data).length} параметров без изменений`);
        }
        return filtered;
    },
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
            });
            
            if (response.ok) {
                const responseData = await response.json();
                console.log(`[BotsManager] ✅ ${sectionName} сохранены успешно, ответ сервера:`, responseData);
                
                // ✅ Проверяем количество изменений из ответа сервера
                const changesCount = responseData.changes_count || 0;
                if (changesCount === 0) {
                    // Нет изменений - показываем соответствующее сообщение
                    this.showNotification(`ℹ️ Нет изменений в настройках`, 'info');
                } else {
                    // Есть изменения - показываем детальное сообщение из сервера
                    const message = responseData.message || `✅ ${sectionName} сохранены успешно`;
                    this.showNotification(message, 'success');
                    
                    // ✅ Логируем только измененные параметры
                    if (responseData.changed_params && responseData.changed_params.length > 0) {
                        console.log(`[BotsManager] 📋 Измененные параметры (${changesCount}):`, responseData.changed_params);
                    }
                }
                console.log(`[BotsManager] 🔔 Уведомление отправлено для ${sectionName}`);
                
                // ✅ ОБНОВЛЯЕМ originalConfig после успешного сохранения
                if (this.originalConfig) {
                    // Обновляем только сохраненные параметры
                    for (const [key, value] of Object.entries(filteredData)) {
                        if (configType === 'system') {
                            this.originalConfig.system[key] = value;
                        } else {
                            this.originalConfig.autoBot[key] = value;
                        }
                    }
                    console.log(`[BotsManager] 💾 originalConfig обновлен после сохранения ${sectionName}`);
                    console.log(`[BotsManager] 🔍 Обновленные параметры в originalConfig:`, Object.keys(filteredData));
                    // ✅ КРИТИЧЕСКИ ВАЖНО: Логируем scope для отладки
                    if (filteredData.scope !== undefined) {
                        console.log(`[BotsManager] ✅ scope обновлен в originalConfig: ${this.originalConfig.autoBot.scope}`);
                    }
                }
                
                // ✅ ПЕРЕЗАГРУЖАЕМ КОНФИГУРАЦИЮ ДЛЯ ОБНОВЛЕНИЯ UI (особенно важно для Enhanced RSI)
                setTimeout(() => {
                    console.log(`[BotsManager] 🔄 Перезагрузка конфигурации после сохранения ${sectionName}...`);
                    this.loadConfigurationData();
                    
                    // Если сохраняли Enhanced RSI - перезагружаем данные монет для применения новых фильтров
                    if (sectionName === 'Enhanced RSI' || (configType === 'system' && filteredData.enhanced_rsi_enabled !== undefined)) {
                        console.log('[BotsManager] 🔄 Перезагрузка RSI данных для применения Enhanced RSI настроек...');
                        setTimeout(() => {
                            this.loadCoinsRsiData();
                        }, 500);
                    }
                }, 300);
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка сохранения ${sectionName}:`, error);
            this.showNotification(`❌ Ошибка: ${error.message}`, 'error');
            throw error;
        }
    },
            async saveConfiguration(isAutoSave = false, skipNotification = false) {
        // Отменяем запланированное автосохранение при ручном сохранении
        if (!isAutoSave && this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
            this.autoSaveTimer = null;
            console.log('[BotsManager] ⏸️ Автосохранение отменено - выполняется ручное сохранение');
        }
        
        console.log('[BotsManager] 💾 Сохранение конфигурации...');
        
        try {
            const config = this.collectConfigurationData();
            
            // Отладочные логи для Enhanced RSI
            console.log('[BotsManager] 🔍 Отправляемая конфигурация Enhanced RSI:');
            console.log('  enhanced_rsi_enabled:', config.autoBot.enhanced_rsi_enabled);
            console.log('  enhanced_rsi_require_volume_confirmation:', config.autoBot.enhanced_rsi_require_volume_confirmation);
            console.log('  enhanced_rsi_require_divergence_confirmation:', config.autoBot.enhanced_rsi_require_divergence_confirmation);
            console.log('  enhanced_rsi_use_stoch_rsi:', config.autoBot.enhanced_rsi_use_stoch_rsi);
            
            // БЕЗ БЛОКИРОВКИ - элементы остаются активными!
            
            // ✅ Проверяем, что есть данные для отправки Auto Bot
            if (!config.autoBot || Object.keys(config.autoBot).length === 0) {
                console.log('[BotsManager] ⚠️ Auto Bot конфигурация пуста, пропускаем сохранение');
            } else {
                // ПРИИ (full_ai_control) не трогаем при «Сохранить все» — только при явном переключении тумблера (иначе баг UI может выключить ПРИИ)
                const autoBotPayload = { ...config.autoBot };
                delete autoBotPayload.full_ai_control;
                // Сохраняем Auto Bot настройки
                const autoBotResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(autoBotPayload)
                });
                const autoBotData = await autoBotResponse.json();
                if (!autoBotData.success) {
                    throw new Error(`Ошибка сохранения Auto Bot: ${autoBotData.message || 'Unknown error'}`);
                }
            }
            
            // ✅ Проверяем, что есть данные для отправки System
            if (!config.system || Object.keys(config.system).length === 0) {
                console.log('[BotsManager] ⚠️ System конфигурация пуста, пропускаем сохранение');
            } else {
                // Сохраняем системные настройки
                const systemResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config.system)
                });
                const systemData = await systemResponse.json();
                if (!systemData.success) {
                    throw new Error(`Ошибка сохранения System: ${systemData.message || 'Unknown error'}`);
                }
            }
            
            // Показываем уведомление только при ручном сохранении (при вызове из saveAllConfiguration — skipNotification)
            if (!isAutoSave && !skipNotification) {
                this.showNotification('✅ Настройки сохранены', 'success');
            }
            console.log('[BotsManager] ✅ Конфигурация сохранена в bot_config.py и перезагружена');
            
            // ✅ ОБНОВЛЯЕМ RSI ПОРОГИ (для фильтров и подписей)
            if (config.autoBot) {
                this.updateRsiThresholds(config.autoBot);
                console.log('[BotsManager] 🔄 RSI пороги обновлены после сохранения');
            }
            
            this.aiConfigDirty = false;
            this.updateFloatingSaveButtonVisibility();
            setTimeout(() => this.loadConfigurationData(), 500);
            
            // ✅ ПЕРЕЗАГРУЖАЕМ ДАННЫЕ RSI (чтобы применить новые фильтры)
            setTimeout(() => {
                console.log('[BotsManager] 🔄 Перезагрузка RSI данных для применения новых настроек...');
                this.loadCoinsRsiData();
            }, 1000);
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения конфигурации:', error);
            // Показываем уведомление об ошибке только если это не автосохранение
            if (!isAutoSave && !skipNotification) {
                this.showNotification('❌ Ошибка сохранения конфигурации: ' + error.message, 'error');
            }
            // Пробрасываем ошибку дальше для обработки в scheduleAutoSave
            throw error;
        }
    },
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
        let errors = [];
        
        if (config.autoBot.rsi_long_threshold >= config.autoBot.rsi_short_threshold) {
            errors.push('RSI для LONG должен быть меньше RSI для SHORT');
        }
        
        // ✅ Валидация новых параметров RSI выхода
        if (config.autoBot.rsi_exit_long_with_trend && config.autoBot.rsi_exit_long_with_trend <= config.autoBot.rsi_long_threshold) {
            errors.push('RSI выход из LONG (по тренду) должен быть больше порога входа');
        }
        
        if (config.autoBot.rsi_exit_long_against_trend && config.autoBot.rsi_exit_long_against_trend <= config.autoBot.rsi_long_threshold) {
            errors.push('RSI выход из LONG (против тренда) должен быть больше порога входа');
        }
        
        if (config.autoBot.rsi_exit_short_with_trend && config.autoBot.rsi_exit_short_with_trend >= config.autoBot.rsi_short_threshold) {
            errors.push('RSI выход из SHORT (по тренду) должен быть меньше порога входа');
        }
        
        if (config.autoBot.rsi_exit_short_against_trend && config.autoBot.rsi_exit_short_against_trend >= config.autoBot.rsi_short_threshold) {
            errors.push('RSI выход из SHORT (против тренда) должен быть меньше порога входа');
        }
        
        if (config.autoBot.max_loss_percent <= 0 || config.autoBot.max_loss_percent > 50) {
            errors.push('Стоп-лосс должен быть от 1% до 50%');
        }
        
        if (config.autoBot.close_at_profit_enabled !== false && (config.autoBot.take_profit_percent <= 0 || config.autoBot.take_profit_percent > 100)) {
            errors.push('При включённом «Закрывать по % прибыли» укажите Take Profit от 1% до 100%');
        }
        
        if (config.autoBot.trailing_stop_activation < config.autoBot.break_even_trigger) {
            errors.push('Активация Trailing Stop должна быть больше триггера безубыточности');
        }
        
        if (errors.length > 0) {
            this.showNotification('❌ Ошибки конфигурации:\n' + errors.join('\n'), 'error');
        } else {
            this.showNotification('✅ Конфигурация корректна!', 'success');
        }
    },
            syncDuplicateSettings(config) {
        console.log('[BotsManager] 🔄 Синхронизация дублированных настроек...');
        
        // КРИТИЧЕСКИ ВАЖНО: Синхронизируем переключатель Auto Bot на главной странице
        const globalAutoBotToggleEl = document.getElementById('globalAutoBotToggle');
        if (globalAutoBotToggleEl) {
            const enabled = config.enabled || false;
            globalAutoBotToggleEl.checked = enabled;
            console.log(`[BotsManager] 🤖 Auto Bot переключатель синхронизирован: ${enabled}`);
            
            // Обновляем визуальное состояние
            const toggleLabel = globalAutoBotToggleEl.closest('.auto-bot-toggle')?.querySelector('.toggle-label');
            if (toggleLabel) {
                toggleLabel.textContent = enabled ? '🤖 Auto Bot (ВКЛ)' : '🤖 Auto Bot (ВЫКЛ)';
            }
        }
        // Синхронизируем переключатель «Полный Режим ИИ» на вкладке Управление
        const fullAiControlToggleEl = document.getElementById('fullAiControlToggle');
        if (fullAiControlToggleEl) {
            const fullAiOn = config.full_ai_control === true;
            fullAiControlToggleEl.checked = fullAiOn;
            const aiEnabled = config.ai_enabled === true;
            const aiLicenseValid = config.ai_license_valid === true;
            // Тумблер FullAI всегда активен — при включении бэкенд при необходимости включит ИИ; при невалидной лицензии FullAI сбросится при загрузке
            fullAiControlToggleEl.disabled = false;
            if (!aiEnabled) {
                fullAiControlToggleEl.title = (window.languageUtils?.translate?.('full_ai_control_disabled_hint') || 'При включении FullAI ИИ будет включён автоматически, если лицензия валидна');
            } else if (!aiLicenseValid) {
                fullAiControlToggleEl.title = (window.languageUtils?.translate?.('full_ai_control_license_warning') || 'При невалидной лицензии FullAI будет сброшен при загрузке');
            } else {
                fullAiControlToggleEl.title = (window.languageUtils?.translate?.('full_ai_control_tooltip') || 'ИИ сам решает когда входить и выходить');
            }
            const fullAiLabel = fullAiControlToggleEl.closest('.full-ai-control-toggle')?.querySelector('.toggle-label');
            if (fullAiLabel) {
                fullAiLabel.textContent = fullAiOn ? '🧠 Полный Режим ИИ (ВКЛ)' : '🧠 Полный Режим ИИ';
            }
            const fullAiModeBadge = document.getElementById('fullAiModeBadge');
            if (fullAiModeBadge) {
                fullAiModeBadge.textContent = fullAiOn
                    ? (window.languageUtils?.translate?.('fullai_mode_full_ai') || 'Режим: FullAI')
                    : (window.languageUtils?.translate?.('fullai_mode_standard') || 'Режим: Стандартный');
                fullAiModeBadge.className = 'full-ai-mode-badge ' + (fullAiOn ? 'mode-full-ai' : 'mode-standard');
            }
            // Дубль переключателя и бейджа на вкладке Конфигурация
            const fullAiControlToggleConfigEl = document.getElementById('fullAiControlToggleConfig');
            if (fullAiControlToggleConfigEl) {
                fullAiControlToggleConfigEl.checked = fullAiOn;
            }
            const fullAiModeBadgeConfig = document.getElementById('fullAiModeBadgeConfig');
            if (fullAiModeBadgeConfig) {
                fullAiModeBadgeConfig.textContent = fullAiOn
                    ? (window.languageUtils?.translate?.('fullai_mode_full_ai') || 'Режим: FullAI')
                    : (window.languageUtils?.translate?.('fullai_mode_standard') || 'Режим: Стандартный');
                fullAiModeBadgeConfig.className = 'full-ai-mode-badge ' + (fullAiOn ? 'mode-full-ai' : 'mode-standard');
            }
            // Параметры обкатки (ниже переключателя) — подтягиваем при загрузке конфига
            if (fullAiOn) this.loadFullaiAdaptiveConfig();
        }
        
        // Синхронизируем мобильный переключатель Auto Bot
        const mobileAutoBotToggleEl = document.getElementById('mobileAutobotToggle');
        if (mobileAutoBotToggleEl) {
            const enabled = config.enabled || false;
            mobileAutoBotToggleEl.checked = enabled;
            console.log(`[BotsManager] 🤖 Мобильный Auto Bot переключатель синхронизирован: ${enabled}`);
            
            // Обновляем визуальное состояние
            const statusText = document.getElementById('mobileAutobotStatusText');
            if (statusText) {
                statusText.textContent = enabled ? 'ВКЛ' : 'ВЫКЛ';
                statusText.className = enabled ? 'mobile-autobot-status enabled' : 'mobile-autobot-status';
            }
        }
        
        // Синхронизируем дублированные элементы на вкладке "Управление"
        const rsiLongDupEl = document.getElementById('rsiLongThresholdDup');
        if (rsiLongDupEl) rsiLongDupEl.value = config.rsi_long_threshold || 29;
        
        const rsiShortDupEl = document.getElementById('rsiShortThresholdDup');
        if (rsiShortDupEl) rsiShortDupEl.value = config.rsi_short_threshold || 71;
        
        const rsiExitLongDupEl = document.getElementById('rsiExitLongDup');
        if (rsiExitLongDupEl) rsiExitLongDupEl.value = config.rsi_exit_long || 65;
        
        const rsiExitShortDupEl = document.getElementById('rsiExitShortDup');
        if (rsiExitShortDupEl) rsiExitShortDupEl.value = config.rsi_exit_short || 35;
        
        const maxLossDupEl = document.getElementById('maxLossPercentDup');
        if (maxLossDupEl) maxLossDupEl.value = config.max_loss_percent || 15.0;
        
        const takeProfitDupEl = document.getElementById('takeProfitPercentDup');
        if (takeProfitDupEl) takeProfitDupEl.value = config.take_profit_percent || 20.0;
        
        const trailingActivationDupEl = document.getElementById('trailingStopActivationDup');
        if (trailingActivationDupEl) {
            const value = Number.parseFloat(config.trailing_stop_activation);
            trailingActivationDupEl.value = Number.isFinite(value) ? value : 20.0;
        }
        
        const trailingDistanceDupEl = document.getElementById('trailingStopDistanceDup');
        if (trailingDistanceDupEl) {
            const value = Number.parseFloat(config.trailing_stop_distance);
            trailingDistanceDupEl.value = Number.isFinite(value) ? value : 5.0;
        }

        const trailingTakeDupEl = document.getElementById('trailingTakeDistanceDup');
        if (trailingTakeDupEl) {
            const value = config.trailing_take_distance;
            trailingTakeDupEl.value = (value !== undefined && value !== null) ? value : 0.5;
        }

        const trailingIntervalDupEl = document.getElementById('trailingUpdateIntervalDup');
        if (trailingIntervalDupEl) {
            const value = config.trailing_update_interval;
            trailingIntervalDupEl.value = (value !== undefined && value !== null) ? value : 3.0;
        }
        
        const maxHoursDupEl = document.getElementById('maxPositionHoursDup');
        if (maxHoursDupEl) {
            const hours = config.max_position_hours || 0;
            maxHoursDupEl.value = Math.round(hours * 3600);
        }
        
        const breakEvenDupEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenDupEl) breakEvenDupEl.checked = config.break_even_protection !== false;

        const lossReentryProtectionDupEl = document.getElementById('lossReentryProtection');
        if (lossReentryProtectionDupEl) lossReentryProtectionDupEl.checked = config.loss_reentry_protection !== false;

        const lossReentryCountDupEl = document.getElementById('lossReentryCount');
        if (lossReentryCountDupEl) lossReentryCountDupEl.value = config.loss_reentry_count || 1;

        const lossReentryCandlesDupEl = document.getElementById('lossReentryCandles');
        if (lossReentryCandlesDupEl) lossReentryCandlesDupEl.value = config.loss_reentry_candles || 3;
        
        const avoidDownTrendDupEl = document.getElementById('avoidDownTrendDup');
        if (avoidDownTrendDupEl) avoidDownTrendDupEl.checked = config.avoid_down_trend !== false;
        
        const avoidUpTrendDupEl = document.getElementById('avoidUpTrendDup');
        if (avoidUpTrendDupEl) avoidUpTrendDupEl.checked = config.avoid_up_trend !== false;
        
        const enableMaturityCheckDupEl = document.getElementById('enableMaturityCheckDup');
        if (enableMaturityCheckDupEl) enableMaturityCheckDupEl.checked = config.enable_maturity_check !== false;
        
        const breakEvenTriggerDupEl = document.getElementById('breakEvenTriggerDup');
        if (breakEvenTriggerDupEl) {
            // Используем значение из конфига, если оно есть, иначе не меняем поле (оставляем текущее значение)
            const triggerValue = config.break_even_trigger_percent ?? config.break_even_trigger;
            if (triggerValue !== undefined && triggerValue !== null) {
                breakEvenTriggerDupEl.value = triggerValue;
            }
        }
        
        console.log('[BotsManager] ✅ Дублированные настройки синхронизированы');
        
        // Обновляем подписи тренд-фильтров после синхронизации
        this.updateTrendFilterLabels();
    },
            async loadDuplicateSettings() {
        console.log('[BotsManager] 📋 Загрузка дублированных настроек...');
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`);
            const data = await response.json();
            
            if (data.success && data.config) {
                this.syncDuplicateSettings(data.config);
                this.initializeGlobalAutoBotToggle();
            this.initializeMobileAutoBotToggle();
                
                // Обновляем RSI пороги из конфигурации
                this.updateRsiThresholds(data.config);
                
                console.log('[BotsManager] ✅ Дублированные настройки загружены');
            } else {
                console.error('[BotsManager] ❌ Ошибка загрузки дублированных настроек:', data.message);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка запроса дублированных настроек:', error);
        }
    },
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
                const data = await response.json();
                
                if (data.success && data.config) {
                    const autoBotEnabled = data.config.enabled;
                    console.log('[BotsManager] 🤖 Текущее состояние Auto Bot с сервера:', autoBotEnabled ? 'ВКЛ' : 'ВЫКЛ');
                    
                    // Устанавливаем состояние переключателя
                    globalAutoBotToggleEl.checked = autoBotEnabled;
                    
                    // Обновляем визуальное состояние
                    const toggleLabel = globalAutoBotToggleEl.closest('.auto-bot-toggle')?.querySelector('.toggle-label');
                    if (toggleLabel) {
                        toggleLabel.textContent = autoBotEnabled ? '🤖 Auto Bot (ВКЛ)' : '🤖 Auto Bot (ВЫКЛ)';
                    }
                    
                    console.log('[BotsManager] ✅ Переключатель Auto Bot инициализирован с состоянием:', autoBotEnabled);
                } else {
                    console.error('[BotsManager] ❌ Ошибка загрузки состояния Auto Bot:', data.message);
                }
            } catch (error) {
                console.error('[BotsManager] ❌ Ошибка запроса состояния Auto Bot:', error);
            }
            
            globalAutoBotToggleEl.addEventListener('change', async (e) => {
                const isEnabled = e.target.checked;
                console.log(`[BotsManager] 🤖 ИЗМЕНЕНИЕ ПЕРЕКЛЮЧАТЕЛЯ: ${isEnabled}`);
                
                // Помечаем, что пользователь изменил переключатель
                globalAutoBotToggleEl.setAttribute('data-user-changed', 'true');
                console.log('[BotsManager] 🔒 Флаг data-user-changed установлен');
                
                // Обновляем визуальное состояние сразу
                const toggleLabel = globalAutoBotToggleEl.closest('.auto-bot-toggle')?.querySelector('.toggle-label');
                if (toggleLabel) {
                    toggleLabel.textContent = isEnabled ? '🤖 Auto Bot (ВКЛ)' : '🤖 Auto Bot (ВЫКЛ)';
                }
                
                try {
                    const url = `${this.BOTS_SERVICE_URL}/api/bots/auto-bot`;
                    console.log(`[BotsManager] 📡 Отправка запроса на ${isEnabled ? 'включение' : 'выключение'} автобота...`);
                    console.log(`[BotsManager] 🌐 URL: ${url}`);
                    console.log(`[BotsManager] 📦 Данные: ${JSON.stringify({ enabled: isEnabled })}`);
                    // Сохраняем изменение через API
                    const response = await fetch(url, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ enabled: isEnabled })
                    });
                    console.log('[BotsManager] 📡 Ответ получен:', response.status);
                    
                    const result = await response.json();
                    console.log('[BotsManager] 📦 Результат от сервера:', result);
                    console.log('[BotsManager] 📊 Состояние enabled в ответе:', result.config?.enabled);
                    
                    if (result.success) {
                        this.showNotification(
                            isEnabled ? '✅ Auto Bot включен' : '⏸️ Auto Bot выключен', 
                            'success'
                        );
                        
                        // Синхронизируем с мобильным переключателем
                        const mobileToggle = document.getElementById('mobileAutobotToggle');
                        if (mobileToggle) {
                            mobileToggle.checked = isEnabled;
                            const mobileStatusText = document.getElementById('mobileAutobotStatusText');
                            if (mobileStatusText) {
                                mobileStatusText.textContent = isEnabled ? 'ВКЛ' : 'ВЫКЛ';
                                mobileStatusText.className = isEnabled ? 'mobile-autobot-status enabled' : 'mobile-autobot-status';
                            }
                            console.log(`[BotsManager] 🔄 Мобильный переключатель синхронизирован: ${isEnabled}`);
                        }
                        
                        // ✅ ИСПРАВЛЕНИЕ: Сбрасываем флаг с задержкой
                        // Даем время автообновлению получить новое состояние с сервера
                        setTimeout(() => {
                            globalAutoBotToggleEl.removeAttribute('data-user-changed');
                            console.log('[BotsManager] 🔓 Флаг data-user-changed снят после задержки');
                        }, 15000);  // 15 секунд - достаточно для автообновления
                        
                        console.log(`[BotsManager] ✅ Auto Bot ${isEnabled ? 'включен' : 'выключен'} и сохранен`);
                } else {
                        console.error('[BotsManager] ❌ Ошибка сохранения Auto Bot:', result.message);
                        // НЕ возвращаем переключатель в исходное состояние при ошибке API
                        // Пользователь может попробовать снова
                        this.showNotification('❌ Ошибка сохранения: ' + result.message, 'error');
                    }
                    
                } catch (error) {
                    console.error('[BotsManager] ❌ Ошибка изменения Auto Bot:', error);
                    // НЕ возвращаем переключатель в исходное состояние при ошибке соединения
                    // Пользователь может попробовать снова
                    this.showNotification('❌ Ошибка соединения с сервисом. Попробуйте еще раз.', 'error');
                }
            });
            
            console.log('[BotsManager] ✅ Обработчик главного переключателя Auto Bot инициализирован');
        }
    },
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
                const statusText = document.getElementById('mobileAutobotStatusText');
                if (statusText) {
                    statusText.textContent = isEnabled ? 'ВКЛ' : 'ВЫКЛ';
                    statusText.className = isEnabled ? 'mobile-autobot-status enabled' : 'mobile-autobot-status';
                }
                
                try {
                    const url = `${this.BOTS_SERVICE_URL}/api/bots/auto-bot`;
                    console.log(`[BotsManager] 📡 Отправка запроса на ${isEnabled ? 'включение' : 'выключение'} автобота...`);
                    console.log(`[BotsManager] 🌐 URL: ${url}`);
                    console.log(`[BotsManager] 📦 Данные: ${JSON.stringify({ enabled: isEnabled })}`);
                    
                    // Сохраняем изменение через API
                    const response = await fetch(url, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ enabled: isEnabled })
                    });
                    
                    const result = await response.json();
                    console.log('[BotsManager] 📨 Ответ сервера:', result);
                    
                    if (result.success) {
                        console.log(`[BotsManager] ✅ Auto Bot ${isEnabled ? 'включен' : 'выключен'} успешно`);
                        this.showNotification(`✅ Auto Bot ${isEnabled ? 'включен' : 'выключен'}`, 'success');
                        
                        // Синхронизируем с основным переключателем
                        const globalToggle = document.getElementById('globalAutoBotToggle');
                        if (globalToggle) {
                            globalToggle.checked = isEnabled;
                            const globalLabel = globalToggle.closest('.auto-bot-toggle')?.querySelector('.toggle-label');
                            if (globalLabel) {
                                globalLabel.textContent = isEnabled ? '🤖 Auto Bot (ВКЛ)' : '🤖 Auto Bot (ВЫКЛ)';
                            }
                            console.log(`[BotsManager] 🔄 Основной переключатель синхронизирован: ${isEnabled}`);
                        }
                        
                        // Убираем флаг изменения после успешного сохранения с задержкой
                        setTimeout(() => {
                            mobileAutoBotToggleEl.removeAttribute('data-user-changed');
                            console.log('[BotsManager] 🔓 Флаг data-user-changed снят для мобильного после задержки');
                        }, 15000);  // 15 секунд - достаточно для автообновления
                        
                    } else {
                        console.error('[BotsManager] ❌ Ошибка сервера:', result.message);
                        this.showNotification('❌ Ошибка сохранения: ' + result.message, 'error');
                    }
                    
                } catch (error) {
                    console.error('[BotsManager] ❌ Ошибка изменения Auto Bot:', error);
                    this.showNotification('❌ Ошибка соединения с сервисом. Попробуйте еще раз.', 'error');
                }
            });
            
            console.log('[BotsManager] ✅ Обработчик мобильного переключателя Auto Bot инициализирован');
        }
    },
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
        const key = [balance, available, pnl, positions, insufficient_funds].join('|');
        if (this._lastAccountDisplay === key) {
            return;
        }
        this._lastAccountDisplay = key;
        
        const activeBotsHeader = document.querySelector('.active-bots-header h3');
        if (!activeBotsHeader) return;
        
        if (accountData && accountData.success) {
            const balanceText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['balance']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['balance'] : 'Баланс';
            const remainderText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['remainder']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['remainder'] : 'Остаток';
            const openPositionsText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['open_positions']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['open_positions'] : 'Открытых позиций';
            
            activeBotsHeader.innerHTML = `
                ${balanceText}  $${balance.toFixed(2)}<br>
                ${remainderText}  $${available.toFixed(2)}<br>
                PnL  ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}<br>
                ${openPositionsText}  ${positions}
            `;
        } else {
            const balanceText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['balance']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['balance'] : 'Баланс';
            const remainderText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['remainder']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['remainder'] : 'Остаток';
            const openPositionsText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['open_positions']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['open_positions'] : 'Открытых позиций';
            
            activeBotsHeader.innerHTML = `
                ${balanceText}  -<br>
                ${remainderText}  -<br>
                PnL  -<br>
                ${openPositionsText}  -
            `;
        }
        
        const showInsufficient = insufficient_funds;
        const trInsufficient = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['insufficient_funds']);
        document.querySelectorAll('.insufficient-funds-alert').forEach(function (el) {
            el.style.display = showInsufficient ? 'block' : 'none';
            if (showInsufficient && trInsufficient) el.textContent = trInsufficient;
        });
    },
            updateBulkControlsVisibility(bots) {
        const bulkControlsEl = document.getElementById('bulkBotControls');
        const countEl = document.getElementById('bulkControlsCount');
        
        if (bulkControlsEl && countEl) {
            if (bots && bots.length > 0) {
                bulkControlsEl.style.display = 'block';
                countEl.textContent = `${bots.length} ${bots.length === 1 ? 'бот' : 'ботов'}`;
                this.initializeBulkControls(bots);
            } else {
                bulkControlsEl.style.display = 'none';
            }
        }
    },
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
        const configTab = document.getElementById('configTab');
        if (!configTab) {
            console.warn('[BotsManager] ⚠️ Вкладка конфигурации не найдена, автосохранение не инициализировано');
            return;
        }
        
        // Находим все поля конфигурации: input, select, checkbox
        // Включая поля в секции AI (aiConfigSection), которая может быть скрыта
        const configInputs = configTab.querySelectorAll('input[type="number"], input[type="text"], input[type="checkbox"], select');
        
        // Также добавляем поля из секции AI, если она существует
        const aiConfigSection = document.getElementById('aiConfigSection');
        let allInputs = Array.from(configInputs);
        
        if (aiConfigSection) {
            const aiInputs = aiConfigSection.querySelectorAll('input[type="number"], input[type="text"], input[type="checkbox"], select');
            console.log(`[BotsManager] 🔍 Найдено полей в секции AI: ${aiInputs.length}`);
            // Добавляем поля из AI секции
            allInputs = Array.from(new Set([...allInputs, ...Array.from(aiInputs)]));
        }
        
        console.log(`[BotsManager] 🔍 Всего полей конфигурации: ${allInputs.length}`);
        
        // Добавляем обработчики для всех полей
        this.addAutoSaveHandlers(allInputs);
        
        // ✅ Явно добавляем обработчик для toggle лимитных ордеров (может не попасть в querySelectorAll)
        const limitOrdersToggle = document.getElementById('limitOrdersEntryEnabled');
        if (limitOrdersToggle && !limitOrdersToggle.hasAttribute('data-autosave-initialized')) {
            limitOrdersToggle.setAttribute('data-autosave-initialized', 'true');
            limitOrdersToggle.addEventListener('change', () => {
                if (!this.isProgrammaticChange) this.scheduleToggleAutoSave(limitOrdersToggle);
            });
            console.log('[BotsManager] ✅ Обработчик автосохранения добавлен для toggle лимитных ордеров');
        }
    }
    
    /**
     * Сохраняет ТОЛЬКО одно значение переключателя (checkbox/select) — предотвращает сброс других настроек
     */,
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

        let value;
        if (input.type === 'checkbox') {
            value = input.checked;
        } else if (input.tagName === 'SELECT' || input.type === 'hidden') {
            value = input.value;
        } else {
            return false;
        }

        try {
            if (isSystem) {
                const systemKey = configKey.startsWith('system_') ? configKey.replace('system_', '') : configKey;
                const payload = { [systemKey]: value };
                const res = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!data.success) throw new Error(data.message || 'System config save failed');
            } else {
                const payload = { [configKey]: value };
                const res = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!data.success) throw new Error(data.message || 'Auto-bot config save failed');
            }
            if (this.originalConfig) {
                const group = isSystem ? this.originalConfig.system : this.originalConfig.autoBot;
                const key = isSystem ? configKey.replace('system_', '') : configKey;
                if (group) group[key] = value;
            }
            if (this.cachedAutoBotConfig && !isSystem) {
                this.cachedAutoBotConfig[configKey] = value;
            }
            return true;
        } catch (e) {
            console.error('[BotsManager] saveSingleToggleToBackend:', e);
            throw e;
        }
    }

    /**
     * Автосохранение при изменении переключателя (checkbox/select) — сохраняет ТОЛЬКО изменённое поле
     */,
            scheduleToggleAutoSave(input) {
        if (this.toggleAutoSaveTimer) clearTimeout(this.toggleAutoSaveTimer);
        const self = this;
        this.toggleAutoSaveTimer = setTimeout(async () => {
            self.toggleAutoSaveTimer = null;
            try {
                if (input && input.closest('#aiConfigSection')) {
                    if (window.aiConfigManager && typeof window.aiConfigManager.saveAIConfig === 'function') {
                        await window.aiConfigManager.saveAIConfig(false, false);
                    }
                    self.aiConfigDirty = false;
                } else {
                    const ok = await self.saveSingleToggleToBackend(input);
                    if (!ok) {
                        await self.saveConfiguration(false, true);
                    }
                }
                self.updateFloatingSaveButtonVisibility();
                self.showConfigNotification('✅ Сохранено', 'Настройка сохранена', 'success');
            } catch (err) {
                console.error('[BotsManager] Ошибка автосохранения переключателя:', err);
                self.showConfigNotification('❌ Ошибка', 'Ошибка сохранения: ' + err.message, 'error');
            }
        }, this.toggleAutoSaveDelay);
    }

    /**
     * Добавляет обработчики автосохранения для списка полей
     */,
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
            
            // Числа и текст: сохраняем только при blur (уход с поля) или Enter — не при каждом нажатии клавиши
            if (input.type === 'number' || input.type === 'text') {
                input.addEventListener('blur', () => {
                    if (!this.isProgrammaticChange) {
                        if (input.closest('#aiConfigSection')) this.aiConfigDirty = true;
                        this.updateFloatingSaveButtonVisibility();
                    }
                });
                input.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') e.target.blur();
                });
            }
            if (input.type === 'checkbox' || input.tagName === 'SELECT') {
                input.addEventListener('change', () => {
                    if (!this.isProgrammaticChange) {
                        this.scheduleToggleAutoSave(input);
                    }
                });
            }
        });
        
        console.log(`[BotsManager] ✅ Обработчики автосохранения добавлены для ${inputs.length} полей`);
    }
    
    /**
     * Добавляет кнопки +/- к числовым полям конфигурации (изменение по step, с учётом min/max).
     */,
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
        
        // Очищаем предыдущий таймер
        if (this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
            this.autoSaveTimer = null;
        }
        
        // Устанавливаем новый таймер на 2 секунды
        this.autoSaveTimer = setTimeout(async () => {
            console.log('[BotsManager] ⏱️ Автосохранение конфигурации...');
            
            try {
                // Сохраняем конфигурацию с флагом автосохранения
                await self.saveConfiguration(true);
                console.log('[BotsManager] ✅ Конфигурация автосохранена');
                
                // ✅ ПРИНУДИТЕЛЬНО показываем toast-уведомление (прямой вызов toastManager)
                console.log('[BotsManager] 🔔 Показ toast-уведомления об автосохранении...');
                
                // ✅ Прямой вызов toastManager - гарантированно работает
                if (window.toastManager) {
                    // Инициализируем, если нужно
                    if (!window.toastManager.container) {
                        window.toastManager.init();
                    }
                    // Проверяем, что контейнер в DOM
                    if (window.toastManager.container && !document.body.contains(window.toastManager.container)) {
                        document.body.appendChild(window.toastManager.container);
                    }
                    // Показываем уведомление
                    window.toastManager.success('✅ Настройки автоматически сохранены', 3000);
                    console.log('[BotsManager] ✅ Toast-уведомление показано');
                } else {
                    console.warn('[BotsManager] ⚠️ toastManager не найден, пытаемся вызвать showNotification...');
                    // Fallback на showNotification
                    try {
                        self.showNotification('✅ Настройки автоматически сохранены', 'success');
                    } catch (e) {
                        console.error('[BotsManager] ❌ Ошибка показа уведомления:', e);
                    }
                }
            } catch (error) {
                console.error('[BotsManager] ❌ Ошибка автосохранения конфигурации:', error);
                // Показываем ошибку при автосохранении
                if (window.toastManager) {
                    window.toastManager.error('❌ Ошибка автосохранения: ' + error.message, 5000);
                } else {
                    try {
                        self.showNotification('❌ Ошибка автосохранения: ' + error.message, 'error');
                    } catch (e) {
                        console.error('[BotsManager] ❌ Ошибка показа уведомления об ошибке:', e);
                    }
                }
            } finally {
                self.autoSaveTimer = null;
            }
        }, this.autoSaveDelay);
    },
            async reloadModules() {
        console.log('[BotsManager] 🔄 Перезагрузка модулей...');
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/system/reload-modules`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(`✅ ${data.message}. Модули перезагружены без перезапуска сервера!`, 'success');
                console.log('[BotsManager] ✅ Перезагружено модулей:', data.reloaded);
                if (data.failed && data.failed.length > 0) {
                    console.error('[BotsManager] ❌ Ошибки при перезагрузке:', data.failed);
                }
                
                // Обновляем данные после перезагрузки
                await this.loadConfiguration();
                await this.loadCoinsRsiData();
            } else {
                this.showNotification(`❌ Ошибка перезагрузки: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка Hot Reload:', error);
            this.showNotification('❌ Ошибка перезагрузки модулей', 'error');
        }
    },
            async startAllBots() {
        if (!this.activeBots || this.activeBots.length === 0) {
            this.showNotification('⚠️ Нет ботов для запуска', 'warning');
            return;
        }

        const stoppedBots = this.activeBots.filter(bot => 
            bot.status === 'paused' || bot.status === 'idle' || bot.status === 'stopped'
        );
        
        if (stoppedBots.length === 0) {
            this.showNotification('ℹ️ Все боты уже запущены', 'info');
            return;
        }
        
        console.log(`[BotsManager] 🚀 Запуск ${stoppedBots.length} ботов...`);
        this.showConfigNotification('🚀 Массовый запуск ботов', `Запускаем ${stoppedBots.length} ботов...`);
        
        let successful = 0;
        let failed = 0;
        
        for (const bot of stoppedBots) {
            try {
                const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/start`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: bot.symbol })
                });
                
                const result = await response.json();
                if (result.success) {
                    successful++;
                } else {
                    failed++;
                }
            } catch (error) {
                failed++;
            }
        }
        
        await this.loadActiveBotsData();
        
        if (failed === 0) {
            this.showConfigNotification('✅ Все боты запущены', `Успешно запущено ${successful} ботов`);
        } else {
            this.showConfigNotification('⚠️ Запуск завершен с ошибками', 
                `Успешно: ${successful}, Ошибок: ${failed}`, 'error');
        }
    },
            async stopAllBots() {
        if (!this.activeBots || this.activeBots.length === 0) {
            this.showNotification('⚠️ Нет ботов для остановки', 'warning');
            return;
        }
        
        const runningBots = this.activeBots.filter(bot => 
            bot.status === 'running' || bot.status === 'idle' || 
            bot.status === 'in_position_long' || bot.status === 'in_position_short'
        );
        
        if (runningBots.length === 0) {
            this.showNotification('ℹ️ Все боты уже остановлены', 'info');
            return;
        }
        
        console.log(`[BotsManager] ⏹️ Остановка ${runningBots.length} ботов...`);
        this.showConfigNotification('⏹️ Массовая остановка ботов', `Останавливаем ${runningBots.length} ботов...`);
        
        // Немедленно обновляем UI для всех ботов
        runningBots.forEach(bot => {
            this.updateBotStatusInUI(bot.symbol, 'stopping');
        });
        
        let successful = 0;
        let failed = 0;
        
        for (const bot of runningBots) {
            try {
                const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/stop`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: bot.symbol })
                });
                
                const result = await response.json();
                if (result.success) {
                    successful++;
                } else {
                    failed++;
                }
            } catch (error) {
                console.error(`[BotsManager] ❌ Ошибка остановки бота ${bot.symbol}:`, error);
                failed++;
            }
        }
        
        await this.loadActiveBotsData();
        
        if (failed === 0) {
            this.showConfigNotification('✅ Все боты остановлены', `Успешно остановлено ${successful} ботов`);
                } else {
            this.showConfigNotification('⚠️ Остановка завершена с ошибками', 
                `Успешно: ${successful}, Ошибок: ${failed}`, 'error');
        }
    },
            async deleteAllBots() {
        if (!this.activeBots || this.activeBots.length === 0) {
            this.showNotification('⚠️ Нет ботов для удаления', 'warning');
            return;
        }
        
        const confirmMessage = `🗑️ Удалить всех ${this.activeBots.length} ботов?\n\nЭто действие нельзя отменить!`;
        
        if (!confirm(confirmMessage)) {
            return;
        }
        
        console.log(`[BotsManager] 🗑️ Удаление ${this.activeBots.length} ботов...`);
        this.showConfigNotification('🗑️ Массовое удаление ботов', `Удаляем ${this.activeBots.length} ботов...`);
        
        // Немедленно обновляем UI для всех ботов
        this.activeBots.forEach(bot => {
            this.updateBotStatusInUI(bot.symbol, 'deleting');
        });
        
        let successful = 0;
        let failed = 0;
        
        for (const bot of this.activeBots) {
            try {
                const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/delete`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: bot.symbol })
                });
                
                const result = await response.json();
                if (result.success) {
                    successful++;
                } else {
                    failed++;
                }
            } catch (error) {
                failed++;
            }
        }
        
        await this.loadActiveBotsData();
        this.updateCoinsListWithBotStatus();
        
        if (failed === 0) {
            this.showConfigNotification('✅ Все боты удалены', `Успешно удалено ${successful} ботов`);
        } else {
            this.showConfigNotification('⚠️ Удаление завершено с ошибками', 
                `Успешно: ${successful}, Ошибок: ${failed}`, 'error');
        }
    },
            showConfigNotification(title, message, type = 'success', changes = null) {
        // Удаляем предыдущее уведомление если есть
        const existingNotification = document.querySelector('.config-save-notification');
        if (existingNotification) {
            existingNotification.remove();
        }
        
        // Создаем новое уведомление
        const notification = document.createElement('div');
        notification.className = `config-save-notification ${type === 'error' ? 'error' : ''}`;
        
        let changesHtml = '';
        if (changes && changes.length > 0) {
            changesHtml = `
                <div class="config-changes-list">
                    <strong>${this.translate('changes_label')}</strong>
                    <ul>
                        ${changes.map(change => `<li>${change}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        notification.innerHTML = `
            <div class="config-notification-header">
                <span class="config-notification-title">${title}</span>
                <button class="config-notification-close" type="button">&times;</button>
            </div>
            <div class="config-notification-body">
                ${message}
                ${changesHtml}
            </div>
        `;
        
        // Добавляем в DOM
        document.body.appendChild(notification);
        
        // Показываем с анимацией
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        // Обработчик закрытия
        const closeBtn = notification.querySelector('.config-notification-close');
        const closeNotification = () => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 400);
        };
        
        closeBtn.addEventListener('click', closeNotification);
        
        // Автоматическое закрытие через 5 секунд
        setTimeout(closeNotification, 5000);
        
        console.log(`[BotsManager] 📢 Уведомление: ${title} - ${message}`);
    },
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
            
            if (typeof value === 'boolean') {
                return isEnglish ? 
                    (value ? 'enabled' : 'disabled') : 
                    (value ? 'включено' : 'выключено');
            }
            if (key === 'scope') {
                if (isEnglish) {
                    return value === 'all' ? 'All coins' : 
                           value === 'whitelist' ? 'Whitelist' : 
                           value === 'blacklist' ? 'Blacklist' : value;
                } else {
                    return value === 'all' ? 'Все монеты' : 
                           value === 'whitelist' ? 'Белый список' : 
                           value === 'blacklist' ? 'Черный список' : value;
                }
            }
            if (key === 'rsi_update_interval') {
                const minutes = Math.round(value / 60);
                return isEnglish ? 
                    `${minutes} min (${value} sec)` : 
                    `${minutes} мин (${value} сек)`;
            }
            if (key === 'auto_save_interval') {
                return isEnglish ? `${value} sec` : `${value} сек`;
            }
            return value;
        };
        
        // Сравниваем Auto Bot настройки
        if (oldAutoBot && newAutoBot) {
            Object.keys(newAutoBot).forEach(key => {
                const oldValue = oldAutoBot[key];
                const newValue = newAutoBot[key];
                
                if (oldValue !== newValue && configLabels[key]) {
                    changes.push(
                        `${configLabels[key]}: ${formatValue(key, oldValue)} → ${formatValue(key, newValue)}`
                    );
                }
            });
        }
        
        // Сравниваем системные настройки
        if (oldSystem && newSystem) {
            Object.keys(newSystem).forEach(key => {
                const oldValue = oldSystem[key];
                const newValue = newSystem[key];
                
                if (oldValue !== newValue && configLabels[key]) {
                    changes.push(
                        `${configLabels[key]}: ${formatValue(key, oldValue)} → ${formatValue(key, newValue)}`
                    );
                }
            });
        }
        
        console.log('[BotsManager] 🔍 Обнаружено изменений:', changes.length);
        changes.forEach(change => console.log('[BotsManager] 📝', change));
        
        return changes;
    }
    
    /** Возвращает компактные данные для карточки бота: объём, позиция, вход, тейк, стоп, текущая цена */
    });
})();
