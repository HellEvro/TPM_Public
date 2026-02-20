/**
 * BotsManager - 06_duplicates_config
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            collectDuplicateSettings() {
        console.log('[BotsManager] 📋 Сбор дублированных настроек...');
        
        const settings = {};
        
        // RSI настройки
        const rsiLongEl = document.getElementById('rsiLongThresholdDup');
        if (rsiLongEl && rsiLongEl.value) settings.rsi_long_threshold = parseInt(rsiLongEl.value);
        
        const rsiShortEl = document.getElementById('rsiShortThresholdDup');
        if (rsiShortEl && rsiShortEl.value) settings.rsi_short_threshold = parseInt(rsiShortEl.value);
        
        // ✅ Новые параметры RSI выхода с учетом тренда
        const rsiExitLongWithTrendEl = document.getElementById('rsiExitLongWithTrendDup');
        if (rsiExitLongWithTrendEl && rsiExitLongWithTrendEl.value) {
            settings.rsi_exit_long_with_trend = parseInt(rsiExitLongWithTrendEl.value);
        }
        
        const rsiExitLongAgainstTrendEl = document.getElementById('rsiExitLongAgainstTrendDup');
        if (rsiExitLongAgainstTrendEl && rsiExitLongAgainstTrendEl.value) {
            settings.rsi_exit_long_against_trend = parseInt(rsiExitLongAgainstTrendEl.value);
        }
        
        const rsiExitShortWithTrendEl = document.getElementById('rsiExitShortWithTrendDup');
        if (rsiExitShortWithTrendEl && rsiExitShortWithTrendEl.value) {
            settings.rsi_exit_short_with_trend = parseInt(rsiExitShortWithTrendEl.value);
        }
        
        const rsiExitShortAgainstTrendEl = document.getElementById('rsiExitShortAgainstTrendDup');
        if (rsiExitShortAgainstTrendEl && rsiExitShortAgainstTrendEl.value) {
            settings.rsi_exit_short_against_trend = parseInt(rsiExitShortAgainstTrendEl.value);
        }
        
        // Защитные механизмы
        const maxLossEl = document.getElementById('maxLossPercentDup');
        if (maxLossEl && maxLossEl.value) settings.max_loss_percent = parseFloat(maxLossEl.value);
        
        const takeProfitEl = document.getElementById('takeProfitPercentDup');
        if (takeProfitEl && takeProfitEl.value !== '') settings.take_profit_percent = parseFloat(takeProfitEl.value);
        const closeAtProfitEl = document.getElementById('closeAtProfitEnabledDup');
        if (closeAtProfitEl) settings.close_at_profit_enabled = closeAtProfitEl.checked;
        
        const trailingActivationEl = document.getElementById('trailingStopActivationDup');
        if (trailingActivationEl && trailingActivationEl.value) settings.trailing_stop_activation = parseFloat(trailingActivationEl.value);
        
        const trailingDistanceEl = document.getElementById('trailingStopDistanceDup');
        if (trailingDistanceEl && trailingDistanceEl.value) settings.trailing_stop_distance = parseFloat(trailingDistanceEl.value);

        const trailingTakeEl = document.getElementById('trailingTakeDistanceDup');
        if (trailingTakeEl && trailingTakeEl.value) settings.trailing_take_distance = parseFloat(trailingTakeEl.value);

        const trailingIntervalEl = document.getElementById('trailingUpdateIntervalDup');
        if (trailingIntervalEl && trailingIntervalEl.value) settings.trailing_update_interval = parseFloat(trailingIntervalEl.value);
        
        const maxHoursEl = document.getElementById('maxPositionHoursDup');
        if (maxHoursEl) {
            const seconds = parseInt(maxHoursEl.value) || 0;
            // В конфиге хранятся часы; передаём часы (секунды / 3600)
            settings.max_position_hours = seconds / 3600;
        }
        
        const breakEvenEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenEl) settings.break_even_protection = breakEvenEl.checked;
        
        const breakEvenTriggerEl = document.getElementById('breakEvenTriggerDup');
        if (breakEvenTriggerEl && breakEvenTriggerEl.value) {
            const triggerValue = parseFloat(breakEvenTriggerEl.value);
            settings.break_even_trigger = triggerValue;
            settings.break_even_trigger_percent = triggerValue;
        }

        const avoidDownTrendEl = document.getElementById('avoidDownTrendDup');
        if (avoidDownTrendEl) settings.avoid_down_trend = avoidDownTrendEl.checked;

        const avoidUpTrendEl = document.getElementById('avoidUpTrendDup');
        if (avoidUpTrendEl) settings.avoid_up_trend = avoidUpTrendEl.checked;

        const lossReentryProtectionEl = document.getElementById('lossReentryProtection');
        if (lossReentryProtectionEl) settings.loss_reentry_protection = lossReentryProtectionEl.checked;

        const lossReentryCountEl = document.getElementById('lossReentryCount');
        if (lossReentryCountEl && lossReentryCountEl.value) {
            settings.loss_reentry_count = parseInt(lossReentryCountEl.value);
        }

        const lossReentryCandlesEl = document.getElementById('lossReentryCandles');
        if (lossReentryCandlesEl && lossReentryCandlesEl.value) {
            settings.loss_reentry_candles = parseInt(lossReentryCandlesEl.value);
        }

        const maturityCheckEl = document.getElementById('enableMaturityCheckDup');
        if (maturityCheckEl) settings.enable_maturity_check = maturityCheckEl.checked;

        const minCandlesMaturityEl = document.getElementById('minCandlesForMaturityDup');
        if (minCandlesMaturityEl && minCandlesMaturityEl.value) {
            settings.min_candles_for_maturity = parseInt(minCandlesMaturityEl.value);
        }

        const minRsiLowEl = document.getElementById('minRsiLowDup');
        if (minRsiLowEl && minRsiLowEl.value) {
            settings.min_rsi_low = parseFloat(minRsiLowEl.value);
        }

        const maxRsiHighEl = document.getElementById('maxRsiHighDup');
        if (maxRsiHighEl && maxRsiHighEl.value) {
            settings.max_rsi_high = parseFloat(maxRsiHighEl.value);
        }

        const rsiTimeFilterEnabledEl = document.getElementById('rsiTimeFilterEnabledDup');
        if (rsiTimeFilterEnabledEl) settings.rsi_time_filter_enabled = rsiTimeFilterEnabledEl.checked;

        const rsiTimeFilterCandlesEl = document.getElementById('rsiTimeFilterCandlesDup');
        if (rsiTimeFilterCandlesEl && rsiTimeFilterCandlesEl.value) {
            const candles = parseInt(rsiTimeFilterCandlesEl.value);
            settings.rsi_time_filter_candles = candles;
        }

        const rsiTimeFilterUpperEl = document.getElementById('rsiTimeFilterUpperDup');
        if (rsiTimeFilterUpperEl && rsiTimeFilterUpperEl.value) {
            settings.rsi_time_filter_upper = parseFloat(rsiTimeFilterUpperEl.value);
        }

        const rsiTimeFilterLowerEl = document.getElementById('rsiTimeFilterLowerDup');
        if (rsiTimeFilterLowerEl && rsiTimeFilterLowerEl.value) {
            settings.rsi_time_filter_lower = parseFloat(rsiTimeFilterLowerEl.value);
        }

        const exitScamEnabledEl = document.getElementById('exitScamEnabledDup');
        if (exitScamEnabledEl) settings.exit_scam_enabled = exitScamEnabledEl.checked;

        const exitScamCandlesEl = document.getElementById('exitScamCandlesDup');
        if (exitScamCandlesEl && exitScamCandlesEl.value) {
            settings.exit_scam_candles = parseInt(exitScamCandlesEl.value);
        }

        const exitScamSingleEl = document.getElementById('exitScamSingleCandleDup');
        if (exitScamSingleEl && exitScamSingleEl.value) {
            settings.exit_scam_single_candle_percent = parseFloat(exitScamSingleEl.value);
        }

        const exitScamMultiCountEl = document.getElementById('exitScamMultiCountDup');
        if (exitScamMultiCountEl && exitScamMultiCountEl.value) {
            settings.exit_scam_multi_candle_count = parseInt(exitScamMultiCountEl.value);
        }

        const exitScamMultiPercentEl = document.getElementById('exitScamMultiPercentDup');
        if (exitScamMultiPercentEl && exitScamMultiPercentEl.value) {
            settings.exit_scam_multi_candle_percent = parseFloat(exitScamMultiPercentEl.value);
        }

        const trendDetectionEnabledEl = document.getElementById('trendDetectionEnabledDup');
        if (trendDetectionEnabledEl) settings.trend_detection_enabled = trendDetectionEnabledEl.checked;

        const trendAnalysisPeriodEl = document.getElementById('trendAnalysisPeriodDup');
        if (trendAnalysisPeriodEl && trendAnalysisPeriodEl.value) {
            settings.trend_analysis_period = parseInt(trendAnalysisPeriodEl.value);
        }

        const trendPriceChangeEl = document.getElementById('trendPriceChangeThresholdDup');
        if (trendPriceChangeEl && trendPriceChangeEl.value) {
            settings.trend_price_change_threshold = parseFloat(trendPriceChangeEl.value);
        }

        const trendCandlesThresholdEl = document.getElementById('trendCandlesThresholdDup');
        if (trendCandlesThresholdEl && trendCandlesThresholdEl.value) {
            settings.trend_candles_threshold = parseInt(trendCandlesThresholdEl.value);
        }
        
        // ✅ Enhanced RSI настройки для индивидуальных настроек монеты
        const enhancedRsiEnabledDupEl = document.getElementById('enhancedRsiEnabledDup');
        if (enhancedRsiEnabledDupEl) {
            settings.enhanced_rsi_enabled = enhancedRsiEnabledDupEl.checked;
        }
        
        const enhancedRsiVolumeConfirmDupEl = document.getElementById('enhancedRsiVolumeConfirmDup');
        if (enhancedRsiVolumeConfirmDupEl) {
            settings.enhanced_rsi_require_volume_confirmation = enhancedRsiVolumeConfirmDupEl.checked;
        }
        
        const enhancedRsiDivergenceConfirmDupEl = document.getElementById('enhancedRsiDivergenceConfirmDup');
        if (enhancedRsiDivergenceConfirmDupEl) {
            settings.enhanced_rsi_require_divergence_confirmation = enhancedRsiDivergenceConfirmDupEl.checked;
        }
        
        const enhancedRsiUseStochRsiDupEl = document.getElementById('enhancedRsiUseStochRsiDup');
        if (enhancedRsiUseStochRsiDupEl) {
            settings.enhanced_rsi_use_stoch_rsi = enhancedRsiUseStochRsiDupEl.checked;
        }
        
        console.log('[BotsManager] 📋 Собранные настройки:', settings);
        return settings;
    },
            async loadIndividualSettings(symbol) {
        if (!symbol) return null;
        
        try {
            console.log(`[BotsManager] 📥 Загрузка индивидуальных настроек для ${symbol}`);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}`);
            
            // 404 - это нормально, значит настроек нет
            if (response.status === 404) {
                console.log(`[BotsManager] ℹ️ Индивидуальных настроек для ${symbol} не найдено (404)`);
                return null;
            }
            
            const data = await response.json();
            
            if (data.success) {
                console.log(`[BotsManager] ✅ Индивидуальные настройки для ${symbol} загружены:`, data.settings);
                return data.settings;
            } else {
                console.log(`[BotsManager] ℹ️ Индивидуальных настроек для ${symbol} не найдено`);
                return null;
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка загрузки индивидуальных настроек для ${symbol}:`, error);
            return null;
        }
    },
            async saveIndividualSettings(symbol, settings) {
        if (!symbol || !settings) return false;
        
        try {
            console.log(`[BotsManager] 💾 Сохранение индивидуальных настроек для ${symbol}:`, settings);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            
            const data = await response.json();
            if (data.success) {
                console.log(`[BotsManager] ✅ Индивидуальные настройки для ${symbol} сохранены`);
                this.showNotification(`✅ Настройки для ${symbol} сохранены`, 'success');
                return true;
            } else {
                console.error(`[BotsManager] ❌ Ошибка сохранения настроек: ${data.error}`);
                this.showNotification(`❌ Ошибка сохранения: ${data.error}`, 'error');
                return false;
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка сохранения индивидуальных настроек для ${symbol}:`, error);
            this.showNotification('❌ Ошибка соединения при сохранении', 'error');
            return false;
        }
    },
            async deleteIndividualSettings(symbol) {
        if (!symbol) return false;
        
        try {
            console.log(`[BotsManager] 🗑️ Удаление индивидуальных настроек для ${symbol}`);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            if (data.success) {
                console.log(`[BotsManager] ✅ Индивидуальные настройки для ${symbol} удалены`);
                this.showNotification(`✅ Настройки для ${symbol} сброшены к общим`, 'success');
                return true;
            } else {
                console.error(`[BotsManager] ❌ Ошибка удаления настроек: ${data.error}`);
                this.showNotification(`❌ Ошибка удаления: ${data.error}`, 'error');
                return false;
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка удаления индивидуальных настроек для ${symbol}:`, error);
            this.showNotification('❌ Ошибка соединения при удалении', 'error');
            return false;
        }
    },
            async copySettingsToAllCoins(symbol) {
        if (!symbol) return false;
        
        try {
            console.log(`[BotsManager] 📋 Копирование настроек ${symbol} ко всем монетам`);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}/copy-to-all`, {
                method: 'POST'
            });
            
            const data = await response.json();
            if (data.success) {
                console.log(`[BotsManager] ✅ Настройки ${symbol} скопированы к ${data.copied_count} монетам`);
                this.showNotification(`✅ Настройки применены к ${data.copied_count} монетам`, 'success');
                return true;
            } else {
                console.error(`[BotsManager] ❌ Ошибка копирования настроек: ${data.error}`);
                this.showNotification(`❌ Ошибка копирования: ${data.error}`, 'error');
                return false;
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка копирования настроек ${symbol}:`, error);
            this.showNotification('❌ Ошибка соединения при копировании', 'error');
            return false;
        }
    }

    /**
     * Подбор параметров ExitScam по истории свечей для выбранной монеты.
     * Результат сохраняется в индивидуальные настройки монеты и подставляется в форму.
     */,
            async learnExitScamForCoin() {
        if (!this.selectedCoin || !this.selectedCoin.symbol) {
            this.showNotification('⚠️ Выберите монету для подбора ExitScam', 'warning');
            return;
        }
        const symbol = this.selectedCoin.symbol;
        const btn = document.getElementById('learnExitScamForCoinBtn');
        const originalText = btn ? btn.innerHTML : '';
        try {
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span>⏳ Анализ свечей...</span>';
            }
            this.showNotification(`🧠 Анализ свечей ${symbol}...`, 'info');
            const exitScamTfEl = document.getElementById('exitScamTimeframe');
            const currentTf = exitScamTfEl?.value || this.cachedAutoBotConfig?.exit_scam_timeframe || '6h';
            const response = await fetch(
                `${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}/learn-exit-scam`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ aggressiveness: 'normal', timeframe: currentTf })
                }
            );
            const data = await response.json();
            if (data.success && data.params) {
                await this.loadAndApplyIndividualSettings(symbol);
                this.updateIndividualSettingsStatus(true);
                const p = data.params;
                this.showNotification(
                    `✅ ExitScam для ${symbol}: 1 св ${p.exit_scam_single_candle_percent}%, ${p.exit_scam_multi_candle_count} св ${p.exit_scam_multi_candle_percent}%`,
                    'success'
                );
            } else {
                const err = data.error || 'Не удалось подобрать параметры';
                this.showNotification(`❌ ExitScam: ${err}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка learn-exit-scam:', error);
            this.showNotification('❌ Ошибка соединения при подборе ExitScam', 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = originalText;
            }
        }
    }

    /**
     * Расчёт ExitScam по истории для всех монет (ручной запуск). Использует текущий ТФ из UI.
     */,
            async learnExitScamForAllCoins() {
        const btn = document.getElementById('learnExitScamForAllCoinsBtn');
        const originalText = btn ? btn.innerHTML : '';
        try {
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span>⏳ Расчёт для всех монет...</span>';
            }
            this.showNotification('🧠 Расчёт индивидуального ExitScam для всех монет...', 'info');
            const exitScamTfEl = document.getElementById('exitScamTimeframe');
            const currentTf = exitScamTfEl?.value || this.cachedAutoBotConfig?.exit_scam_timeframe || '6h';
            const response = await fetch(
                `${this.BOTS_SERVICE_URL}/api/bots/individual-settings/learn-exit-scam-all`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ aggressiveness: 'normal', timeframe: currentTf })
                }
            );
            const data = await response.json();
            if (data.success) {
                const u = data.updated_count || 0;
                const f = data.failed_count || 0;
                const sample = (data.sample_params || []).slice(0, 5);
                const sampleStr = sample.length
                    ? sample.map(s => `${s.symbol} ${s.exit_scam_single_candle_percent}%/${s.exit_scam_multi_candle_count}св ${s.exit_scam_multi_candle_percent}%`).join(', ')
                    : '';
                const msg = sampleStr
                    ? `✅ Обновлено ${u} монет (ошибок: ${f}). Примеры: ${sampleStr}. Выберите монету и загрузите настройки — значения индивидуальны.`
                    : `✅ Индивидуальный ExitScam для всех: обновлено ${u} монет, без данных/ошибок: ${f}`;
                this.showNotification(msg, 'success');
            } else {
                this.showNotification(`❌ ${data.error || 'Ошибка расчёта'}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка learn-exit-scam-all:', error);
            this.showNotification('❌ Ошибка соединения при расчёте для всех', 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = originalText;
            }
        }
    }

    /**
     * Сброс индивидуальных настроек ExitScam для всех монет — будут использоваться значения из конфига.
     */,
            async resetExitScamToConfigForAll() {
        const btn = document.getElementById('resetExitScamToConfigForAllBtn');
        const originalText = btn ? btn.innerHTML : '';
        try {
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span>⏳ Сброс...</span>';
            }
            this.showNotification('🔄 Сброс ExitScam к общим настройкам для всех монет...', 'info');
            const response = await fetch(
                `${this.BOTS_SERVICE_URL}/api/bots/individual-settings/reset-exit-scam-all`,
                { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }
            );
            const data = await response.json();
            if (data.success) {
                const n = data.reset_count || 0;
                this.showNotification(
                    n > 0 ? `✅ ExitScam сброшен к общим настройкам для ${n} монет` : '✅ Нет индивидуальных ExitScam — все уже используют конфиг',
                    'success'
                );
            } else {
                this.showNotification(`❌ ${data.error || 'Ошибка сброса'}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка reset-exit-scam-all:', error);
            this.showNotification('❌ Ошибка соединения при сбросе ExitScam', 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = originalText;
            }
        }
    },
            async resetAllCoinsToGlobalSettings() {
        try {
            const confirmed = confirm('⚠️ Вы уверены, что хотите сбросить индивидуальные настройки ВСЕХ монет к глобальным настройкам?\n\nЭто действие нельзя отменить!');
            if (!confirmed) {
                return false;
            }
            
            console.log('[BotsManager] 🔄 Сброс всех индивидуальных настроек к глобальным');
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/reset-all`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            if (data.success) {
                console.log(`[BotsManager] ✅ Сброшены индивидуальные настройки для ${data.removed_count} монет`);
                
                // Формируем красивое сообщение
                const coinWord = data.removed_count === 1 ? 'монеты' : 
                                data.removed_count >= 2 && data.removed_count <= 4 ? 'монет' : 'монет';
                const message = data.removed_count > 0 
                    ? `✅ Сброшены индивидуальные настройки для ${data.removed_count} ${coinWord}. Все монеты теперь используют глобальные настройки.`
                    : '✅ Индивидуальные настройки отсутствуют. Все монеты используют глобальные настройки.';
                
                this.showNotification(message, 'success');
                
                // Обновляем статус индивидуальных настроек, если выбрана монета
                if (this.selectedCoin) {
                    this.updateIndividualSettingsStatus(false);
                }
                
                return true;
            } else {
                console.error(`[BotsManager] ❌ Ошибка сброса настроек: ${data.error}`);
                this.showNotification(`❌ Ошибка сброса: ${data.error}`, 'error');
                return false;
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сброса всех индивидуальных настроек:', error);
            this.showNotification('❌ Ошибка соединения при сбросе настроек', 'error');
            return false;
        }
    }

    /**
     * Маппинг ключей конфига на ID элементов для подсветки отличий от основного конфига.
     * Только ключи, присутствующие в индивидуальных настройках и отличающиеся от main config, подсвечиваются.
     */,
            getIndividualSettingsElementMap() {
        return {
            rsi_long_threshold: 'rsiLongThresholdDup',
            rsi_short_threshold: 'rsiShortThresholdDup',
            rsi_exit_long_with_trend: 'rsiExitLongWithTrendDup',
            rsi_exit_long_against_trend: 'rsiExitLongAgainstTrendDup',
            rsi_exit_short_with_trend: 'rsiExitShortWithTrendDup',
            rsi_exit_short_against_trend: 'rsiExitShortAgainstTrendDup',
            max_loss_percent: 'maxLossPercentDup',
            take_profit_percent: 'takeProfitPercentDup',
            close_at_profit_enabled: 'closeAtProfitEnabledDup',
            trailing_stop_activation: 'trailingStopActivationDup',
            trailing_stop_distance: 'trailingStopDistanceDup',
            trailing_take_distance: 'trailingTakeDistanceDup',
            trailing_update_interval: 'trailingUpdateIntervalDup',
            max_position_hours: 'maxPositionHoursDup',
            break_even_protection: 'breakEvenProtectionDup',
            break_even_trigger: 'breakEvenTriggerDup',
            break_even_trigger_percent: 'breakEvenTriggerDup',
            avoid_down_trend: 'avoidDownTrendDup',
            avoid_up_trend: 'avoidUpTrendDup',
            enable_maturity_check: 'enableMaturityCheckDup',
            min_candles_for_maturity: 'minCandlesForMaturityDup',
            min_rsi_low: 'minRsiLowDup',
            max_rsi_high: 'maxRsiHighDup',
            rsi_time_filter_enabled: 'rsiTimeFilterEnabledDup',
            rsi_time_filter_candles: 'rsiTimeFilterCandlesDup',
            rsi_time_filter_upper: 'rsiTimeFilterUpperDup',
            rsi_time_filter_lower: 'rsiTimeFilterLowerDup',
            exit_scam_enabled: 'exitScamEnabledDup',
            exit_scam_candles: 'exitScamCandlesDup',
            exit_scam_single_candle_percent: 'exitScamSingleCandleDup',
            exit_scam_multi_candle_count: 'exitScamMultiCountDup',
            exit_scam_multi_candle_percent: 'exitScamMultiPercentDup',
            trend_detection_enabled: 'trendDetectionEnabledDup',
            trend_analysis_period: 'trendAnalysisPeriodDup',
            trend_price_change_threshold: 'trendPriceChangeThresholdDup',
            trend_candles_threshold: 'trendCandlesThresholdDup',
            volume_mode: 'volumeModeSelect',
            volume_value: 'volumeValueInput',
            leverage: 'leverageCoinInput',
            enhanced_rsi_enabled: 'enhancedRsiEnabledDup',
            enhanced_rsi_require_volume_confirmation: 'enhancedRsiVolumeConfirmDup',
            enhanced_rsi_require_divergence_confirmation: 'enhancedRsiDivergenceConfirmDup',
            enhanced_rsi_use_stoch_rsi: 'enhancedRsiUseStochRsiDup'
        };
    },
            clearIndividualSettingDiffHighlights() {
        document.querySelectorAll('.setting-item.individual-setting-diff').forEach(el => {
            el.classList.remove('individual-setting-diff');
        });
    }

    /**
     * Подсвечивает настройки, которые отличаются от основного конфига.
     * @param {Object} individualSettings - индивидуальные настройки монеты
     */,
            highlightIndividualSettingDiffs(individualSettings) {
        this.clearIndividualSettingDiffHighlights();
        if (!individualSettings || typeof individualSettings !== 'object') return;

        const config = this.cachedAutoBotConfig || {};
        const fallback = {
            rsi_long_threshold: 29, rsi_short_threshold: 71,
            rsi_exit_long_with_trend: 65, rsi_exit_long_against_trend: 60,
            rsi_exit_short_with_trend: 35, rsi_exit_short_against_trend: 40,
            max_loss_percent: 15.0, take_profit_percent: 5.0, close_at_profit_enabled: true,
            trailing_stop_activation: 20.0, trailing_stop_distance: 5.0,
            trailing_take_distance: 0.5, trailing_update_interval: 3.0,
            max_position_hours: 0, break_even_protection: true,
            break_even_trigger: 20.0, break_even_trigger_percent: 20.0,
            avoid_down_trend: true, avoid_up_trend: true,
            enable_maturity_check: true, min_candles_for_maturity: 400,
            min_rsi_low: 35, max_rsi_high: 65,
            rsi_time_filter_enabled: true, rsi_time_filter_candles: 6,
            rsi_time_filter_upper: 65, rsi_time_filter_lower: 35,
            exit_scam_enabled: true, exit_scam_candles: 8,
            exit_scam_single_candle_percent: 15, exit_scam_multi_candle_count: 4,
            exit_scam_multi_candle_percent: 50, trend_detection_enabled: false,
            trend_analysis_period: 30, trend_price_change_threshold: 7,
            trend_candles_threshold: 70, volume_mode: 'usdt', volume_value: 10,
            leverage: 10, enhanced_rsi_enabled: false,
            enhanced_rsi_require_volume_confirmation: false,
            enhanced_rsi_require_divergence_confirmation: false,
            enhanced_rsi_use_stoch_rsi: false
        };

        const getMainValue = (key) => {
            const v = config[key];
            return v !== undefined ? v : fallback[key];
        };

        const valuesEqual = (a, b) => {
            if (a === b) return true;
            if (typeof a === 'boolean' || typeof b === 'boolean') return Boolean(a) === Boolean(b);
            const na = Number(a);
            const nb = Number(b);
            if (!Number.isNaN(na) && !Number.isNaN(nb)) return na === nb;
            return String(a) === String(b);
        };

        const elementMap = this.getIndividualSettingsElementMap();

        for (const [configKey, elementId] of Object.entries(elementMap)) {
            if (!(configKey in individualSettings)) continue;
            if (configKey === 'break_even_trigger' && 'break_even_trigger_percent' in individualSettings) continue;

            const indVal = individualSettings[configKey];
            let mainVal = getMainValue(configKey);
            if (configKey === 'break_even_trigger_percent') {
                mainVal = getMainValue('break_even_trigger') ?? getMainValue('break_even_trigger_percent');
            }

            if (!valuesEqual(indVal, mainVal)) {
                const el = document.getElementById(elementId);
                if (el) {
                    const parent = el.closest('.setting-item');
                    if (parent) parent.classList.add('individual-setting-diff');
                }
            }
        }
    },
            applyIndividualSettingsToUI(settings) {
        if (!settings) return;
        
        console.log('[BotsManager] 🎨 Применение индивидуальных настроек к UI:', settings);
        const fallbackConfig = this.cachedAutoBotConfig || {};
        const getSettingValue = (key) => {
            if (settings[key] !== undefined) return settings[key];
            return fallbackConfig[key];
        };
        
        // RSI настройки
        const rsiLongEl = document.getElementById('rsiLongThresholdDup');
        if (rsiLongEl && settings.rsi_long_threshold !== undefined) {
            rsiLongEl.value = settings.rsi_long_threshold;
        }
        
        const rsiShortEl = document.getElementById('rsiShortThresholdDup');
        if (rsiShortEl && settings.rsi_short_threshold !== undefined) {
            rsiShortEl.value = settings.rsi_short_threshold;
        }
        
        // ✅ Новые параметры RSI выхода с учетом тренда
        const rsiExitLongWithTrendEl = document.getElementById('rsiExitLongWithTrendDup');
        if (rsiExitLongWithTrendEl && settings.rsi_exit_long_with_trend !== undefined) {
            rsiExitLongWithTrendEl.value = settings.rsi_exit_long_with_trend;
        }
        
        const rsiExitLongAgainstTrendEl = document.getElementById('rsiExitLongAgainstTrendDup');
        if (rsiExitLongAgainstTrendEl && settings.rsi_exit_long_against_trend !== undefined) {
            rsiExitLongAgainstTrendEl.value = settings.rsi_exit_long_against_trend;
        }
        
        const rsiExitShortWithTrendEl = document.getElementById('rsiExitShortWithTrendDup');
        if (rsiExitShortWithTrendEl && settings.rsi_exit_short_with_trend !== undefined) {
            rsiExitShortWithTrendEl.value = settings.rsi_exit_short_with_trend;
        }
        
        const rsiExitShortAgainstTrendEl = document.getElementById('rsiExitShortAgainstTrendDup');
        if (rsiExitShortAgainstTrendEl && settings.rsi_exit_short_against_trend !== undefined) {
            rsiExitShortAgainstTrendEl.value = settings.rsi_exit_short_against_trend;
        }
        
        // Защитные механизмы
        const maxLossEl = document.getElementById('maxLossPercentDup');
        if (maxLossEl && settings.max_loss_percent !== undefined) {
            maxLossEl.value = settings.max_loss_percent;
        }
        
        const trailingActivationEl = document.getElementById('trailingStopActivationDup');
        if (trailingActivationEl && settings.trailing_stop_activation !== undefined) {
            trailingActivationEl.value = settings.trailing_stop_activation;
        }
        
        const trailingDistanceEl = document.getElementById('trailingStopDistanceDup');
        if (trailingDistanceEl && settings.trailing_stop_distance !== undefined) {
            trailingDistanceEl.value = settings.trailing_stop_distance;
        }
        
        const maxHoursEl = document.getElementById('maxPositionHoursDup');
        if (maxHoursEl && settings.max_position_hours !== undefined) {
            // В конфиге часы; показываем в секундах
            maxHoursEl.value = Math.round((settings.max_position_hours || 0) * 3600);
        }
        
        const breakEvenEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenEl && settings.break_even_protection !== undefined) {
            breakEvenEl.checked = settings.break_even_protection;
        }
        
        const breakEvenTriggerEl = document.getElementById('breakEvenTriggerDup');
        const breakEvenTriggerValue = settings.break_even_trigger_percent ?? settings.break_even_trigger;
        if (breakEvenTriggerEl && breakEvenTriggerValue !== undefined) {
            breakEvenTriggerEl.value = breakEvenTriggerValue;
        }
        
        // Трендовые настройки
        const avoidDownTrendEl = document.getElementById('avoidDownTrendDup');
        if (avoidDownTrendEl) {
            const value = getSettingValue('avoid_down_trend');
            if (value !== undefined) {
                avoidDownTrendEl.checked = Boolean(value);
            }
        }
        
        const avoidUpTrendEl = document.getElementById('avoidUpTrendDup');
        if (avoidUpTrendEl) {
            const value = getSettingValue('avoid_up_trend');
            if (value !== undefined) {
                avoidUpTrendEl.checked = Boolean(value);
            }
        }
        
        const enableMaturityEl = document.getElementById('enableMaturityCheckDup');
        if (enableMaturityEl) {
            const value = getSettingValue('enable_maturity_check');
            if (value !== undefined) {
                enableMaturityEl.checked = Boolean(value);
            }
        }

        const minCandlesMaturityEl = document.getElementById('minCandlesForMaturityDup');
        if (minCandlesMaturityEl) {
            const value = getSettingValue('min_candles_for_maturity');
            if (value !== undefined) {
                minCandlesMaturityEl.value = value;
            }
        }

        const minRsiLowEl = document.getElementById('minRsiLowDup');
        if (minRsiLowEl) {
            const value = getSettingValue('min_rsi_low');
            if (value !== undefined) {
                minRsiLowEl.value = value;
            }
        }

        const maxRsiHighEl = document.getElementById('maxRsiHighDup');
        if (maxRsiHighEl) {
            const value = getSettingValue('max_rsi_high');
            if (value !== undefined) {
                maxRsiHighEl.value = value;
            }
        }

        const rsiTimeFilterEnabledEl = document.getElementById('rsiTimeFilterEnabledDup');
        if (rsiTimeFilterEnabledEl) {
            const value = getSettingValue('rsi_time_filter_enabled');
            if (value !== undefined) {
                rsiTimeFilterEnabledEl.checked = Boolean(value);
            }
        }

        const rsiTimeFilterCandlesEl = document.getElementById('rsiTimeFilterCandlesDup');
        if (rsiTimeFilterCandlesEl) {
            const value = getSettingValue('rsi_time_filter_candles');
            if (value !== undefined) {
                rsiTimeFilterCandlesEl.value = value;
            }
        }

        const rsiTimeFilterUpperEl = document.getElementById('rsiTimeFilterUpperDup');
        if (rsiTimeFilterUpperEl) {
            const value = getSettingValue('rsi_time_filter_upper');
            if (value !== undefined) {
                rsiTimeFilterUpperEl.value = value;
            }
        }

        const rsiTimeFilterLowerEl = document.getElementById('rsiTimeFilterLowerDup');
        if (rsiTimeFilterLowerEl) {
            const value = getSettingValue('rsi_time_filter_lower');
            if (value !== undefined) {
                rsiTimeFilterLowerEl.value = value;
            }
        }

        const exitScamEnabledEl = document.getElementById('exitScamEnabledDup');
        if (exitScamEnabledEl) {
            const value = getSettingValue('exit_scam_enabled');
            if (value !== undefined) {
                exitScamEnabledEl.checked = Boolean(value);
            }
        }

        const exitScamCandlesEl = document.getElementById('exitScamCandlesDup');
        if (exitScamCandlesEl) {
            const value = getSettingValue('exit_scam_candles');
            if (value !== undefined) {
                exitScamCandlesEl.value = value;
            }
        }

        const exitScamSingleEl = document.getElementById('exitScamSingleCandleDup');
        if (exitScamSingleEl) {
            const value = getSettingValue('exit_scam_single_candle_percent');
            if (value !== undefined) {
                exitScamSingleEl.value = value;
            }
        }

        const exitScamMultiCountEl = document.getElementById('exitScamMultiCountDup');
        if (exitScamMultiCountEl) {
            const value = getSettingValue('exit_scam_multi_candle_count');
            if (value !== undefined) {
                exitScamMultiCountEl.value = value;
            }
        }

        const exitScamMultiPercentEl = document.getElementById('exitScamMultiPercentDup');
        if (exitScamMultiPercentEl) {
            const value = getSettingValue('exit_scam_multi_candle_percent');
            if (value !== undefined) {
                exitScamMultiPercentEl.value = value;
            }
        }

        const trendDetectionEnabledEl = document.getElementById('trendDetectionEnabledDup');
        if (trendDetectionEnabledEl) {
            const value = getSettingValue('trend_detection_enabled');
            if (value !== undefined) {
                trendDetectionEnabledEl.checked = Boolean(value);
            }
        }

        const trendAnalysisPeriodEl = document.getElementById('trendAnalysisPeriodDup');
        if (trendAnalysisPeriodEl) {
            const value = getSettingValue('trend_analysis_period');
            if (value !== undefined) {
                trendAnalysisPeriodEl.value = value;
            }
        }

        const trendPriceChangeEl = document.getElementById('trendPriceChangeThresholdDup');
        if (trendPriceChangeEl) {
            const value = getSettingValue('trend_price_change_threshold');
            if (value !== undefined) {
                trendPriceChangeEl.value = value;
            }
        }

        const trendCandlesThresholdEl = document.getElementById('trendCandlesThresholdDup');
        if (trendCandlesThresholdEl) {
            const value = getSettingValue('trend_candles_threshold');
            if (value !== undefined) {
                trendCandlesThresholdEl.value = value;
            }
        }
        
        // Объем торговли
        const volumeModeEl = document.getElementById('volumeModeSelect');
        if (volumeModeEl && settings.volume_mode !== undefined) {
            volumeModeEl.value = settings.volume_mode;
        }
        
        const volumeValueEl = document.getElementById('volumeValueInput');
        if (volumeValueEl && settings.volume_value !== undefined) {
            volumeValueEl.value = settings.volume_value;
        }
        
        const leverageCoinEl = document.getElementById('leverageCoinInput');
        if (leverageCoinEl) {
            // Загружаем из индивидуальных настроек, если есть, иначе из глобального конфига
            const leverageValue = getSettingValue('leverage');
            if (leverageValue !== undefined) {
                leverageCoinEl.value = leverageValue;
            }
        }
        
        // ✅ Enhanced RSI настройки для индивидуальных настроек монеты
        const enhancedRsiEnabledDupEl = document.getElementById('enhancedRsiEnabledDup');
        if (enhancedRsiEnabledDupEl) {
            const value = getSettingValue('enhanced_rsi_enabled');
            if (value !== undefined) {
                enhancedRsiEnabledDupEl.checked = Boolean(value);
            }
        }
        
        const enhancedRsiVolumeConfirmDupEl = document.getElementById('enhancedRsiVolumeConfirmDup');
        if (enhancedRsiVolumeConfirmDupEl) {
            const value = getSettingValue('enhanced_rsi_require_volume_confirmation');
            if (value !== undefined) {
                enhancedRsiVolumeConfirmDupEl.checked = Boolean(value);
            }
        }
        
        const enhancedRsiDivergenceConfirmDupEl = document.getElementById('enhancedRsiDivergenceConfirmDup');
        if (enhancedRsiDivergenceConfirmDupEl) {
            const value = getSettingValue('enhanced_rsi_require_divergence_confirmation');
            if (value !== undefined) {
                enhancedRsiDivergenceConfirmDupEl.checked = Boolean(value);
            }
        }
        
        const enhancedRsiUseStochRsiDupEl = document.getElementById('enhancedRsiUseStochRsiDup');
        if (enhancedRsiUseStochRsiDupEl) {
            const value = getSettingValue('enhanced_rsi_use_stoch_rsi');
            if (value !== undefined) {
                enhancedRsiUseStochRsiDupEl.checked = Boolean(value);
            }
        }
        
        this.highlightIndividualSettingDiffs(settings);
        console.log('[BotsManager] ✅ Индивидуальные настройки применены к UI');
    },
            updateIndividualSettingsStatus(hasSettings) {
        const statusEl = document.getElementById('individualSettingsStatus');
        if (statusEl) {
            if (hasSettings) {
                statusEl.innerHTML = '<span style="color: #4CAF50;">✅ Есть индивидуальные настройки</span>';
            } else {
                statusEl.innerHTML = '<span style="color: #888;">Нет индивидуальных настроек для этой монеты</span>';
            }
        }
    },
            async loadAndApplyIndividualSettings(symbol) {
        if (!symbol) return;
        
        try {
            console.log(`[BotsManager] 📥 Загрузка и применение индивидуальных настроек для ${symbol}`);
            this.pendingIndividualSettingsSymbol = symbol;
             const settings = await this.loadIndividualSettings(symbol);
            if (this.pendingIndividualSettingsSymbol !== symbol) {
                console.log('[BotsManager] ⏭️ Ответ для старой монеты, игнорируем');
                return;
            }
             
             if (settings) {
                 // Применяем настройки к UI и подсвечиваем отличия от основного конфига
                 this.applyIndividualSettingsToUI(settings);
                 this.updateIndividualSettingsStatus(true);
                 console.log(`[BotsManager] ✅ Индивидуальные настройки для ${symbol} применены`);
             } else {
                 // Сбрасываем UI к общим настройкам и убираем подсветку
                 this.clearIndividualSettingDiffHighlights();
                 this.resetToGeneralSettings();
                 this.updateIndividualSettingsStatus(false);
                 console.log(`[BotsManager] ℹ️ Используются общие настройки для ${symbol}`);
             }
         } catch (error) {
             console.error(`[BotsManager] ❌ Ошибка загрузки индивидуальных настроек для ${symbol}:`, error);
             this.updateIndividualSettingsStatus(false);
         }
     }

     resetToGeneralSettings() {
        console.log('[BotsManager] 🔄 Сброс к общим настройкам');
        this.clearIndividualSettingDiffHighlights();
        const config = this.cachedAutoBotConfig || {};
        const fallback = {
            rsi_long_threshold: 29,
            rsi_short_threshold: 71,
            rsi_exit_long_with_trend: 65,
            rsi_exit_long_against_trend: 60,
            rsi_exit_short_with_trend: 35,
            rsi_exit_short_against_trend: 40,
            max_loss_percent: 15.0,
            take_profit_percent: 5.0,
            close_at_profit_enabled: true,
            trailing_stop_activation: 20.0,
            trailing_stop_distance: 5.0,
            trailing_take_distance: 0.5,
            trailing_update_interval: 3.0,
            max_position_hours: 0,
            break_even_protection: true,
            break_even_trigger: 20.0,
                    loss_reentry_protection: true,
                    loss_reentry_count: 1,
                    loss_reentry_candles: 3,
            avoid_down_trend: config.avoid_down_trend !== false,
            loss_reentry_protection: config.loss_reentry_protection !== false,
            loss_reentry_count: config.loss_reentry_count || 1,
            loss_reentry_candles: config.loss_reentry_candles || 3,
            avoid_up_trend: config.avoid_up_trend !== false,
            enable_maturity_check: config.enable_maturity_check !== false,
            min_candles_for_maturity: (config.min_candles_for_maturity !== undefined ? config.min_candles_for_maturity : 400),
            min_rsi_low: (config.min_rsi_low !== undefined ? config.min_rsi_low : 35),
            max_rsi_high: (config.max_rsi_high !== undefined ? config.max_rsi_high : 65),
            rsi_time_filter_enabled: (config.rsi_time_filter_enabled !== undefined ? config.rsi_time_filter_enabled : true),
            rsi_time_filter_candles: (config.rsi_time_filter_candles !== undefined ? config.rsi_time_filter_candles : 6),
            rsi_time_filter_upper: (config.rsi_time_filter_upper !== undefined ? config.rsi_time_filter_upper : 65),
            rsi_time_filter_lower: (config.rsi_time_filter_lower !== undefined ? config.rsi_time_filter_lower : 35),
            exit_scam_enabled: (config.exit_scam_enabled !== undefined ? config.exit_scam_enabled : true),
            exit_scam_candles: (config.exit_scam_candles !== undefined ? config.exit_scam_candles : 8),
            exit_scam_single_candle_percent: (config.exit_scam_single_candle_percent !== undefined ? config.exit_scam_single_candle_percent : 15),
            exit_scam_multi_candle_count: (config.exit_scam_multi_candle_count !== undefined ? config.exit_scam_multi_candle_count : 4),
            exit_scam_multi_candle_percent: (config.exit_scam_multi_candle_percent !== undefined ? config.exit_scam_multi_candle_percent : 50),
            trend_detection_enabled: (config.trend_detection_enabled !== undefined ? config.trend_detection_enabled : false),
            trend_analysis_period: (config.trend_analysis_period !== undefined ? config.trend_analysis_period : 30),
            trend_price_change_threshold: (config.trend_price_change_threshold !== undefined ? config.trend_price_change_threshold : 7),
            trend_candles_threshold: (config.trend_candles_threshold !== undefined ? config.trend_candles_threshold : 70)
        };

        const get = (key, defaultValue) => {
            const value = config[key];
            return value !== undefined ? value : defaultValue;
        };

        const setValue = (id, value) => {
            const el = document.getElementById(id);
            if (el !== null && value !== undefined) {
                el.value = value;
            }
        };

        setValue('rsiLongThresholdDup', get('rsi_long_threshold', fallback.rsi_long_threshold));
        setValue('rsiShortThresholdDup', get('rsi_short_threshold', fallback.rsi_short_threshold));
        setValue('rsiExitLongWithTrendDup', get('rsi_exit_long_with_trend', fallback.rsi_exit_long_with_trend));
        setValue('rsiExitLongAgainstTrendDup', get('rsi_exit_long_against_trend', fallback.rsi_exit_long_against_trend));
        setValue('rsiExitShortWithTrendDup', get('rsi_exit_short_with_trend', fallback.rsi_exit_short_with_trend));
        setValue('rsiExitShortAgainstTrendDup', get('rsi_exit_short_against_trend', fallback.rsi_exit_short_against_trend));
        setValue('maxLossPercentDup', get('max_loss_percent', fallback.max_loss_percent));
        setValue('takeProfitPercentDup', get('take_profit_percent', fallback.take_profit_percent));
        const closeAtProfitDupEl = document.getElementById('closeAtProfitEnabledDup');
        if (closeAtProfitDupEl) closeAtProfitDupEl.checked = get('close_at_profit_enabled', true) !== false;
        setValue('trailingStopActivationDup', get('trailing_stop_activation', fallback.trailing_stop_activation));
        setValue('trailingStopDistanceDup', get('trailing_stop_distance', fallback.trailing_stop_distance));
        setValue('trailingTakeDistanceDup', get('trailing_take_distance', fallback.trailing_take_distance));
        setValue('trailingUpdateIntervalDup', get('trailing_update_interval', fallback.trailing_update_interval));

        const maxHoursEl = document.getElementById('maxPositionHoursDup');
        if (maxHoursEl) {
            const hours = get('max_position_hours', fallback.max_position_hours);
            maxHoursEl.value = Math.round((hours || 0) * 3600);
        }

        const breakEvenEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenEl) {
            breakEvenEl.checked = get('break_even_protection', fallback.break_even_protection);
        }

        const breakEvenTriggerEl = document.getElementById('breakEvenTriggerDup');
        if (breakEvenTriggerEl) {
            breakEvenTriggerEl.value = get('break_even_trigger', fallback.break_even_trigger);
        }

        const avoidDownTrendEl = document.getElementById('avoidDownTrendDup');
        if (avoidDownTrendEl) {
            avoidDownTrendEl.checked = get('avoid_down_trend', fallback.avoid_down_trend);
        }

        const avoidUpTrendEl = document.getElementById('avoidUpTrendDup');
        if (avoidUpTrendEl) {
            avoidUpTrendEl.checked = get('avoid_up_trend', fallback.avoid_up_trend);
        }

        const maturityEl = document.getElementById('enableMaturityCheckDup');
        if (maturityEl) {
            maturityEl.checked = get('enable_maturity_check', fallback.enable_maturity_check);
        }

        const minCandlesMaturityEl = document.getElementById('minCandlesForMaturityDup');
        if (minCandlesMaturityEl) {
            minCandlesMaturityEl.value = get('min_candles_for_maturity', fallback.min_candles_for_maturity);
        }

        const minRsiLowEl = document.getElementById('minRsiLowDup');
        if (minRsiLowEl) {
            minRsiLowEl.value = get('min_rsi_low', fallback.min_rsi_low);
        }

        const maxRsiHighEl = document.getElementById('maxRsiHighDup');
        if (maxRsiHighEl) {
            maxRsiHighEl.value = get('max_rsi_high', fallback.max_rsi_high);
        }

        const rsiTimeFilterEnabledEl = document.getElementById('rsiTimeFilterEnabledDup');
        if (rsiTimeFilterEnabledEl) {
            rsiTimeFilterEnabledEl.checked = get('rsi_time_filter_enabled', fallback.rsi_time_filter_enabled);
        }

        const rsiTimeFilterCandlesEl = document.getElementById('rsiTimeFilterCandlesDup');
        if (rsiTimeFilterCandlesEl) {
            rsiTimeFilterCandlesEl.value = get('rsi_time_filter_candles', fallback.rsi_time_filter_candles);
        }

        const rsiTimeFilterUpperEl = document.getElementById('rsiTimeFilterUpperDup');
        if (rsiTimeFilterUpperEl) {
            rsiTimeFilterUpperEl.value = get('rsi_time_filter_upper', fallback.rsi_time_filter_upper);
        }

        const rsiTimeFilterLowerEl = document.getElementById('rsiTimeFilterLowerDup');
        if (rsiTimeFilterLowerEl) {
            rsiTimeFilterLowerEl.value = get('rsi_time_filter_lower', fallback.rsi_time_filter_lower);
        }

        const exitScamEnabledEl = document.getElementById('exitScamEnabledDup');
        if (exitScamEnabledEl) {
            exitScamEnabledEl.checked = get('exit_scam_enabled', fallback.exit_scam_enabled);
        }

        const exitScamCandlesEl = document.getElementById('exitScamCandlesDup');
        if (exitScamCandlesEl) {
            exitScamCandlesEl.value = get('exit_scam_candles', fallback.exit_scam_candles);
        }

        const exitScamSingleEl = document.getElementById('exitScamSingleCandleDup');
        if (exitScamSingleEl) {
            exitScamSingleEl.value = get('exit_scam_single_candle_percent', fallback.exit_scam_single_candle_percent);
        }

        const exitScamMultiCountEl = document.getElementById('exitScamMultiCountDup');
        if (exitScamMultiCountEl) {
            exitScamMultiCountEl.value = get('exit_scam_multi_candle_count', fallback.exit_scam_multi_candle_count);
        }

        const exitScamMultiPercentEl = document.getElementById('exitScamMultiPercentDup');
        if (exitScamMultiPercentEl) {
            exitScamMultiPercentEl.value = get('exit_scam_multi_candle_percent', fallback.exit_scam_multi_candle_percent);
        }

        const trendDetectionEnabledEl = document.getElementById('trendDetectionEnabledDup');
        if (trendDetectionEnabledEl) {
            trendDetectionEnabledEl.checked = get('trend_detection_enabled', fallback.trend_detection_enabled);
        }

        const trendAnalysisPeriodEl = document.getElementById('trendAnalysisPeriodDup');
        if (trendAnalysisPeriodEl) {
            trendAnalysisPeriodEl.value = get('trend_analysis_period', fallback.trend_analysis_period);
        }

        const trendPriceChangeEl = document.getElementById('trendPriceChangeThresholdDup');
        if (trendPriceChangeEl) {
            trendPriceChangeEl.value = get('trend_price_change_threshold', fallback.trend_price_change_threshold);
        }

        const trendCandlesThresholdEl = document.getElementById('trendCandlesThresholdDup');
        if (trendCandlesThresholdEl) {
            trendCandlesThresholdEl.value = get('trend_candles_threshold', fallback.trend_candles_threshold);
        }
        
        // Объем торговли и плечо
        const volumeModeEl = document.getElementById('volumeModeSelect');
        if (volumeModeEl) {
            volumeModeEl.value = get('default_position_mode', 'usdt');
        }
        
        const volumeValueEl = document.getElementById('volumeValueInput');
        if (volumeValueEl) {
            volumeValueEl.value = get('default_position_size', 10);
        }
        
        const leverageCoinEl = document.getElementById('leverageCoinInput');
        if (leverageCoinEl) {
            leverageCoinEl.value = get('leverage', 10);
        }
    },
            initializeIndividualSettingsButtons() {
        console.log('[BotsManager] 🔧 Инициализация кнопок индивидуальных настроек...');
        
        // Кнопка сохранения индивидуальных настроек
        const saveIndividualBtn = document.getElementById('saveIndividualSettingsBtn');
        if (saveIndividualBtn) {
            saveIndividualBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для сохранения настроек', 'warning');
                    return;
                }
                
                const settings = this.collectDuplicateSettings();
                // Добавляем основные настройки торговли (volume_mode, volume_value, leverage)
                const volumeModeEl = document.getElementById('volumeModeSelect');
                if (volumeModeEl) settings.volume_mode = volumeModeEl.value;
                const volumeValueEl = document.getElementById('volumeValueInput');
                if (volumeValueEl) settings.volume_value = parseFloat(volumeValueEl.value) || 10;
                const leverageCoinEl = document.getElementById('leverageCoinInput');
                if (leverageCoinEl) settings.leverage = parseInt(leverageCoinEl.value) || 10;
                const success = await this.saveIndividualSettings(this.selectedCoin.symbol, settings);
                if (success) {
                    this.highlightIndividualSettingDiffs(settings);
                    this.updateIndividualSettingsStatus(true);
                }
            });
        }
        
        // Кнопка загрузки индивидуальных настроек
        const loadIndividualBtn = document.getElementById('loadIndividualSettingsBtn');
        if (loadIndividualBtn) {
            loadIndividualBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для загрузки настроек', 'warning');
                    return;
                }
                
                await this.loadAndApplyIndividualSettings(this.selectedCoin.symbol);
            });
        }
        
        // Кнопка сброса к общим настройкам
        const resetIndividualBtn = document.getElementById('resetIndividualSettingsBtn');
        if (resetIndividualBtn) {
            resetIndividualBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для сброса настроек', 'warning');
                    return;
                }
                
                await this.deleteIndividualSettings(this.selectedCoin.symbol);
                this.resetToGeneralSettings();
                this.updateIndividualSettingsStatus(false);
            });
        }
        
        // Кнопка копирования настроек ко всем монетам
        const copyToAllBtn = document.getElementById('copyToAllCoinsBtn');
        if (copyToAllBtn) {
            copyToAllBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для копирования настроек', 'warning');
                    return;
                }
                
                const confirmed = confirm(`Вы уверены, что хотите применить настройки ${this.selectedCoin.symbol} ко всем монетам?`);
                if (confirmed) {
                    await this.copySettingsToAllCoins(this.selectedCoin.symbol);
                }
            });
        }
        
        // Кнопка «Подобрать ExitScam по истории» для выбранной монеты
        const learnExitScamBtn = document.getElementById('learnExitScamForCoinBtn');
        if (learnExitScamBtn) {
            learnExitScamBtn.addEventListener('click', () => this.learnExitScamForCoin());
        }
        // Кнопка «Индивидуальный ExitScam для всех монет» — расчёт по истории для каждой монеты
        const learnExitScamAllBtn = document.getElementById('learnExitScamForAllCoinsBtn');
        if (learnExitScamAllBtn) {
            learnExitScamAllBtn.addEventListener('click', () => this.learnExitScamForAllCoins());
        }
        const resetExitScamToConfigBtn = document.getElementById('resetExitScamToConfigForAllBtn');
        if (resetExitScamToConfigBtn) {
            resetExitScamToConfigBtn.addEventListener('click', () => this.resetExitScamToConfigForAll());
        }
        
        console.log('[BotsManager] ✅ Кнопки индивидуальных настроек инициализированы');
    },
            initializeQuickLaunchButtons() {
        console.log('[BotsManager] 🚀 Инициализация кнопок быстрого запуска...');
        
        // Кнопка быстрого запуска LONG
        const quickStartLongBtn = document.getElementById('quickStartLongBtn');
        if (quickStartLongBtn) {
            quickStartLongBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для запуска', 'warning');
                    return;
                }
                
                await this.quickLaunchBot('LONG');
            });
        }
        
        // Кнопка быстрого запуска SHORT
        const quickStartShortBtn = document.getElementById('quickStartShortBtn');
        if (quickStartShortBtn) {
            quickStartShortBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для запуска', 'warning');
                    return;
                }
                
                await this.quickLaunchBot('SHORT');
            });
        }
        
        // Кнопка быстрой остановки
        const quickStopBtn = document.getElementById('quickStopBtn');
        if (quickStopBtn) {
            quickStopBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для остановки', 'warning');
                    return;
                }
                
                await this.stopBot();
            });
        }
        
        // Обработчики для кнопок ручного запуска в секции настроек
        const manualLaunchLongBtn = document.getElementById('manualLaunchLongBtn');
        if (manualLaunchLongBtn) {
            manualLaunchLongBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для запуска', 'warning');
                    return;
                }
                
                await this.quickLaunchBot('LONG');
            });
        }
        
        const manualLaunchShortBtn = document.getElementById('manualLaunchShortBtn');
        if (manualLaunchShortBtn) {
            manualLaunchShortBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для запуска', 'warning');
                    return;
                }
                
                await this.quickLaunchBot('SHORT');
            });
        }
        
        console.log('[BotsManager] ✅ Кнопки быстрого запуска инициализированы');
    }
    });
})();
