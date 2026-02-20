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
        const rsiExitLongWithTrendEl = document.getElementById('rsiExitLongWithTrendDup');,
            async loadIndividualSettings(symbol) {
        if (!symbol) return null;
        
        try {
            console.log(`[BotsManager] 📥 Загрузка индивидуальных настроек для ${symbol}`);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}`);
            
            // 404 - это нормально, значит настроек нет,
            async saveIndividualSettings(symbol, settings) {
        if (!symbol || !settings) return false;
        
        try {
            console.log(`[BotsManager] 💾 Сохранение индивидуальных настроек для ${symbol}:`, settings);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            
            const data = await response.json();,
            async deleteIndividualSettings(symbol) {
        if (!symbol) return false;
        
        try {
            console.log(`[BotsManager] 🗑️ Удаление индивидуальных настроек для ${symbol}`);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();,
            async copySettingsToAllCoins(symbol) {
        if (!symbol) return false;
        
        try {
            console.log(`[BotsManager] 📋 Копирование настроек ${symbol} ко всем монетам`);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}/copy-to-all`, {
                method: 'POST'
            });
            
            const data = await response.json();,
            async learnExitScamForCoin() {,
            async learnExitScamForAllCoins() {
        const btn = document.getElementById('learnExitScamForAllCoinsBtn');
        const originalText = btn ? btn.innerHTML : '';
        try {,
            async resetExitScamToConfigForAll() {
        const btn = document.getElementById('resetExitScamToConfigForAllBtn');
        const originalText = btn ? btn.innerHTML : '';
        try {,
            async resetAllCoinsToGlobalSettings() {
        try {
            const confirmed = confirm('⚠️ Вы уверены, что хотите сбросить индивидуальные настройки ВСЕХ монет к глобальным настройкам?\n\nЭто действие нельзя отменить!');,
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
            let mainVal = getMainValue(configKey);,
            applyIndividualSettingsToUI(settings) {
        if (!settings) return;
        
        console.log('[BotsManager] 🎨 Применение индивидуальных настроек к UI:', settings);
        const fallbackConfig = this.cachedAutoBotConfig || {};
        const getSettingValue = (key) => {
            if (settings[key] !== undefined) return settings[key];
            return fallbackConfig[key];
        };
        
        // RSI настройки
        const rsiLongEl = document.getElementById('rsiLongThresholdDup');,
            updateIndividualSettingsStatus(hasSettings) {
        const statusEl = document.getElementById('individualSettingsStatus');,
            async loadAndApplyIndividualSettings(symbol) {
        if (!symbol) return;
        
        try {
            console.log(`[BotsManager] 📥 Загрузка и применение индивидуальных настроек для ${symbol}`);
            this.pendingIndividualSettingsSymbol = symbol;
             const settings = await this.loadIndividualSettings(symbol);,
            initializeIndividualSettingsButtons() {
        console.log('[BotsManager] 🔧 Инициализация кнопок индивидуальных настроек...');
        
        // Кнопка сохранения индивидуальных настроек
        const saveIndividualBtn = document.getElementById('saveIndividualSettingsBtn');,
            initializeQuickLaunchButtons() {
        console.log('[BotsManager] 🚀 Инициализация кнопок быстрого запуска...');
        
        // Кнопка быстрого запуска LONG
        const quickStartLongBtn = document.getElementById('quickStartLongBtn');
    });
})();
