/**
 * BotsManager - 09_periodic
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            updateBotsSummaryStats() {
        this.logDebug('[BotsManager] 📊 Обновление статистики ботов...');
        const bots = Array.isArray(this.activeBots) ? this.activeBots : [];

        const activeStatuses = new Set([
            'running',
            'idle',
            'in_position_long',
            'in_position_short',
            'armed_up',
            'armed_down'
        ]);

        let totalPnL = 0;
        let activeCount = 0;
        let inPositionCount = 0;

        bots.forEach(bot => {
            const rawPnL = bot.unrealized_pnl_usdt ?? bot.unrealized_pnl ?? 0;
            const botPnL = Number.parseFloat(rawPnL) || 0;
            totalPnL += botPnL;

            if (activeStatuses.has(bot.status)) {
                activeCount += 1;
            },
            startPeriodicUpdate() {
        // Обновляем данные с единым интервалом
        this.updateInterval = setInterval(() => {,
            startBotMonitoring() {
        console.log('[BotsManager] 📊 Запуск мониторинга активных ботов...');
        
        // Останавливаем предыдущий таймер если есть,
            stopBotMonitoring() {,
            updateBotsDetailedDisplay(bots) {
        // Обновляем отображение каждого бота с детальной информацией
        bots.forEach(bot => {
            this.updateSingleBotDisplay(bot);
        });
    },
            updateSingleBotDisplay(bot) {
        const pnl = bot.pnl || 0;
        const price = bot.current_price != null ? Number(bot.current_price).toFixed(6) : '';
        const side = bot.position_side || '';
        const trailing = !!bot.trailing_stop_active;
        const key = `${bot.symbol}|${pnl}|${price}|${side}|${trailing}`;
        if (!this._lastBotDisplay) this._lastBotDisplay = {};
        if (this._lastBotDisplay[bot.symbol] === key) return;
        this._lastBotDisplay[bot.symbol] = key;
        
        const botElement = document.querySelector(`[data-bot-symbol="${bot.symbol}"]:not([data-is-virtual="true"])`);
        if (!botElement) return;
        
        const pnlElement = botElement.querySelector('.bot-pnl');,
            calculateTimeLeft(startTime, maxHours, maxHoursIsHours = true) {
        const start = new Date(startTime);
        const now = new Date();
        const elapsed = now - start;
        const maxMs = (maxHoursIsHours ? maxHours * 3600 : maxHours) * 1000;
        const remaining = maxMs - elapsed;,
            destroy() {
    });
})();
