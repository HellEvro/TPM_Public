/**
 * BotsManager - 14_history_display
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            displayBotActions(actions) {
        const container = document.getElementById('botActionsList');
        if (!container) return;,
            displayBotTrades(trades) {
        const container = document.getElementById('botTradesList');
        if (!container) return;,
            displayBotSignals(signals) {
        const container = document.getElementById('botSignalsList');
        if (!container) return;,
            displayHistoryStatistics(stats) {
        // Обновляем карточки статистики
        const totalActionsEl = document.querySelector('.history-stats .stat-card:nth-child(1) .stat-value');
        const totalTradesEl = document.querySelector('.history-stats .stat-card:nth-child(2) .stat-value');
        const totalPnlEl = document.querySelector('.history-stats .stat-card:nth-child(3) .stat-value');
        const successRateEl = document.querySelector('.history-stats .stat-card:nth-child(4) .stat-value');
        
        const totalActions = typeof stats.total_actions === 'number' ? stats.total_actions : 0;
        const totalTrades = typeof stats.total_trades === 'number' ? stats.total_trades : 0;
        const totalPnL = typeof stats.total_pnl === 'number' ? stats.total_pnl : 0;
        const successRate = typeof stats.success_rate === 'number'
            ? stats.success_rate
            : (typeof stats.win_rate === 'number' ? stats.win_rate : 0);

        if (totalActionsEl) totalActionsEl.textContent = totalActions;
        if (totalTradesEl) totalTradesEl.textContent = totalTrades;
        if (totalPnlEl) totalPnlEl.textContent = `$${totalPnL.toFixed(2)}`;
        if (successRateEl) successRateEl.textContent = `${successRate.toFixed(1)}%`;

        if (Array.isArray(stats.symbols)) {
            this.updateHistoryBotFilterOptions(stats.symbols);
        }
    },
            updateHistoryBotFilterOptions(symbols = []) {
        const botFilter = document.getElementById('historyBotFilter');,
            clearHistoryFilters() {
        const botFilter = document.getElementById('historyBotFilter');
        const actionFilter = document.getElementById('historyActionFilter');
        const dateFilter = document.getElementById('historyDateFilter');
        
        if (botFilter) botFilter.value = 'all';
        if (actionFilter) actionFilter.value = 'all';
        if (dateFilter) dateFilter.value = 'all';
        
        this.loadHistoryData();
    }

    /**
     * Экспортирует данные истории
     */,
            exportHistoryData() {
        console.log('[BotsManager] 📤 Экспорт данных истории (функция в разработке)');
        this.showNotification('Функция экспорта в разработке', 'info');
    }

    /**
     * Создает демо-данные истории
     */,
            async createDemoHistoryData() {
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/history/demo`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();,
            async clearAllHistory() {
        if (!confirm('Вы уверены, что хотите очистить всю историю? Это действие нельзя отменить.')) {
            return;
        }
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/history/clear`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });
            
            const data = await response.json();,
            getActionIcon(actionType) {
        const icons = {
            'BOT_START': '🚀',
            'BOT_STOP': '🛑',
            'SIGNAL': '⚡',
            'POSITION_OPENED': '📈',
            'POSITION_CLOSED': '📉',
            'STOP_LOSS': '🛡️',
            'TAKE_PROFIT': '🎯',
            'TRAILING_STOP': '📊',
            'ERROR': '❌'
        };
        return icons[actionType] || '📋';
    }

    /**
     * Форматирует timestamp
     */,
            formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString('ru-RU', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    },
            formatDuration(seconds) {,
            saveCollapseState(symbol, isCollapsed) {
        // Сохраняем состояние сворачивания для конкретного бота,
            preserveCollapseState(container) {
        // Восстанавливаем сохраненное состояние сворачивания для каждого бота
    });
})();
