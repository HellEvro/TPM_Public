/**
 * BotsManager - 11_trades_display
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            getCompactCardData(bot) {
        const entryPrice = parseFloat(bot.entry_price) || 0;
        const currentPrice = parseFloat(bot.current_price || bot.mark_price) || 0;
        let stopLoss = bot.exchange_position?.stop_loss || bot.stop_loss || bot.stop_loss_price || '';
        let takeProfit = bot.exchange_position?.take_profit || bot.take_profit || bot.take_profit_price || bot.trailing_take_profit_price || '';,
            getBotPositionInfo(bot) {
        // Проверяем, есть ли активная позиция,
            getBotTimeInfo(bot) {
        let timeInfoHtml = '';
        
        // Время работы бота,
            renderTradesInfo(coinSymbol) {
        console.log(`[DEBUG] renderTradesInfo для ${coinSymbol}`);
        console.log(`[DEBUG] this.activeBots:`, this.activeBots);
        console.log(`[DEBUG] this.selectedCoin:`, this.selectedCoin);
        
        const tradesSection = document.getElementById('tradesInfoSection');
        const tradesContainer = document.getElementById('tradesContainer');
        
        console.log(`[DEBUG] tradesSection:`, tradesSection);
        console.log(`[DEBUG] tradesContainer:`, tradesContainer);,
            getBotTrades(bot) {
        console.log(`[DEBUG] getBotTrades для ${bot.symbol}:`, {
            position_side: bot.position_side,
            entry_price: bot.entry_price,
            position_size: bot.position_size,
            exchange_position: bot.exchange_position
        });
        
        const trades = [];
        
        // Определяем currentRsi в начале функции для использования во всех блоках
        const currentRsi = bot.rsi_data?.rsi6h || 50;
        
        // Проверяем, есть ли позиция LONG,
            renderTradeItem(trade) {
        const sideIcon = trade.side === 'LONG' ? '📈' : '📉';
        const sideClass = trade.side.toLowerCase();
        const pnlClass = trade.pnl >= 0 ? 'positive' : 'negative';
        const pnlIcon = trade.pnl >= 0 ? '↗️' : '↘️';
        
        // Рассчитываем изменение цены в процентах
        const priceChange = trade.side === 'LONG' 
            ? ((trade.currentPrice - trade.entryPrice) / trade.entryPrice) * 100
            : ((trade.entryPrice - trade.currentPrice) / trade.entryPrice) * 100;
        
        const priceChangeClass = priceChange >= 0 ? 'positive' : 'negative';
        
        return `
            <div class="trade-item" style="border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; margin: 8px 0; background: var(--section-bg); transition: all 0.3s ease;" onmouseover="this.style.backgroundColor='var(--hover-bg, var(--button-bg))'" onmouseout="this.style.backgroundColor='var(--section-bg)'">
                <div class="trade-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color);">
                    <div class="trade-side ${sideClass}" style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 16px;">${sideIcon}</span>
                        <span style="color: ${trade.side === 'LONG' ? 'var(--green-color)' : 'var(--red-color)'}; font-weight: bold;">${trade.side}</span>
                    </div>
                    <div class="trade-status ${trade.status}" style="background: ${trade.status === 'active' ? 'var(--green-color)' : 'var(--red-bright)'}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px;">
                        ${trade.status === 'active' ? window.languageUtils.translate('active_trade_status') : window.languageUtils.translate('closed_trade_status')}
                    </div>
                </div>
                
                <div class="trade-details" style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; color: var(--text-color);">
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: var(--input-bg); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: var(--text-muted);">${window.languageUtils.translate('entry_price_label')}</span>
                        <span class="trade-detail-value" style="color: var(--text-color); font-weight: 600;">$${(parseFloat(trade.entryPrice) || 0).toFixed(6)}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: var(--input-bg); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: var(--text-muted);">${window.languageUtils.translate('current_price_label')}</span>
                        <span class="trade-detail-value" style="color: var(--text-color); font-weight: 600;">$${(parseFloat(trade.currentPrice) || 0).toFixed(6)}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: var(--input-bg); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: var(--text-muted);">${window.languageUtils.translate('change_price_label')}</span>
                        <span class="trade-detail-value ${priceChangeClass}" style="color: ${priceChange >= 0 ? 'var(--green-color)' : 'var(--red-color)'}; font-weight: 600;">${priceChange.toFixed(2)}%</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: var(--input-bg); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: var(--text-muted);">${window.languageUtils.translate('volume_price_label')}</span>
                        <span class="trade-detail-value" style="color: var(--text-color); font-weight: 600;">${trade.volume.toFixed(2)} ${trade.volumeMode.toUpperCase()}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: var(--input-bg); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: var(--text-muted);">${window.languageUtils.translate('stop_loss_price_label')}</span>
                        <span class="trade-detail-value" style="color: var(--warning-color); font-weight: 600;">$${parseFloat(trade.stopLossPrice).toFixed(6)} (${trade.stopLossPercent}%)</span>
                    </div>
                    
                    ${trade.takeProfitPrice ? `
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: var(--input-bg); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: var(--text-muted);">${window.languageUtils.translate('take_profit_price_label')}</span>
                        <span class="trade-detail-value" style="color: var(--green-color); font-weight: 600;">$${parseFloat(trade.takeProfitPrice).toFixed(6)}</span>
                    </div>
                    ` : ''}
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: var(--input-bg); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: var(--text-muted);">${window.languageUtils.translate('rsi_label')}</span>
                        <span class="trade-detail-value" style="color: var(--text-color); font-weight: 600;">${trade.rsi ? trade.rsi.toFixed(1) : 'N/A'}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: var(--input-bg); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: var(--text-muted);">➡️ ${window.languageUtils.translate('trend_label')}:</span>
                        <span class="trade-detail-value" style="color: ${trade.trend === 'UP' ? 'var(--green-color)' : trade.trend === 'DOWN' ? 'var(--red-color)' : 'var(--warning-color)'}; font-weight: 600;">${trade.trend || 'NEUTRAL'}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: var(--input-bg); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: var(--text-muted);">${window.languageUtils.translate('time_detail_label')}</span>
                        <span class="trade-detail-value" style="color: var(--text-color); font-weight: 600;">${trade.workTime || '0м'}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: var(--input-bg); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: var(--text-muted);">${window.languageUtils.translate('updated_detail_label')}</span>
                        <span class="trade-detail-value" style="color: var(--text-color); font-weight: 600;">${trade.lastUpdate || window.languageUtils.translate('unknown')}</span>
                    </div>
                </div>
                
                <div class="trade-pnl ${pnlClass}">
                    <span>${pnlIcon} PnL:</span>
                    <span>$${trade.pnl.toFixed(3)}</span>
                </div>
            </div>
        `;
    }
    
    /**
     * Инициализирует обработчики для кнопки обновления ручных позиций
     */,
            initializeManualPositionsControls() {
        console.log('[BotsManager] 🔄 Инициализация кнопки обновления ручных позиций...');
        
        // Кнопка обновления ручных позиций
        const refreshBtn = document.getElementById('refreshManualPositionsBtn');,
            initializeRSILoadingButtons() {
        console.log('[BotsManager] 🚀 Инициализация кнопок загрузки RSI... (кнопки удалены - используется инкрементальная загрузка)');
    }
    
    /**
     * Запускает загрузку RSI данных (устаревшая функция - удалена)
     * Теперь используется инкрементальная загрузка автоматически
     */
    
    /**
     * Загружает счётчик зрелых монет
     */
    });
})();
