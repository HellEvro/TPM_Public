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
        let takeProfit = bot.exchange_position?.take_profit || bot.take_profit || bot.take_profit_price || bot.trailing_take_profit_price || '';
        if (!stopLoss && entryPrice) {
            const pct = (bot.config?.max_loss_percent ?? bot.max_loss_percent) || 15.0;
            stopLoss = bot.position_side === 'LONG' ? entryPrice * (1 - pct / 100) : entryPrice * (1 + pct / 100);
        }
        if (!takeProfit && entryPrice) {
            const tpPct = (bot.config?.take_profit_percent ?? bot.take_profit_percent) || 20.0;
            takeProfit = bot.position_side === 'LONG' ? entryPrice * (1 + tpPct / 100) : entryPrice * (1 - tpPct / 100);
        }
        const volMode = (bot.volume_mode || 'USDT').toUpperCase();
        const volVal = bot.volume_value ?? (entryPrice > 0 ? (bot.position_size || 0) * entryPrice : 0);
        const volStr = volMode === 'PERCENT' ? `${parseFloat(volVal || 0).toFixed(2)} ${volMode}` : `${parseFloat(volVal || 0).toFixed(2)} ${volMode}`;
        const sideColor = bot.position_side === 'LONG' ? 'var(--green-color)' : 'var(--red-color)';
        return {
            volume: volStr,
            position: bot.position_side || '-',
            positionColor: sideColor,
            entry: entryPrice ? `$${entryPrice.toFixed(6)}` : '-',
            takeProfit: takeProfit ? `$${parseFloat(takeProfit).toFixed(6)}` : '-',
            stopLoss: stopLoss ? `$${parseFloat(stopLoss).toFixed(6)}` : '-',
            currentPrice: currentPrice ? `$${currentPrice.toFixed(6)}` : '-'
        };
    },
            getBotPositionInfo(bot) {
        // Проверяем, есть ли активная позиция
        if (!bot.position_side || !bot.entry_price) {
            // Если нет активной позиции, показываем информацию о статусе бота
            let statusText = '';
            let statusColor = 'var(--text-muted)';
            let statusIcon = '📍';
            
            if (bot.status === 'in_position_long') {
                statusText = window.languageUtils.translate('long_closed');
                statusColor = 'var(--green-color)';
                statusIcon = '📈';
            } else if (bot.status === 'in_position_short') {
                statusText = window.languageUtils.translate('short_closed');
                statusColor = 'var(--red-color)';
                statusIcon = '📉';
            } else if (bot.status === 'running' || bot.status === 'waiting') {
                statusText = window.languageUtils.translate('entry_by_market');
                statusColor = 'var(--blue-color)';
                statusIcon = '🔄';
            } else {
                statusText = window.languageUtils.translate('no_position');
                statusColor = 'var(--text-muted)';
                statusIcon = '📍';
            }
            
            return `<div style="display: flex; justify-content: space-between; margin-bottom: 4px;"><span style="color: var(--text-muted);">${statusIcon} ${this.getTranslation('position_label')}:</span><span style="color: ${statusColor};">${statusText}</span></div>`;
        }
        
        const sideColor = bot.position_side === 'LONG' ? 'var(--green-color)' : 'var(--red-color)';
        const sideIcon = bot.position_side === 'LONG' ? '📈' : '📉';
        
        let positionHtml = `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: var(--input-bg); border-radius: 6px;">
                <span style="color: var(--text-muted);">${sideIcon} ${this.getTranslation('position_label')}</span>
                <span style="color: ${sideColor}; font-weight: 600;">${bot.position_side}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: var(--input-bg); border-radius: 6px;">
                <span style="color: var(--text-muted);">💵 ${this.getTranslation('entry_label')}</span>
                <span style="color: var(--text-color); font-weight: 600;">$${(parseFloat(bot.entry_price) || 0).toFixed(6)}</span>
            </div>
        `;
        
        // ✅ ИСПРАВЛЕНО: Используем current_price напрямую из bot (обновляется каждую секунду)
        if (bot.current_price || bot.mark_price) {
            const currentPrice = parseFloat(bot.current_price || bot.mark_price) || 0;
            const entryPrice = parseFloat(bot.entry_price) || 0;
            const priceChange = entryPrice > 0 ? ((currentPrice - entryPrice) / entryPrice) * 100 : 0;
            const priceChangeColor = priceChange >= 0 ? 'var(--green-color)' : 'var(--red-color)';
            const priceChangeIcon = priceChange >= 0 ? '↗️' : '↘️';
            
            positionHtml += `
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: var(--input-bg); border-radius: 6px;">
                    <span style="color: var(--text-muted);">📊 ${this.getTranslation('current_label')}</span>
                    <span style="color: ${priceChangeColor}; font-weight: 600;">$${currentPrice.toFixed(6)} ${priceChangeIcon}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: var(--input-bg); border-radius: 6px;">
                    <span style="color: var(--text-muted);">📈 ${this.getTranslation('change_label')}</span>
                    <span style="color: ${priceChangeColor}; font-weight: 600;">${priceChange.toFixed(2)}%</span>
                </div>
            `;
        }
        
        // Добавляем стоп-лосс и тейк-профит (используем данные с биржи)
        let stopLoss = bot.exchange_position?.stop_loss || '';
        let takeProfit = bot.exchange_position?.take_profit || '';
        
        // Если стоп-лосс не установлен на бирже, рассчитываем на основе настроек бота
        if (!stopLoss && bot.entry_price) {
            const stopLossPercent = bot.max_loss_percent || 15.0;
            if (bot.position_side === 'LONG') {
                stopLoss = bot.entry_price * (1 - stopLossPercent / 100);
            } else if (bot.position_side === 'SHORT') {
                stopLoss = bot.entry_price * (1 + stopLossPercent / 100);
            }
        }
        
        // Если тейк-профит не установлен, рассчитываем на основе RSI настроек бота
        if (!takeProfit && bot.entry_price) {
            const rsiExitLong = bot.rsi_exit_long || 55;
            const rsiExitShort = bot.rsi_exit_short || 45;
            // Получаем RSI с учетом текущего таймфрейма
            const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
            const rsiKey = `rsi${currentTimeframe}`;
            const currentRsi = bot.rsi_data?.[rsiKey] || bot.rsi_data?.rsi6h || bot.rsi_data?.rsi || 50;
            
            if (bot.position_side === 'LONG' && currentRsi < rsiExitLong) {
                const takeProfitPercent = (rsiExitLong - currentRsi) * 0.5;
                takeProfit = bot.entry_price * (1 + takeProfitPercent / 100);
            } else if (bot.position_side === 'SHORT' && currentRsi > rsiExitShort) {
                const takeProfitPercent = (currentRsi - rsiExitShort) * 0.5;
                takeProfit = bot.entry_price * (1 - takeProfitPercent / 100);
            }
        }
        
        positionHtml += `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: var(--input-bg); border-radius: 6px;">
                <span style="color: var(--text-muted);">🛡️ ${this.getTranslation('stop_loss_label_detailed')}</span>
                <span style="color: ${stopLoss ? 'var(--warning-color)' : 'var(--text-muted)'}; font-weight: 600;">${stopLoss ? `$${parseFloat(stopLoss).toFixed(6)}` : this.getTranslation('not_set')}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: var(--input-bg); border-radius: 6px;">
                <span style="color: var(--text-muted);">🎯 ${this.getTranslation('take_profit_label_detailed')}</span>
                <span style="color: ${takeProfit ? 'var(--green-color)' : 'var(--text-muted)'}; font-weight: 600;">${takeProfit ? `$${parseFloat(takeProfit).toFixed(6)}` : this.getTranslation('not_set')}</span>
            </div>
        `;
        
        // Добавляем RSI данные если есть
        if (bot.rsi_data) {
            // Получаем RSI и тренд с учетом текущего таймфрейма
            const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
            const rsiKey = `rsi${currentTimeframe}`;
            const trendKey = `trend${currentTimeframe}`;
            const rsi = bot.rsi_data[rsiKey] || bot.rsi_data.rsi6h || bot.rsi_data.rsi || 50;
            const trend = bot.rsi_data[trendKey] || bot.rsi_data.trend6h || bot.rsi_data.trend || 'NEUTRAL';
            
            if (rsi) {
                let rsiColor = 'var(--text-muted)';
                if (rsi > 70) rsiColor = 'var(--red-color)'; // Перекупленность
                else if (rsi < 30) rsiColor = 'var(--green-color)'; // Перепроданность
                
                positionHtml += `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: var(--input-bg); border-radius: 6px;">
                        <span style="color: var(--text-muted);">📊 RSI</span>
                        <span style="color: ${rsiColor}; font-weight: 600;">${rsi.toFixed(1)}</span>
                    </div>
                `;
            }
            
            if (trend) {
                let trendColor = 'var(--text-muted)';
                let trendIcon = '➡️';
                if (trend === 'UP') { trendColor = 'var(--green-color)'; trendIcon = '📈'; }
                else if (trend === 'DOWN') { trendColor = 'var(--red-color)'; trendIcon = '📉'; }
                else if (trend === 'NEUTRAL') { trendColor = 'var(--warning-color)'; trendIcon = '➡️'; }
                
                positionHtml += `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: var(--input-bg); border-radius: 6px;">
                        <span style="color: var(--text-muted);">${trendIcon} ${this.getTranslation('trend_label')}</span>
                        <span style="color: ${trendColor}; font-weight: 600;">${trend}</span>
                    </div>
                `;
            }
        }
        
        return positionHtml;
    },
            getBotTimeInfo(bot) {
        let timeInfoHtml = '';
        
        // Время работы бота
        if (bot.created_at) {
        const createdTime = new Date(bot.created_at);
        const now = new Date();
        const timeDiff = now - createdTime;
        const hours = Math.floor(timeDiff / (1000 * 60 * 60));
        const minutes = Math.floor((timeDiff % (1000 * 60 * 60)) / (1000 * 60));
        
        let timeText = '';
        if (hours > 0) {
            timeText = `${hours}ч ${minutes}м`;
        } else {
            timeText = `${minutes}м`;
        }
        
            timeInfoHtml += `
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="color: var(--text-muted);">⏱️ ${window.languageUtils.translate('time_label')}</span>
                <span style="color: var(--text-color); font-weight: 500;">${timeText}</span>
            </div>
        `;
        }
        
        // Время обновления данных позиции (если бот в позиции)
        if (bot.status && (bot.status.includes('position') || bot.status.includes('in_position')) && bot.last_update) {
            const lastUpdateTime = new Date(bot.last_update);
            const now = new Date();
            const updateDiff = now - lastUpdateTime;
            const updateMinutes = Math.floor(updateDiff / (1000 * 60));
            const updateSeconds = Math.floor((updateDiff % (1000 * 60)) / 1000);
            
            let updateTimeText = '';
            if (updateMinutes > 0) {
                updateTimeText = `${updateMinutes}м ${updateSeconds}с назад`;
            } else {
                updateTimeText = `${updateSeconds}с назад`;
            }
            
            // Цвет в зависимости от давности обновления
            let updateColor = 'var(--green-color)'; // зеленый - свежие данные
            if (updateMinutes > 1) {
                updateColor = 'var(--warning-color)'; // оранжевый - данные старше минуты
            }
            if (updateMinutes > 5) {
                updateColor = 'var(--red-color)'; // красный - данные старше 5 минут
            }
            
            timeInfoHtml += `
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: var(--text-muted);">🔄 ${this.getTranslation('updated_label')}</span>
                    <span style="color: ${updateColor}; font-weight: 500;">${updateTimeText}</span>
                </div>
            `;
        }
        
        return timeInfoHtml;
    },
            renderTradesInfo(coinSymbol) {
        console.log(`[DEBUG] renderTradesInfo для ${coinSymbol}`);
        console.log(`[DEBUG] this.activeBots:`, this.activeBots);
        console.log(`[DEBUG] this.selectedCoin:`, this.selectedCoin);
        
        const tradesSection = document.getElementById('tradesInfoSection');
        const tradesContainer = document.getElementById('tradesContainer');
        
        console.log(`[DEBUG] tradesSection:`, tradesSection);
        console.log(`[DEBUG] tradesContainer:`, tradesContainer);
        
        if (!tradesSection || !tradesContainer) {
            console.log(`[DEBUG] Не найдены элементы tradesSection или tradesContainer`);
            return;
        }
        
        // Находим бота для этой монеты
        const bot = this.activeBots.find(b => b.symbol === coinSymbol);
        
        console.log(`[DEBUG] Найденный бот для ${coinSymbol}:`, bot);
        
        if (!bot) {
            console.log(`[DEBUG] Бот не найден для ${coinSymbol}`);
            tradesSection.style.display = 'none';
            return;
        }
        
        // Показываем секцию сделок
        console.log(`[DEBUG] Показываем секцию сделок для ${coinSymbol}`);
        tradesSection.style.display = 'block';
        
        // Получаем информацию о сделках
        const trades = this.getBotTrades(bot);
        
        console.log(`[DEBUG] Полученные сделки для ${coinSymbol}:`, trades);
        
        if (trades.length === 0) {
            console.log(`[DEBUG] Нет активных сделок для ${coinSymbol}`);
            tradesContainer.innerHTML = '<div class="no-trades">Нет активных сделок</div>';
            return;
        }
        
        // Рендерим сделки
        const tradesHtml = trades.map(trade => this.renderTradeItem(trade)).join('');
        console.log(`[DEBUG] HTML для сделок ${coinSymbol}:`, tradesHtml);
        tradesContainer.innerHTML = tradesHtml;
    },
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
        
        // Проверяем, есть ли позиция LONG
        if (bot.position_side === 'LONG' && bot.entry_price) {
            console.log(`[DEBUG] Создаем LONG позицию для ${bot.symbol}`);
            
            // Используем данные с биржи для стоп-лосса и тейк-профита
            const stopLossPrice = bot.exchange_position?.stop_loss || bot.entry_price * 0.95; // Используем данные с биржи или 5% от входа
            const takeProfitPrice = bot.exchange_position?.take_profit || null; // Используем данные с биржи
            
            // Если нет данных с биржи, рассчитываем на основе настроек бота
            let calculatedStopLoss = stopLossPrice;
            let calculatedTakeProfit = takeProfitPrice;
            
            if (!bot.exchange_position?.stop_loss) {
                const stopLossPercent = bot.max_loss_percent || 15.0;
                calculatedStopLoss = bot.entry_price * (1 - stopLossPercent / 100);
            }
            
            if (!bot.exchange_position?.take_profit) {
                const rsiExitLong = bot.rsi_exit_long || 55;
                
                if (currentRsi < rsiExitLong) {
                    // Рассчитываем тейк-профит как процент от входа
                    const takeProfitPercent = (rsiExitLong - currentRsi) * 0.5; // Примерная формула
                    calculatedTakeProfit = bot.entry_price * (1 + takeProfitPercent / 100);
                }
            }
            
            // Рассчитываем объем в USDT точно
            const volumeInTokens = bot.position_size || 0; // Количество токенов (70 AWE)
            const volumeInUsdt = parseFloat((volumeInTokens * bot.entry_price).toFixed(2)); // Точный объем в USDT (70 * 0.074190 = 5.19 USDT)
            
            console.log(`[DEBUG] Расчеты для ${bot.symbol}:`, {
                volumeInTokens,
                volumeInUsdt,
                calculatedStopLoss,
                calculatedTakeProfit
            });
            
            trades.push({
                side: 'LONG',
                entryPrice: bot.entry_price,
                currentPrice: bot.current_price || bot.mark_price || bot.entry_price,
                stopLossPrice: calculatedStopLoss,
                stopLossPercent: bot.max_loss_percent || 15.0,
                takeProfitPrice: calculatedTakeProfit,
                pnl: bot.unrealized_pnl || 0,
                status: 'active',
                volume: volumeInUsdt, // Объем в USDT
                volumeInTokens: volumeInTokens, // Количество токенов
                volumeMode: 'USDT',
                startTime: bot.created_at,
                rsi: currentRsi,
                // Получаем тренд с учетом текущего таймфрейма
                trend: (() => {
                    const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
                    const trendKey = `trend${currentTimeframe}`;
                    return bot[trendKey] || bot.trend6h || bot.trend || 'NEUTRAL';
                })(),
                workTime: bot.work_time || '0м',
                lastUpdate: bot.last_update || 'Неизвестно'
            });
        } else {
            console.log(`[DEBUG] Нет LONG позиции для ${bot.symbol}:`, {
                position_side: bot.position_side,
                entry_price: bot.entry_price
            });
        }
        
        // Проверяем, есть ли позиция SHORT (для кросс-сделок)
        if (bot.position_side === 'SHORT' && bot.entry_price) {
            // Используем данные с биржи для стоп-лосса и тейк-профита
            const stopLossPrice = bot.exchange_position?.stop_loss || bot.entry_price * 1.05; // Используем данные с биржи или 5% от входа
            const takeProfitPrice = bot.exchange_position?.take_profit || null; // Используем данные с биржи
            
            // Если нет данных с биржи, рассчитываем на основе настроек бота
            let calculatedStopLoss = stopLossPrice;
            let calculatedTakeProfit = takeProfitPrice;
            
            if (!bot.exchange_position?.stop_loss) {
                const stopLossPercent = bot.max_loss_percent || 15.0;
                calculatedStopLoss = bot.entry_price * (1 + stopLossPercent / 100);
            }
            
            if (!bot.exchange_position?.take_profit) {
                const rsiExitShort = bot.rsi_exit_short || 45;
                
                if (currentRsi > rsiExitShort) {
                    // Рассчитываем тейк-профит как процент от входа
                    const takeProfitPercent = (currentRsi - rsiExitShort) * 0.5; // Примерная формула
                    calculatedTakeProfit = bot.entry_price * (1 - takeProfitPercent / 100);
                }
            }
            
            // Рассчитываем объем в USDT точно
            const volumeInTokens = bot.position_size || 0; // Количество токенов
            const volumeInUsdt = parseFloat((volumeInTokens * bot.entry_price).toFixed(2)); // Точный объем в USDT
            
            trades.push({
                side: 'SHORT',
                entryPrice: bot.entry_price,
                currentPrice: bot.current_price || bot.mark_price || bot.entry_price,
                stopLossPrice: calculatedStopLoss,
                stopLossPercent: bot.max_loss_percent || 15.0,
                takeProfitPrice: calculatedTakeProfit,
                pnl: bot.unrealized_pnl || 0,
                status: 'active',
                volume: volumeInUsdt, // Объем в USDT
                volumeInTokens: volumeInTokens, // Количество токенов
                volumeMode: 'USDT',
                startTime: bot.created_at,
                rsi: currentRsi,
                // Получаем тренд с учетом текущего таймфрейма
                trend: (() => {
                    const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
                    const trendKey = `trend${currentTimeframe}`;
                    return bot[trendKey] || bot.trend6h || bot.trend || 'NEUTRAL';
                })(),
                workTime: bot.work_time || '0м',
                lastUpdate: bot.last_update || 'Неизвестно'
            });
        }
        
        return trades;
    },
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
        const refreshBtn = document.getElementById('refreshManualPositionsBtn');
        if (!refreshBtn) {
            console.warn('[BotsManager] ⚠️ Кнопка refreshManualPositionsBtn не найдена в DOM. Попытка повторной инициализации через 1 секунду...');
            // Повторная попытка через 1 секунду (на случай, если DOM еще не загружен)
            setTimeout(() => {
                this.initializeManualPositionsControls();
            }, 1000);
            return;
        }
        
        console.log('[BotsManager] ✅ Кнопка refreshManualPositionsBtn найдена, добавляем обработчик...');
        
        // Удаляем старый обработчик, если он есть (для предотвращения дублирования)
        const newRefreshBtn = refreshBtn.cloneNode(true);
        refreshBtn.parentNode.replaceChild(newRefreshBtn, refreshBtn);
        
        newRefreshBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('[BotsManager] 🔄 Обновление ручных позиций...');
            
            // Блокируем кнопку на время запроса
            newRefreshBtn.disabled = true;
            const originalContent = newRefreshBtn.innerHTML;
            newRefreshBtn.innerHTML = '<span>⏳</span>';
            
            try {
                const response = await fetch(`${this.apiUrl}/manual-positions/refresh`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('[BotsManager] ✅ Ручные позиции обновлены:', result);
                    
                    // Принудительно обновляем список (без проверки data_version), чтобы применился новый manual_positions
                    await this.loadCoinsRsiData(true);
                    
                    // Показываем уведомление
                    if (window.showToast) {
                        window.showToast(`${window.languageUtils.translate('updated')} ${result.count || 0} ${window.languageUtils.translate('manual_positions')}`, 'success');
                    }
                } else {
                    const errorText = await response.text();
                    throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
                }
            } catch (error) {
                console.error('[BotsManager] ❌ Ошибка обновления ручных позиций:', error);
                if (window.showToast) {
                    window.showToast(`Ошибка обновления: ${error.message}`, 'error');
                }
            } finally {
                // Разблокируем кнопку
                newRefreshBtn.disabled = false;
                newRefreshBtn.innerHTML = originalContent;
            }
        });
        
        console.log('[BotsManager] ✅ Обработчик для кнопки обновления ручных позиций успешно добавлен');
    }
    
    /**
     * Инициализирует кнопки загрузки RSI данных
     */,
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
