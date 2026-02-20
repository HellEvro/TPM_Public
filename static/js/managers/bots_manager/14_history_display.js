/**
 * BotsManager - 14_history_display
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            displayBotActions(actions) {
        const container = document.getElementById('botActionsList');
        if (!container) return;
        
        if (actions.length === 0) {
            container.innerHTML = `
                <div class="empty-history-state">
                    <div class="empty-icon">📊</div>
                    <p data-translate="no_actions_found">История действий не найдена</p>
                    <p data-translate="actions_will_appear">Действия ботов будут отображаться здесь</p>
                </div>
            `;
            return;
        }
        
        const html = actions.map(action => `
            <div class="history-item">
                <div class="history-item-header">
                    <span class="history-action-type">${this.getActionIcon(action.action_type)} ${action.action_name}</span>
                    <span class="history-timestamp">${this.formatTimestamp(action.timestamp)}</span>
                </div>
                <div class="history-item-content">
                    <div class="history-symbol">${action.symbol || 'N/A'}</div>
                    <div class="history-details">${action.details}</div>
                    ${action.bot_id ? `<div class="history-bot-id">Bot ID: ${action.bot_id}</div>` : ''}
                </div>
            </div>
        `).join('');
        
        container.innerHTML = html;
    }

    /**
     * Отображает сделки ботов
     */,
            displayBotTrades(trades) {
        const container = document.getElementById('botTradesList');
        if (!container) return;
        
        if (trades.length === 0) {
            container.innerHTML = `
                <div class="empty-history-state">
                    <div class="empty-icon">💼</div>
                    <p data-translate="no_trades_found">История сделок не найдена</p>
                    <p data-translate="trades_will_appear">Сделки ботов будут отображаться здесь</p>
                </div>
            `;
            return;
        }
        
        const html = trades.map(trade => {
            // Определяем индикатор источника решения
            const decisionSource = trade.decision_source || 'SCRIPT';
            const aiIndicator = decisionSource === 'AI' 
                ? `<span class="ai-indicator" title="AI решение${trade.ai_confidence ? ` (уверенность: ${(trade.ai_confidence * 100).toFixed(0)}%)` : ''}">🤖 AI</span>`
                : `<span class="script-indicator" title="Скриптовое правило">📜 SCRIPT</span>`;
            
            const resultIndicator = trade.is_successful !== undefined 
                ? (trade.is_successful ? '<span class="result-indicator success" title="Успешная сделка">✅</span>' : '<span class="result-indicator failed" title="Неуспешная сделка">❌</span>')
                : '';
            
            return `
            <div class="history-item trade-item ${trade.status === 'CLOSED' ? 'closed' : 'open'} ${decisionSource.toLowerCase()}">
                <div class="history-item-header">
                    <span class="history-trade-direction ${trade.direction.toLowerCase()}">${trade.direction}</span>
                    ${aiIndicator}
                    ${resultIndicator}
                    <span class="history-timestamp">${this.formatTimestamp(trade.timestamp)}</span>
                </div>
                <div class="history-item-content">
                    <div class="history-symbol">${trade.symbol}</div>
                    <div class="trade-details">
                        <div class="trade-price">Вход: ${trade.entry_price?.toFixed(4) || 'N/A'}</div>
                        ${trade.exit_price ? `<div class="trade-price">Выход: ${trade.exit_price.toFixed(4)}</div>` : ''}
                        <div class="trade-size">Размер: ${trade.size}</div>
                        ${trade.pnl !== null ? `<div class="trade-pnl ${trade.pnl >= 0 ? 'profit' : 'loss'}">PnL: ${trade.pnl.toFixed(2)} USDT</div>` : ''}
                        ${trade.roi !== null ? `<div class="trade-roi ${trade.roi >= 0 ? 'profit' : 'loss'}">ROI: ${trade.roi.toFixed(2)}%</div>` : ''}
                        ${trade.ai_confidence ? `<div class="ai-confidence">AI уверенность: ${(trade.ai_confidence * 100).toFixed(0)}%</div>` : ''}
                    </div>
                    <div class="trade-status">Статус: ${trade.status === 'OPEN' ? 'Открыта' : 'Закрыта'}</div>
                </div>
            </div>
        `;
        }).join('');
        
        container.innerHTML = html;
    }
    /**
     * Отображает сигналы ботов
     */,
            displayBotSignals(signals) {
        const container = document.getElementById('botSignalsList');
        if (!container) return;
        
        if (signals.length === 0) {
            container.innerHTML = `
                <div class="empty-history-state">
                    <div class="empty-icon">⚡</div>
                    <p data-translate="no_signals_found">История сигналов не найдена</p>
                    <p data-translate="signals_will_appear">Сигналы ботов будут отображаться здесь</p>
                </div>
            `;
            return;
        }
        
        const html = signals.map(signal => `
            <div class="history-item signal-item">
                <div class="history-item-header">
                    <span class="history-signal-type">⚡ ${signal.signal_type || 'SIGNAL'}</span>
                    <span class="history-timestamp">${this.formatTimestamp(signal.timestamp)}</span>
                </div>
                <div class="history-item-content">
                    <div class="history-symbol">${signal.symbol}</div>
                    <div class="signal-details">
                        <div class="signal-rsi">RSI: ${signal.rsi?.toFixed(2) || 'N/A'}</div>
                        <div class="signal-price">Цена: ${signal.price?.toFixed(4) || 'N/A'}</div>
                    </div>
                    <div class="signal-description">${signal.details}</div>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = html;
    }

    /**
     * Отображает статистику истории
     */,
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
        const botFilter = document.getElementById('historyBotFilter');
        if (!botFilter) {
            return;
        }

        const uniqueSymbols = Array.from(new Set(symbols.filter(Boolean))).sort();
        this.historyBotSymbols = uniqueSymbols;

        const currentValue = botFilter.value;

        const allBotsLabel = typeof this.getTranslation === 'function'
            ? this.getTranslation('all_bots')
            : 'Все боты';

        const options = [
            `<option value="all" data-translate="all_bots">${allBotsLabel}</option>`
        ];

        uniqueSymbols.forEach(symbol => {
            options.push(`<option value="${symbol}">${symbol}</option>`);
        });

        botFilter.innerHTML = options.join('');

        if (uniqueSymbols.includes(currentValue)) {
            botFilter.value = currentValue;
        } else {
            botFilter.value = 'all';
        }
    }

    /**
     * Очищает фильтры истории
     */,
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
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Демо-данные созданы успешно', 'success');
                this.loadHistoryData();
            } else {
                throw new Error(data.error || 'Ошибка создания демо-данных');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка создания демо-данных:', error);
            this.showNotification(`Ошибка создания демо-данных: ${error.message}`, 'error');
        }
    }

    /**
     * Очищает всю историю
     */,
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
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('История очищена', 'success');
                this.loadHistoryData();
            } else {
                throw new Error(data.error || 'Ошибка очистки истории');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка очистки истории:', error);
            this.showNotification(`Ошибка очистки истории: ${error.message}`, 'error');
        }
    }

    /**
     * Получает иконку для типа действия
     */,
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
            formatDuration(seconds) {
        if (seconds === undefined || seconds === null) {
            return '—';
        }
        const totalSeconds = Math.max(0, Number(seconds));
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const secs = Math.floor(totalSeconds % 60);
        const parts = [];
        if (hours) parts.push(`${hours}ч`);
        if (minutes) parts.push(`${minutes}м`);
        if (!hours && !minutes) parts.push(`${secs}с`);
        else if (secs) parts.push(`${secs}с`);
        return parts.join(' ');
    },
            saveCollapseState(symbol, isCollapsed) {
        // Сохраняем состояние сворачивания для конкретного бота
        if (!this.collapseStates) {
            this.collapseStates = {};
        }
        
        this.collapseStates[symbol] = {
            isCollapsed: isCollapsed,
            buttonText: isCollapsed ? '▲' : '▼'
        };
        
        console.log(`[DEBUG] Сохранено состояние для ${symbol}:`, this.collapseStates[symbol]);
        console.log(`[DEBUG] Все сохраненные состояния:`, this.collapseStates);
    },
            preserveCollapseState(container) {
        // Восстанавливаем сохраненное состояние сворачивания для каждого бота
        if (!this.collapseStates) {
            this.collapseStates = {};
        }
        
        console.log(`[DEBUG] Восстанавливаем состояние для контейнера:`, container.id);
        console.log(`[DEBUG] Доступные состояния:`, this.collapseStates);
        
        container.querySelectorAll('.active-bot-item').forEach(item => {
            const symbol = item.dataset.symbol;
            const details = item.querySelector('.bot-details');
            const collapseBtn = item.querySelector('.collapse-btn');
            
            console.log(`[DEBUG] Обрабатываем бота ${symbol}:`, {
                hasDetails: !!details,
                hasCollapseBtn: !!collapseBtn,
                hasState: !!this.collapseStates[symbol],
                currentDisplay: details ? details.style.display : 'N/A'
            });
            
            if (details && collapseBtn && this.collapseStates[symbol]) {
                const state = this.collapseStates[symbol];
                console.log(`[DEBUG] Восстанавливаем состояние для ${symbol}:`, state);
                
                if (state.isCollapsed) {
                    // Блок должен быть свернут
                    details.style.display = 'none';
                    collapseBtn.textContent = '▼';
                    console.log(`[DEBUG] ${symbol}: СВЕРНУТ (display: none, кнопка: ▼)`);
                } else {
                    // Блок должен быть развернут
                    // Определяем правильный display на основе контейнера
                    const isDetailsElement = container.id === 'activeBotsDetailsList';
                    const displayValue = isDetailsElement ? 'grid' : 'block';
                    details.style.display = displayValue;
                    collapseBtn.textContent = '▲';
                    console.log(`[DEBUG] ${symbol}: РАЗВЕРНУТ (display: ${displayValue}, кнопка: ▲, контейнер: ${container.id})`);
                }
            } else {
                console.log(`[DEBUG] ${symbol}: НЕ ВОССТАНАВЛИВАЕМ - отсутствуют элементы или состояние`);
            }
        });
    }
    });
})();
