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
            }

            if (bot.status === 'in_position_long' || bot.status === 'in_position_short') {
                inPositionCount += 1;
            }

            this.logDebug(`[BotsManager] 📊 Бот ${bot.symbol}: PnL=$${botPnL.toFixed(3)}, Статус=${bot.status}`);
        });

        const totalBotsElement = document.getElementById('totalBotsCount');
        if (totalBotsElement) {
            totalBotsElement.textContent = bots.length;
        } else {
            this.logDebug('[BotsManager] ⚠️ Элемент totalBotsCount не найден');
        }

        const totalPnLElement = document.getElementById('totalPnLValue');
        const headerPnLElement = document.getElementById('totalBotsePnL');
        const positiveColor = 'var(--green-color, #4caf50)';
        const negativeColor = 'var(--red-color, #f44336)';
        const formattedPnL = `$${totalPnL.toFixed(3)}`;

        if (totalPnLElement) {
            totalPnLElement.textContent = formattedPnL;
            totalPnLElement.style.color = totalPnL >= 0 ? positiveColor : negativeColor;
            this.logDebug(`[BotsManager] 📊 Обновлен элемент totalPnLValue: ${formattedPnL}`);
        } else {
            console.warn('[BotsManager] ⚠️ Элемент totalPnLValue не найден!');
        }

        if (headerPnLElement) {
            headerPnLElement.textContent = formattedPnL;
            headerPnLElement.style.color = totalPnL >= 0 ? positiveColor : negativeColor;
        } else {
            this.logDebug('[BotsManager] ⚠️ Элемент totalBotsePnL не найден');
        }

        this.logDebug(`[BotsManager] 📊 Статистика обновлена: всего=${bots.length}, активных=${activeCount}, в позиции=${inPositionCount}, PnL=${formattedPnL}`);
    },
            stopPeriodicUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        if (this.accountUpdateInterval) {
            clearInterval(this.accountUpdateInterval);
            this.accountUpdateInterval = null;
        }
        this.stopBotMonitoring();
        this.logDebug('[BotsManager] ⏹️ Периодическое обновление остановлено');
    },
            restartPeriodicUpdate() {
        this.stopPeriodicUpdate();
        this.startPeriodicUpdate();
        console.log('[BotsManager] 🔄 Периодическое обновление перезапущено (интервал: ' + (this.refreshInterval/1000) + ' сек)');
    },
            startPeriodicUpdate() {
        this.stopPeriodicUpdate();
        // Обновляем данные с интервалом = position_sync_interval (список монет слева, фильтры, боты, мониторинг)
        this.updateInterval = setInterval(() => {
            if (this.serviceOnline) {
                this.logDebug('[BotsManager] 🔄 Автообновление данных...');
                
                // Обновляем основные данные (список монет, фильтры, конфиг, боты)
                this.loadCoinsRsiData();
                this.loadFiltersData();      // Whitelist, blacklist, scope
                // НЕ загружаем конфиг при активной вкладке «Конфигурация» — иначе перезаписываем форму и теряются несохранённые изменения
                const configTab = document.getElementById('configTab');
                if (!configTab || !configTab.classList.contains('active')) {
                    this.loadConfigurationData();
                }
                this.loadDelistedCoins();    // Делистинговые монеты
                this.loadAccountInfo();
                
                // КРИТИЧЕСКИ ВАЖНО: Всегда обновляем состояние автобота и ботов
                this.loadActiveBotsData();
        } else {
                this.checkBotsService();
            }
        }, this.refreshInterval);
        
        // Отдельный интервал для данных аккаунта (баланс, PnL) — не чаще 10 сек, иначе мигает интерфейс
        const accountIntervalMs = Math.max(10000, this.refreshInterval);
        this.accountUpdateInterval = setInterval(() => {
            if (this.serviceOnline) {
                this.logDebug('[BotsManager] 💰 Обновление данных аккаунта...');
                this.loadAccountInfo();
            }
        }, accountIntervalMs);
        
        console.log(`[BotsManager] ⏰ Запущено периодическое обновление (${this.refreshInterval/1000} сек)`);
        
        // Запускаем мониторинг активных ботов с тем же интервалом
        this.startBotMonitoring();
    },
            startBotMonitoring() {
        console.log('[BotsManager] 📊 Запуск мониторинга активных ботов...');
        
        // Останавливаем предыдущий таймер если есть
        if (this.monitoringTimer) {
            clearInterval(this.monitoringTimer);
        }
        
        // Запускаем мониторинг с единым интервалом
        this.monitoringTimer = setInterval(() => {
            this.updateActiveBotsDetailed();
        }, this.refreshInterval);
        
        console.log(`[BotsManager] ✅ Мониторинг активных ботов запущен (интервал: ${this.refreshInterval}мс)`);
    },
            stopBotMonitoring() {
        if (this.monitoringTimer) {
            clearInterval(this.monitoringTimer);
            this.monitoringTimer = null;
            console.log('[BotsManager] ⏹️ Мониторинг активных ботов остановлен');
        }
    },
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
        
        const pnlElement = botElement.querySelector('.bot-pnl');
        if (pnlElement) {
            pnlElement.textContent = `PnL: $${pnl.toFixed(2)}`;
            pnlElement.style.color = pnl >= 0 ? 'var(--green-color)' : 'var(--red-color)';
        }
        
        const priceElement = botElement.querySelector('.bot-price');
        if (priceElement && bot.current_price) {
            priceElement.textContent = `$${bot.current_price.toFixed(6)}`;
        }
        
        const directionElement = botElement.querySelector('.bot-direction');
        if (directionElement) {
            if (bot.position_side === 'Long') {
                directionElement.textContent = '📈 LONG';
                directionElement.style.color = 'var(--green-color)';
            } else if (bot.position_side === 'Short') {
                directionElement.textContent = '📉 SHORT';
                directionElement.style.color = 'var(--red-color)';
            } else {
                directionElement.textContent = '⏸️ НЕТ';
                directionElement.style.color = 'var(--gray-color)';
            }
        }
        
        const trailingElement = botElement.querySelector('.bot-trailing');
        if (trailingElement) {
            if (bot.trailing_stop_active) {
                trailingElement.textContent = '🎯 Трейлинг активен';
                trailingElement.style.color = 'var(--orange-color)';
            } else {
                trailingElement.textContent = '⏸️ Трейлинг неактивен';
                trailingElement.style.color = 'var(--gray-color)';
            }
        }
        
        // Обновляем потенциальный убыток по стоп-лоссу
        const stopLossElement = botElement.querySelector('.bot-stop-loss');
        if (stopLossElement && bot.stop_loss_price) {
            const stopLossPnL = bot.stop_loss_pnl || 0;
            stopLossElement.textContent = `Стоп: $${stopLossPnL.toFixed(2)}`;
            stopLossElement.style.color = 'var(--red-color)';
        }
        
        // Обновляем оставшееся время позиции
        const timeElement = botElement.querySelector('.bot-time-left');
        if (timeElement && bot.position_start_time && bot.max_position_hours > 0) {
            const timeLeft = this.calculateTimeLeft(bot.position_start_time, bot.max_position_hours, true);
            timeElement.textContent = `${this.getTranslation('time_label')} ${timeLeft}`;
            timeElement.style.color = timeLeft.includes('0:00') ? 'var(--red-color)' : 'var(--blue-color)';
        } else if (timeElement) {
            timeElement.textContent = `${this.getTranslation('time_label')} ∞`;
            timeElement.style.color = 'var(--gray-color)';
        }
    },
            calculateTimeLeft(startTime, maxHours, maxHoursIsHours = true) {
        const start = new Date(startTime);
        const now = new Date();
        const elapsed = now - start;
        const maxMs = (maxHoursIsHours ? maxHours * 3600 : maxHours) * 1000;
        const remaining = maxMs - elapsed;
        
        if (remaining <= 0) {
            return '0:00';
        }
        
        const hours = Math.floor(remaining / (60 * 60 * 1000));
        const minutes = Math.floor((remaining % (60 * 60 * 1000)) / (60 * 1000));
        
        return `${hours}:${minutes.toString().padStart(2, '0')}`;
    },
            destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        
        if (this.accountUpdateInterval) {
            clearInterval(this.accountUpdateInterval);
            this.accountUpdateInterval = null;
        }
        
        if (this.monitoringTimer) {
            clearInterval(this.monitoringTimer);
            this.monitoringTimer = null;
        }
        
        console.log('[BotsManager] 🛑 Менеджер ботов уничтожен');
    }
    });
})();
