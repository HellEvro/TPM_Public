/**
 * BotsManager - 07_bot_controls
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            async createBot(manualDirection = null) {
        console.log('[BotsManager] 🚀 Запуск создания бота...');
        
        if (!this.selectedCoin) {
            console.log('[BotsManager] ❌ Нет выбранной монеты!');
            this.showNotification('⚠️ ' + this.translate('select_coin_to_create_bot'), 'warning');
            return null;
        }
        
        console.log(`[BotsManager] 🤖 Создание бота для ${this.selectedCoin.symbol}`);
        const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
        const rsiKey = `rsi${currentTimeframe}`;
        const rsiValue = this.selectedCoin[rsiKey] || this.selectedCoin.rsi6h || this.selectedCoin.rsi || 'неизвестно';
        console.log(`[BotsManager] 📊 RSI текущий (${currentTimeframe}): ${rsiValue}`);
        
        // Показываем уведомление о начале процесса
        this.showNotification(`🔄 ${this.translate('creating_bot_for')} ${this.selectedCoin.symbol}...`, 'info');
        
        try {
            const config = {
                volume_mode: document.getElementById('volumeModeSelect')?.value || 'usdt',
                volume_value: parseFloat(document.getElementById('volumeValueInput')?.value || '10'),
                leverage: parseInt(document.getElementById('leverageCoinInput')?.value || '10')
            };
            
            console.log('[BotsManager] 📊 Параметры запуска бота (overrides):', config);
            console.log('[BotsManager] 🌐 Отправка запроса на создание бота...');
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: this.selectedCoin.symbol,
                    config: config,
                    signal: manualDirection ? (manualDirection === 'SHORT' ? 'ENTER_SHORT' : 'ENTER_LONG') : (this.selectedCoin.signal || 'ENTER_LONG'),
                    skip_maturity_check: true,
                    force_manual_entry: true
                })
            });
            
            console.log(`[BotsManager] 📡 Ответ сервера: статус ${response.status}`);
            const data = await response.json();
            console.log('[BotsManager] 📥 Данные ответа:', data);
            
            if (data.success) {
                console.log('[BotsManager] ✅ Бот создан успешно:', data);
                console.log(`[BotsManager] 🎯 ID бота: ${data.bot?.id || 'неизвестно'}`);
                console.log(`[BotsManager] 📈 Статус бота: ${data.bot?.status || 'неизвестно'}`);
                
                this.showNotification(`✅ Бот для ${this.selectedCoin.symbol} запущен и работает!`, 'success');
                
                // Логируем процесс обновления UI
                console.log('[BotsManager] 🔄 Обновление интерфейса...');
                
                // Добавляем созданного бота в локальный массив для немедленного обновления UI
                const newBot = {
                    symbol: this.selectedCoin.symbol,
                    status: data.bot?.status || 'running',
                    volume_mode: data.bot?.volume_mode || 'usdt',
                    volume_value: data.bot?.volume_value || 10,
                    created_at: data.bot?.created_at || new Date().toISOString(),
                    unrealized_pnl: data.bot?.unrealized_pnl || 0,
                    entry_price: data.bot?.entry_price || null,
                    position_side: data.bot?.position_side || null,
                    rsi_data: this.selectedCoin
                };
                
                // Обновляем локальный массив
                if (!this.activeBots) this.activeBots = [];
                const existingIndex = this.activeBots.findIndex(bot => bot.symbol === this.selectedCoin.symbol);
                if (existingIndex >= 0) {
                    this.activeBots[existingIndex] = newBot;
                } else {
                    this.activeBots.push(newBot);
                }
                
                // Обновляем статус выбранной монеты
                console.log('[BotsManager] 🎯 Обновление статуса бота...');
                this.updateBotStatus();
                
                // Обновляем кнопки управления
                console.log('[BotsManager] 🎮 Обновление кнопок управления...');
                this.updateBotControlButtons();
                
                // Обновляем данные активных ботов
                console.log('[BotsManager] 📊 Загрузка списка активных ботов...');
                await this.loadActiveBotsData();
                
                // Обновляем список монет с пометками о ботах
                this.logDebug('[BotsManager] 💰 Обновление списка монет с пометками...');
                this.updateCoinsListWithBotStatus();
                
                // Обновляем список на вкладке "Боты в работе"
                console.log('[BotsManager] 🚀 Обновление вкладки "Боты в работе"...');
                this.updateActiveBotsTab();
                
                console.log('[BotsManager] ✅ Все обновления интерфейса завершены!');
                
                const manualButtons = document.getElementById('manualBotButtons');
                if (manualButtons) manualButtons.style.display = 'none';
                const longBtn = document.getElementById('enableBotLongBtn');
                const shortBtn = document.getElementById('enableBotShortBtn');
                if (longBtn) longBtn.style.display = 'none';
                if (shortBtn) shortBtn.style.display = 'none';
                
            } else {
                console.error('[BotsManager] ❌ Ошибка создания бота:', data.error);
                this.showNotification(`❌ Ошибка создания бота: ${data.error}`, 'error');
            }
            
            return data;
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка создания бота:', error);
            this.showNotification('❌ ' + this.translate('connection_error_bot_service'), 'error');
            return null;
        }
    },
            async startBot(symbol) {
        const targetSymbol = symbol || this.selectedCoin?.symbol;
        if (!targetSymbol) {
            this.showNotification('⚠️ Выберите монету для запуска бота', 'warning');
            return;
        }
        
        console.log(`[BotsManager] ▶️ Запуск бота для ${targetSymbol}`);
        this.showNotification(`🔄 Запуск бота ${targetSymbol}...`, 'info');

        // Немедленно обновляем UI
        this.updateBotStatusInUI(targetSymbol, 'starting');

        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: targetSymbol })
            });

            const data = await response.json();
            if (data.success) {
                this.showNotification(`✅ Бот ${targetSymbol} запущен`, 'success');
                // Обновляем UI после успешного запуска
                this.updateBotStatusInUI(targetSymbol, 'active');
                
                // Обновляем локальные данные бота
                if (this.activeBots) {
                    const botIndex = this.activeBots.findIndex(bot => bot.symbol === targetSymbol);
                    if (botIndex >= 0) {
                        this.activeBots[botIndex].status = 'running';
                    }
                }
                
                // Обновляем все элементы интерфейса
                await this.loadActiveBotsData();
                this.updateBotControlButtons();
                this.updateBotStatus();
                this.updateCoinsListWithBotStatus();
                this.renderActiveBotsDetails();
            } else {
                this.showNotification(`❌ Ошибка запуска бота: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка запуска бота:', error);
            this.showNotification('❌ ' + this.translate('connection_error_bot_service'), 'error');
        }
    },
            async stopBot(symbol) {
        const targetSymbol = symbol || this.selectedCoin?.symbol;
        if (!targetSymbol) {
            this.showNotification('⚠️ Выберите монету для остановки бота', 'warning');
            return;
        }
        
        console.log(`[BotsManager] ⏹️ Остановка бота для ${targetSymbol}`);
        this.showNotification(`🔄 Остановка бота ${targetSymbol}...`, 'info');

        // Немедленно обновляем UI
        this.updateBotStatusInUI(targetSymbol, 'stopping');

        try {
            // Добавляем таймаут для запроса
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 секунд таймаут
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: targetSymbol }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            const data = await response.json();
            if (data.success) {
                this.showNotification(`✅ Бот ${targetSymbol} остановлен`, 'success');
                // Обновляем UI после успешной остановки - используем 'paused' вместо 'stopped'
                this.updateBotStatusInUI(targetSymbol, 'paused');
                
                // Обновляем локальные данные бота
                if (this.activeBots) {
                    const botIndex = this.activeBots.findIndex(bot => bot.symbol === targetSymbol);
                    if (botIndex >= 0) {
                        this.activeBots[botIndex].status = 'paused';
                    }
                }
                
                // Обновляем все элементы интерфейса
                await this.loadActiveBotsData();
                this.updateBotControlButtons();
                this.updateBotStatus();
                this.updateCoinsListWithBotStatus();
                this.renderActiveBotsDetails();
            } else {
                this.showNotification(`❌ Ошибка остановки бота: ${data.error}`, 'error');
                // Возвращаем UI в исходное состояние при ошибке
                this.updateBotStatusInUI(targetSymbol, 'active');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка остановки бота:', error);
            
            if (error.name === 'AbortError') {
                this.showNotification('⏰ Таймаут операции остановки бота', 'error');
            } else {
                this.showNotification('❌ ' + this.translate('connection_error_bot_service'), 'error');
            }
            
            // Возвращаем UI в исходное состояние при ошибке
            this.updateBotStatusInUI(targetSymbol, 'active');
        }
    },
            async pauseBot(symbol) {
        const targetSymbol = symbol || this.selectedCoin?.symbol;
        if (!targetSymbol) {
            this.showNotification('⚠️ Выберите монету для паузы бота', 'warning');
            return;
        }
        
        console.log(`[BotsManager] ⏸️ Пауза бота для ${targetSymbol}`);
        this.showNotification(`🔄 Пауза бота ${targetSymbol}...`, 'info');

        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/pause`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: targetSymbol })
            });

            const data = await response.json();
            if (data.success) {
                this.showNotification(`✅ Бот ${targetSymbol} поставлен на паузу`, 'success');
                await this.loadActiveBotsData();
                this.updateBotControlButtons();
            } else {
                this.showNotification(`❌ Ошибка паузы бота: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка паузы бота:', error);
            this.showNotification('❌ ' + this.translate('connection_error_bot_service'), 'error');
        }
    },
            async resumeBot(symbol) {
        const targetSymbol = symbol || this.selectedCoin?.symbol;
        if (!targetSymbol) {
            this.showNotification('⚠️ Выберите монету для возобновления бота', 'warning');
            return;
        }
        
        console.log(`[BotsManager] ⏯️ Возобновление бота для ${targetSymbol}`);
        this.showNotification(`🔄 Возобновление бота ${targetSymbol}...`, 'info');

        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/resume`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: targetSymbol })
            });

            const data = await response.json();
            if (data.success) {
                this.showNotification(`✅ Бот ${targetSymbol} возобновлен`, 'success');
                await this.loadActiveBotsData();
                this.updateBotControlButtons();
            } else {
                this.showNotification(`❌ Ошибка возобновления бота: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка возобновления бота:', error);
            this.showNotification('❌ ' + this.translate('connection_error_bot_service'), 'error');
        }
    },
            async deleteBot(symbol) {
        const targetSymbol = symbol || this.selectedCoin?.symbol;
        if (!targetSymbol) {
            this.showNotification('⚠️ Выберите монету для удаления бота', 'warning');
            return;
        }
        
        console.log(`[BotsManager] 🗑️ Удаление бота для ${targetSymbol}`);
        this.showNotification(`🔄 Удаление бота ${targetSymbol}...`, 'info');

        // Немедленно обновляем UI
        this.updateBotStatusInUI(targetSymbol, 'deleting');

        try {
            // Добавляем таймаут для запроса
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 секунд таймаут
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: targetSymbol }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            const data = await response.json();
            if (data.success) {
                this.showNotification(`✅ Бот ${targetSymbol} удален`, 'success');
                // Обновляем UI после успешного удаления
                this.removeBotFromUI(targetSymbol);
                await this.loadActiveBotsData();
                this.updateBotControlButtons();
                this.updateCoinsListWithBotStatus();
            } else {
                this.showNotification(`❌ Ошибка удаления бота: ${data.error}`, 'error');
                // Возвращаем UI в исходное состояние при ошибке
                this.updateBotStatusInUI(targetSymbol, 'active');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка удаления бота:', error);
            
            if (error.name === 'AbortError') {
                this.showNotification('⏰ Таймаут операции удаления бота', 'error');
            } else {
                this.showNotification('❌ ' + this.translate('connection_error_bot_service'), 'error');
            }
            
            // Возвращаем UI в исходное состояние при ошибке
            this.updateBotStatusInUI(targetSymbol, 'active');
        }
    },
            async quickLaunchBot(direction) {
        if (!this.selectedCoin) return;
        
        try {
            console.log(`[BotsManager] 🚀 Быстрый запуск ${direction} бота для ${this.selectedCoin.symbol}`);
            await this.createBot(direction);
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка быстрого запуска ${direction} бота:`, error);
            this.showNotification('❌ Ошибка соединения при создании бота', 'error');
        }
    },
            updateBotStatusInUI(symbol, status) {
        const botCard = document.querySelector(`[data-symbol="${symbol}"]`);
        if (!botCard) return;

        const statusElement = botCard.querySelector('.bot-status');
        const startButton = botCard.querySelector('.start-bot-btn');
        const stopButton = botCard.querySelector('.stop-bot-btn');
        const deleteButton = botCard.querySelector('.delete-bot-btn');

        if (statusElement) {
            switch (status) {
                case 'starting':
                    statusElement.textContent = window.languageUtils.translate('bot_status_starting');
                    statusElement.className = 'bot-status status-starting';
                    if (startButton) startButton.disabled = true;
                    if (stopButton) stopButton.disabled = true;
                    break;
                case 'active':
                    statusElement.textContent = window.languageUtils.translate('active_status');
                    statusElement.className = 'bot-status status-active';
                    if (startButton) startButton.disabled = true;
                    if (stopButton) stopButton.disabled = false;
                    break;
                case 'stopping':
                    statusElement.textContent = 'Остановка...';
                    statusElement.className = 'bot-status status-stopping';
                    if (startButton) startButton.disabled = true;
                    if (stopButton) stopButton.disabled = true;
                    break;
                case 'idle':
                    statusElement.textContent = window.languageUtils.translate('waiting_status');
                    statusElement.className = 'bot-status status-idle';
                    if (startButton) startButton.disabled = false;
                    if (stopButton) stopButton.disabled = true;
                    break;
                case 'stopped':
                    statusElement.textContent = window.languageUtils.translate('stopped_status');
                    statusElement.className = 'bot-status status-stopped';
                    if (startButton) startButton.disabled = false;
                    if (stopButton) stopButton.disabled = true;
                    break;
                case 'paused':
                    statusElement.textContent = 'На паузе';
                    statusElement.className = 'bot-status status-paused';
                    if (startButton) startButton.disabled = false;
                    if (stopButton) stopButton.disabled = true;
                    break;
                case 'deleting':
                    statusElement.textContent = 'Удаление...';
                    statusElement.className = 'bot-status status-deleting';
                    if (startButton) startButton.disabled = true;
                    if (stopButton) stopButton.disabled = true;
                    if (deleteButton) deleteButton.disabled = true;
                    break;
            }
        }
    },
            removeBotFromUI(symbol) {
        const botCard = document.querySelector(`[data-symbol="${symbol}"]`);
        if (botCard) {
            botCard.remove();
        }
    },
            getBotStopButtonHtml(bot) {
        const isRunning = bot.status === 'running' || bot.status === 'idle' || 
                         bot.status === 'in_position_long' || bot.status === 'in_position_short';
        const isStopped = bot.status === 'stopped' || bot.status === 'paused';
        if (isRunning) {
            return `<span onclick="event.stopPropagation(); window.app.botsManager.stopBot('${bot.symbol}')" title="${window.languageUtils.translate('stop_btn')}" class="bot-icon-btn bot-icon-stop">&#x2298;</span>`;
        }
        if (isStopped) {
            return `<span onclick="event.stopPropagation(); window.app.botsManager.startBot('${bot.symbol}')" title="${window.languageUtils.translate('start_btn') || 'Старт'}" class="bot-icon-btn bot-icon-start">&#x25B6;</span>`;
        }
        return '';
    },
            getBotDeleteButtonHtml(bot) {
        return `<span onclick="event.stopPropagation(); window.app.botsManager.deleteBot('${bot.symbol}')" title="${window.languageUtils.translate('delete_btn')}" class="bot-icon-btn bot-icon-delete">🗑</span>`;
    },
            getBotControlButtonsHtml(bot) {
        return (this.getBotStopButtonHtml(bot) || '') + this.getBotDeleteButtonHtml(bot);
    },
            getBotDetailButtonsHtml(bot) {
        // Бот активен если running, idle, или в позиции
        const isRunning = bot.status === 'running' || bot.status === 'idle' || 
                         bot.status === 'in_position_long' || bot.status === 'in_position_short';
        const isStopped = bot.status === 'stopped' || bot.status === 'paused';
        
        let buttons = [];
        
        if (isRunning) {
            buttons.push(`<button onclick="window.app.botsManager.stopBot('${bot.symbol}')" title="${window.languageUtils.translate('stop_btn')}" style="padding: 5px 10px; background: #f44336; border: none; border-radius: 4px; color: white; cursor: pointer; font-size: 14px;">&#x2298;</button>`);
        } else if (isStopped) {
            buttons.push(`<button onclick="window.app.botsManager.startBot('${bot.symbol}')" title="${window.languageUtils.translate('start_btn') || 'Старт'}" style="padding: 5px 10px; background: #4caf50; border: none; border-radius: 4px; color: white; cursor: pointer; font-size: 14px;">&#x25B6;</button>`);
        }
        buttons.push(`<button onclick="window.app.botsManager.deleteBot('${bot.symbol}')" title="${window.languageUtils.translate('delete_btn')}" style="padding: 5px 10px; background: #9e9e9e; border: none; border-radius: 4px; color: white; cursor: pointer; font-size: 14px;">🗑</button>`);
        
        return buttons.join('');
    },
            updateBotStatus(status) {
        const statusText = document.getElementById('botStatusText');
        const statusIndicator = document.getElementById('botStatusIndicator');
        
        // Проверяем есть ли бот для выбранной монеты
        const selectedBot = this.selectedCoin && this.activeBots ? 
                           this.activeBots.find(bot => bot.symbol === this.selectedCoin.symbol) : null;
        
        if (statusText) {
            if (selectedBot) {
                switch(selectedBot.status) {
                    case 'idle':
                        statusText.textContent = window.languageUtils.translate('waiting_status') || 'Бот создан (ожидает)';
                        break;
                    case 'running':
                        statusText.textContent = window.languageUtils.translate('active_status');
                        break;
                    case 'in_position_long':
                        statusText.textContent = window.languageUtils.translate('active_status') + ' (LONG)';
                        break;
                    case 'in_position_short':
                        statusText.textContent = window.languageUtils.translate('active_status') + ' (SHORT)';
                        break;
                    case 'stopped':
                        statusText.textContent = window.languageUtils.translate('bot_stopped_desc');
                        break;
                    case 'paused':
                        statusText.textContent = window.languageUtils.translate('paused_status');
                        break;
                    default:
                        statusText.textContent = window.languageUtils.translate('bot_created');
                }
            } else {
                statusText.textContent = window.languageUtils.translate('bot_not_created');
            }
        }
        
        if (statusIndicator) {
            if (selectedBot) {
                const color = selectedBot.status === 'running' || 
                             selectedBot.status === 'in_position_long' || 
                             selectedBot.status === 'in_position_short' ? '#4caf50' : 
                             selectedBot.status === 'idle' ? '#ffd700' : '#ff5722';
                statusIndicator.style.color = color;
            } else {
                statusIndicator.style.color = '#888';
            }
        }
    },
            updateBotControlButtons() {
        console.log(`[BotsManager] 🎮 Обновление кнопок управления...`);
        
        const createBtn = document.getElementById('createBotBtn');
        const startBtn = document.getElementById('startBotBtn');
        const stopBtn = document.getElementById('stopBotBtn');
        const pauseBtn = document.getElementById('pauseBotBtn');
        const resumeBtn = document.getElementById('resumeBotBtn');
        
        // Кнопки быстрого запуска
        const quickStartLongBtn = document.getElementById('quickStartLongBtn');
        const quickStartShortBtn = document.getElementById('quickStartShortBtn');
        const quickStopBtn = document.getElementById('quickStopBtn');
        
        // Кнопки ручного запуска в секции настроек
        const manualLaunchLongBtn = document.getElementById('manualLaunchLongBtn');
        const manualLaunchShortBtn = document.getElementById('manualLaunchShortBtn');
        
        // Проверяем есть ли бот для выбранной монеты
        const selectedBot = this.selectedCoin && this.activeBots ? 
                           this.activeBots.find(bot => bot.symbol === this.selectedCoin.symbol) : null;
        
        // Проверяем, есть ли активная позиция
        const hasActivePosition = selectedBot && (
            selectedBot.status === 'in_position_long' || 
            selectedBot.status === 'in_position_short' ||
            selectedBot.status === 'running'
        );
        
        console.log(`[BotsManager] 🔍 Выбранная монета: ${this.selectedCoin?.symbol}`);
        console.log(`[BotsManager] 🤖 Найден бот:`, selectedBot);
        console.log(`[BotsManager] 📊 Есть активная позиция:`, hasActivePosition);
        
        if (selectedBot) {
            // Есть бот для выбранной монеты
            const isRunning = selectedBot.status === 'running';
            const isStopped = selectedBot.status === 'idle' || selectedBot.status === 'stopped' || selectedBot.status === 'paused';
            const inPosition = selectedBot.status === 'in_position_long' || selectedBot.status === 'in_position_short';
            
            if (createBtn) createBtn.style.display = 'none';
            
            if (inPosition) {
                // Бот в позиции - показываем только Стоп и Закрыть
                if (startBtn) startBtn.style.display = 'none';
                if (stopBtn) stopBtn.style.display = 'inline-block';
                if (pauseBtn) pauseBtn.style.display = 'none';
                if (resumeBtn) resumeBtn.style.display = 'none';
                
                // Кнопки запуска скрыты
                if (quickStartLongBtn) quickStartLongBtn.style.display = 'none';
                if (quickStartShortBtn) quickStartShortBtn.style.display = 'none';
                if (manualLaunchLongBtn) manualLaunchLongBtn.style.display = 'none';
                if (manualLaunchShortBtn) manualLaunchShortBtn.style.display = 'none';
                if (quickStopBtn) quickStopBtn.style.display = 'none';
            } else if (isRunning) {
                // Бот работает, но не в позиции - показываем Стоп и кнопки запуска
                if (startBtn) startBtn.style.display = 'none';
                if (stopBtn) stopBtn.style.display = 'inline-block';
                if (pauseBtn) pauseBtn.style.display = 'none';
                if (resumeBtn) resumeBtn.style.display = 'none';
                
                // Показываем кнопки быстрого запуска LONG/SHORT
                if (quickStartLongBtn) quickStartLongBtn.style.display = 'inline-block';
                if (quickStartShortBtn) quickStartShortBtn.style.display = 'inline-block';
                if (manualLaunchLongBtn) manualLaunchLongBtn.style.display = 'inline-block';
                if (manualLaunchShortBtn) manualLaunchShortBtn.style.display = 'inline-block';
                if (quickStopBtn) quickStopBtn.style.display = 'none';
            } else if (isStopped) {
                // Бот остановлен - показываем Старт и кнопки запуска
                if (startBtn) startBtn.style.display = 'inline-block';
                if (stopBtn) stopBtn.style.display = 'none';
                if (pauseBtn) pauseBtn.style.display = 'none';
                if (resumeBtn) resumeBtn.style.display = 'none';
                
                // Показываем кнопки быстрого запуска LONG/SHORT
                if (quickStartLongBtn) quickStartLongBtn.style.display = 'inline-block';
                if (quickStartShortBtn) quickStartShortBtn.style.display = 'inline-block';
                if (manualLaunchLongBtn) manualLaunchLongBtn.style.display = 'inline-block';
                if (manualLaunchShortBtn) manualLaunchShortBtn.style.display = 'inline-block';
                if (quickStopBtn) quickStopBtn.style.display = 'none';
            }
            
            console.log(`[BotsManager] 🎮 Статус бота: ${selectedBot.status}, показаны кнопки управления`);
        } else {
            // Нет бота для выбранной монеты - показываем Создать и быстрые кнопки
            if (createBtn) createBtn.style.display = 'inline-block';
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'none';
            if (pauseBtn) pauseBtn.style.display = 'none';
            if (resumeBtn) resumeBtn.style.display = 'none';
            
            // Показываем кнопки быстрого запуска LONG/SHORT
            if (quickStartLongBtn) quickStartLongBtn.style.display = 'inline-block';
            if (quickStartShortBtn) quickStartShortBtn.style.display = 'inline-block';
            if (manualLaunchLongBtn) manualLaunchLongBtn.style.display = 'inline-block';
            if (manualLaunchShortBtn) manualLaunchShortBtn.style.display = 'inline-block';
            if (quickStopBtn) quickStopBtn.style.display = 'none';
            
            console.log(`[BotsManager] 🆕 Нет бота, показаны кнопки создания и быстрого запуска LONG/SHORT`);
        }
    },
            updateCoinsListWithBotStatus() {
        this.logDebug('[BotsManager] 💰 Обновление списка монет с пометками о ботах...');
        
        if (!this.activeBots) return;
        
        // Создаем set с символами только активных ботов (не idle/paused) для быстрого поиска
        const activeBotsSymbols = new Set(
            this.activeBots
                .filter(bot => bot.status !== 'idle' && bot.status !== 'paused')
                .map(bot => bot.symbol)
        );
        
        this.logDebug(`[BotsManager] 🤖 Найдено ${activeBotsSymbols.size} активных ботов из ${this.activeBots.length} общих`);
        
        // Обновляем отображение монет в списке
        const coinItems = document.querySelectorAll('.coin-item');
        coinItems.forEach(item => {
            const symbolElement = item.querySelector('.coin-symbol');
            if (symbolElement) {
                const symbol = symbolElement.textContent.trim();
                
                // Добавляем или убираем индикатор бота
                let botIndicator = item.querySelector('.bot-indicator');
                
                if (activeBotsSymbols.has(symbol)) {
                    // Есть активный бот для этой монеты
                    if (!botIndicator) {
                        botIndicator = document.createElement('span');
                        botIndicator.className = 'bot-indicator';
                        botIndicator.textContent = '🤖';
                        botIndicator.title = 'Активный бот';
                        symbolElement.appendChild(botIndicator);
                    }
                } else {
                    // Нет активного бота
                    if (botIndicator) {
                        botIndicator.remove();
                    }
                }
            }
        });
    },
            updateActiveBotsTab() {
        console.log('[BotsManager] 🚀 Обновление вкладки "Боты в работе"...');
        
        // Если мы сейчас на вкладке "Боты в работе", обновляем данные
        const activeTab = document.querySelector('.tab-btn.active');
        if (activeTab && activeTab.id === 'activeBotsTab') {
            this.renderActiveBotsDetails();
        }
        
        // Обновляем счетчик активных ботов в заголовке вкладки
        const activeBotsTabBtn = document.getElementById('activeBotsTab');
        if (activeBotsTabBtn && this.activeBots) {
            const count = this.activeBots.length;
            const tabText = activeBotsTabBtn.querySelector('[data-translate]');
            if (tabText) {
                // Убираем старый счетчик и добавляем новый
                const baseText = tabText.getAttribute('data-translate') === 'active_bots' ? 'Боты в работе' : 'Active Bots';
                tabText.textContent = count > 0 ? `${baseText} (${count})` : baseText;
            }
        }
    },
                renderActiveBotsDetails() {
        this.logDebug('[BotsManager] 🎨 Отрисовка деталей активных ботов...');
        
        // Обновляем вкладку "Боты в работе"
        const detailsElement = document.getElementById('activeBotsDetailsList');
        
        // Обновляем правую панель на вкладке "Управление"
        const scrollListElement = document.getElementById('activeBotsScrollList');
        const emptyStateElement = document.getElementById('emptyActiveBotsState');
        
        this.logDebug(`[BotsManager] 📊 Количество активных ботов: ${this.activeBots ? this.activeBots.length : 0}`);
        this.logDebug(`[BotsManager] 🔍 Элементы найдены:`, {
            detailsElement: !!detailsElement,
            scrollListElement: !!scrollListElement,
            emptyStateElement: !!emptyStateElement
        });

        const hasActiveBots = this.activeBots && this.activeBots.length > 0;
        
        // Обновляем счётчики фильтров вкладки "Боты в работе"
        this.updateActiveBotsFilterCounts();
        
        // Проверяем, нужно ли полностью перерисовывать HTML
        const existingBots = scrollListElement ? Array.from(scrollListElement.querySelectorAll('.active-bot-item')).map(item => item.dataset.symbol) : [];
        const currentBots = hasActiveBots ? this.activeBots.map(bot => bot.symbol) : [];
        const needsFullRedraw = JSON.stringify(existingBots.sort()) !== JSON.stringify(currentBots.sort());
        const filteredBots = this.getFilteredActiveBotsForDetails();
        const virtualAsBots = this.getVirtualPositionsAsBots();
        const displayListForDetails = filteredBots.concat(virtualAsBots);
        const detailsKey = (b) => b.is_virtual ? `${b.symbol}_v${b._virtualIndex}` : b.symbol;
        const existingDetailsBots = detailsElement ? Array.from(detailsElement.querySelectorAll('.active-bot-item')).map(i => (i.dataset.isVirtual === 'true' ? `${i.dataset.symbol}_v${i.dataset.virtualIndex || 0}` : i.dataset.symbol)).sort() : [];
        const displayKeys = displayListForDetails.map(detailsKey).sort();
        const needsDetailsRedraw = needsFullRedraw || (this.activeBotsFilter !== this._lastActiveBotsFilter) ||
            JSON.stringify(displayKeys) !== JSON.stringify(existingDetailsBots);
        
        console.log(`[DEBUG] Проверка перерисовки:`, { existingBots, currentBots, needsFullRedraw, needsDetailsRedraw });

        // Обновляем правую панель (вкладка "Управление")
        if (emptyStateElement && scrollListElement) {
            if (hasActiveBots) {
                emptyStateElement.style.display = 'none';
                scrollListElement.style.display = 'block';
                
                // Если список ботов изменился - полная перерисовка
                if (needsFullRedraw) {
                    this._lastBotDisplay = {};
                    console.log(`[DEBUG] Полная перерисовка правой панели`);
                    // Отображаем список активных ботов в правой панели
                    const rightPanelHtml = this.activeBots.map(bot => {
                    // Определяем статус бота (активен если running, idle, или в позиции)
                    const isActive = bot.status === 'running' || bot.status === 'idle' || 
                                    bot.status === 'in_position_long' || bot.status === 'in_position_short' ||
                                    bot.status === 'armed_up' || bot.status === 'armed_down';
                    
                    const statusColor = isActive ? '#4caf50' : '#ff5722';
                    const statusText = isActive ? window.languageUtils.translate('active_status') : (bot.status === 'paused' ? window.languageUtils.translate('paused_status') : (bot.status === 'idle' ? window.languageUtils.translate('waiting_status') : window.languageUtils.translate('stopped_status')));
                    
                    // Определяем информацию о позиции
                    console.log(`[DEBUG] renderActiveBotsDetails для ${bot.symbol}:`, {
                        position_side: bot.position_side,
                        entry_price: bot.entry_price,
                        current_price: bot.current_price,
                        rsi_data: bot.rsi_data
                    });
                    
                    const positionInfo = this.getBotPositionInfo(bot);
                    const timeInfo = this.getBotTimeInfo(bot);
                    const htmlResult = `
                        <div class="active-bot-item clickable-bot-item active-bot-sidebar-item" data-symbol="${bot.symbol}" style="border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; margin: 8px 0; background: var(--section-bg); cursor: pointer;" onmouseover="this.style.backgroundColor='var(--hover-bg, var(--button-bg))'" onmouseout="this.style.backgroundColor='var(--section-bg)'">
                            <div class="bot-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="color: var(--text-color); font-weight: bold; font-size: 14px;">${bot.symbol}</span>
                                    <span style="background: ${statusColor}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px;">${statusText}</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <div style="color: ${(bot.unrealized_pnl || bot.unrealized_pnl_usdt || 0) >= 0 ? 'var(--green-color)' : 'var(--red-color)'}; font-weight: bold; font-size: 12px;">$${(bot.unrealized_pnl_usdt || bot.unrealized_pnl || 0).toFixed(3)}</div>
                                    <button class="collapse-btn" onclick="event.stopPropagation(); const details = this.closest('.active-bot-sidebar-item').querySelector('.bot-details'); const isCollapsed = details.style.display === 'none'; details.style.display = isCollapsed ? 'block' : 'none'; this.textContent = isCollapsed ? '▲' : '▼'; window.botsManager && window.botsManager.saveCollapseState(this.closest('.active-bot-sidebar-item').dataset.symbol, !isCollapsed);" style="background: none; border: none; color: var(--text-muted); font-size: 11px; cursor: pointer; padding: 2px;">▼</button>
                                </div>
                            </div>
                            <div class="bot-details" style="font-size: 11px; color: var(--text-color); margin-bottom: 8px; display: none;">
                                <div style="margin-bottom: 4px;">💰 ${this.getTranslation('position_volume')} ${parseFloat(((bot.position_size || 0) * (bot.entry_price || 0)).toFixed(2))} USDT</div>
                                ${positionInfo}
                                ${timeInfo}
                            </div>
                            <div class="bot-controls" style="display: flex; gap: 6px; justify-content: center; flex-wrap: wrap;">
                                ${this.getBotDetailButtonsHtml(bot)}
                            </div>
                        </div>
                    `;
                    
                    console.log(`[DEBUG] Финальный HTML для ${bot.symbol}:`, htmlResult);
                    return htmlResult;
                }).join('');
                
                console.log(`[DEBUG] Вставляем HTML в DOM:`, rightPanelHtml);
                console.log(`[DEBUG] Элемент для вставки:`, scrollListElement);
                
                scrollListElement.innerHTML = rightPanelHtml;
                this.preserveCollapseState(scrollListElement);
                    // Добавляем обработчики кликов для плашек ботов
                    scrollListElement.querySelectorAll('.clickable-bot-item').forEach(item => {
                        item.addEventListener('click', (e) => {
                            // Предотвращаем клик если нажали на кнопку управления
                            if (e.target.closest('.bot-controls button')) {
                return;
            }

                            const symbol = item.dataset.symbol;
                            console.log(`[BotsManager] 🎯 Клик по плашке бота: ${symbol}`);
                            this.selectCoin(symbol);
                        });
                    });
                } else {
                    // Обновляем только данные в существующих карточках
                    console.log(`[DEBUG] Обновление данных в правой панели без перерисовки`);
                    this.activeBots.forEach(bot => {
                        const botItem = scrollListElement.querySelector(`.active-bot-item[data-symbol="${bot.symbol}"]`);
                        if (botItem) {
                            const statusBadge = botItem.querySelector('.bot-header span[style*="background"]');
                            if (statusBadge) {
                                const isActive = bot.status === 'running' || bot.status === 'idle' || bot.status === 'in_position_long' || bot.status === 'in_position_short' || bot.status === 'armed_up' || bot.status === 'armed_down';
                                const statusColor = isActive ? '#4caf50' : '#ff5722';
                                const statusText = isActive ? window.languageUtils.translate('active_status') : (bot.status === 'paused' ? window.languageUtils.translate('paused_status') : (bot.status === 'idle' ? window.languageUtils.translate('waiting_status') : window.languageUtils.translate('stopped_status')));
                                statusBadge.style.background = statusColor;
                                statusBadge.textContent = statusText;
                            }
                            const pnlElement = botItem.querySelector('.bot-header > div:last-child > div:first-child');
                            if (pnlElement) {
                                const pnlValue = (bot.unrealized_pnl_usdt || bot.unrealized_pnl || 0);
                                pnlElement.textContent = `$${pnlValue.toFixed(3)}`;
                                pnlElement.style.color = pnlValue >= 0 ? '#4caf50' : '#f44336';
                            }
                            const controlsDiv = botItem.querySelector('.bot-controls');
                            if (controlsDiv) controlsDiv.innerHTML = this.getBotDetailButtonsHtml(bot);
                            const details = botItem.querySelector('.bot-details');
                            if (details && details.style.display !== 'none') {
                                const posInfo = this.getBotPositionInfo(bot);
                                const tInfo = this.getBotTimeInfo(bot);
                                const volHtml = `💰 ${this.getTranslation('position_volume')} ${parseFloat(((bot.position_size || 0) * (bot.entry_price || 0)).toFixed(2))} USDT`;
                                details.innerHTML = `<div style="margin-bottom: 4px;">${volHtml}</div>${posInfo}${tInfo}`;
                            }
                        }
                    });
                }
            } else {
                emptyStateElement.style.display = 'block';
                scrollListElement.style.display = 'none';
            }
        }

        // Обновляем вкладку "Боты в работе" (реальные боты + виртуальные позиции ПРИИ)
        if (detailsElement) {
            const hasFilteredBots = displayListForDetails.length > 0;
            if (!hasFilteredBots) {
                const currentLang = document.documentElement.lang || 'ru';
                const noActiveBotsText = TRANSLATIONS[currentLang]['no_active_bots'] || 'Нет активных ботов';
                const createBotsText = TRANSLATIONS[currentLang]['create_bots_for_trading'] || 'Создайте ботов для торговли';
                
                detailsElement.innerHTML = `
                    <div class="empty-bots-state" style="text-align: center; padding: 20px; color: #888;">
                        <div style="font-size: 48px; margin-bottom: 10px;">🤖</div>
                        <p style="margin: 10px 0; font-size: 16px;">${noActiveBotsText}</p>
                        <small style="color: #666;">${hasActiveBots ? (window.languageUtils?.translate('active_bots_filter_no_results') || 'Нет ботов по выбранному фильтру') : createBotsText}</small>
                    </div>
                `;
            } else {
                // Если список или фильтр изменился - полная перерисовка
                if (needsDetailsRedraw) {
                    this._lastActiveBotsFilter = this.activeBotsFilter;
                    console.log(`[DEBUG] Полная перерисовка вкладки "Боты в работе"`);
                    
                    const renderBotCard = (bot) => {
                        const isVirtual = !!bot.is_virtual;
                        const isActive = isVirtual || bot.status === 'running' || bot.status === 'idle' ||
                                        bot.status === 'in_position_long' || bot.status === 'in_position_short' ||
                                        bot.status === 'armed_up' || bot.status === 'armed_down';
                        const statusColor = isActive ? '#4caf50' : '#ff5722';
                        const statusText = isVirtual ? (window.languageUtils?.translate('fullai_virtual_position') || 'Виртуальная') : (isActive ? window.languageUtils.translate('active_status') : (bot.status === 'paused' ? window.languageUtils.translate('paused_status') : (bot.status === 'idle' ? window.languageUtils.translate('waiting_status') : window.languageUtils.translate('stopped_status'))));
                        const d = this.getCompactCardData(bot);
                        const t = k => window.languageUtils?.translate(k) || this.getTranslation(k);
                        const exchangeUrl = this.getExchangeLink(bot.symbol, window.app?.exchangeManager?.getSelectedExchange?.() || 'bybit');
                        const pnlValue = isVirtual ? (bot.unrealized_pnl ?? 0) : (bot.unrealized_pnl_usdt ?? bot.unrealized_pnl ?? 0);
                        const isProfit = Number(pnlValue) >= 0;
                        const cardBg = isVirtual ? 'rgba(156, 39, 176, 0.12)' : (isProfit ? 'rgba(76, 175, 80, 0.08)' : 'rgba(244, 67, 54, 0.08)');
                        const virtualAttrs = isVirtual ? ` data-is-virtual="true" data-virtual-index="${bot._virtualIndex || 0}"` : '';
                        const pnlVal = isVirtual ? (bot.unrealized_pnl != null ? `${(bot.unrealized_pnl || 0).toFixed(2)}%` : '-') : `$${(bot.unrealized_pnl_usdt || bot.unrealized_pnl || 0).toFixed(3)}`;
                        return `
                        <div class="active-bot-item clickable-bot-item active-bot-card" data-symbol="${bot.symbol}" data-bot-symbol="${bot.symbol}"${virtualAttrs} data-exchange-url="${exchangeUrl}" data-card-bg="${cardBg.replace(/"/g, '&quot;')}" style="border: 1px solid var(--border-color); border-radius: 10px; padding: 12px; background: ${cardBg}; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" onmouseover="this.style.backgroundColor='var(--hover-bg, var(--button-bg))'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)'" onmouseout="var b=this.dataset.cardBg; this.style.backgroundColor=b||'var(--section-bg)'; this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.1)'">
                            <div class="bot-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color); flex-wrap: wrap; gap: 6px;">
                                <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                                    <span style="color: var(--text-color); font-weight: bold; font-size: 17px;">${bot.symbol}</span>
                                    <span style="background: ${isVirtual ? '#9c27b0' : statusColor}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;">${statusText}</span>
                                    <span class="bot-direction" style="color: ${d.positionColor}; font-weight: 600; font-size: 12px;">${d.position}</span>
                                    <a href="${exchangeUrl}" target="_blank" class="bot-exchange-link" title="Открыть на бирже" onclick="event.stopPropagation();">↗</a>
                                </div>
                                <div style="color: ${(bot.unrealized_pnl != null ? bot.unrealized_pnl : (bot.unrealized_pnl_usdt || 0)) >= 0 ? 'var(--green-color)' : 'var(--red-color)'}; font-weight: bold; font-size: 15px;">${isVirtual ? pnlVal : '$' + (bot.unrealized_pnl_usdt || bot.unrealized_pnl || 0).toFixed(3)}</div>
                            </div>
                            <div class="bot-details bot-details-compact" style="margin-bottom: 8px;">
                                <div class="compact-row"><span class="compact-lbl">${t('position_volume')}</span><span class="compact-val">${d.volume}</span></div>
                                <div class="compact-row"><span class="compact-lbl">${t('entry_label')}</span><span class="compact-val">${d.entry}</span></div>
                                <div class="compact-row"><span class="compact-lbl">${t('take_profit_label_detailed')}</span><span class="compact-val" style="color: var(--green-color)">${d.takeProfit}</span></div>
                                <div class="compact-row"><span class="compact-lbl">${t('current_label')}</span><span class="compact-val" style="color: var(--blue-color)">${d.currentPrice}</span></div>
                                <div class="compact-row"><span class="compact-lbl">${t('stop_loss_label_detailed')}</span><span class="compact-val" style="color: var(--red-color)">${d.stopLoss}</span></div>
                            </div>
                            <div class="bot-card-controls" style="display: flex; gap: 6px; justify-content: flex-end; padding-top: 6px; border-top: 1px solid var(--border-color);">
                                ${isVirtual ? '<span class="text-muted" style="font-size: 11px;">ПРИИ виртуальная обкатка</span>' : this.getBotDetailButtonsHtml(bot)}
                            </div>
                        </div>
                    `;
                    };
                    const realSection = filteredBots.map(bot => renderBotCard(bot)).join('');
                    const virtualSectionHeader = virtualAsBots.length > 0 ? `<div class="virtual-positions-header" style="margin: 16px 0 12px; padding: 8px 12px; background: rgba(156, 39, 176, 0.15); border-radius: 8px; border-left: 4px solid #9c27b0; font-weight: 600; font-size: 13px; color: var(--text-color);">📊 Виртуальные позиции ПРИИ (без ограничения по количеству)</div>` : '';
                    const virtualSection = virtualAsBots.map(bot => renderBotCard(bot)).join('');
                    const rightPanelHtml = realSection + virtualSectionHeader + virtualSection;

                    console.log(`[DEBUG] Вставляем ПОЛНЫЙ HTML в detailsElement:`, rightPanelHtml);
                    detailsElement.innerHTML = rightPanelHtml;
                    detailsElement.querySelectorAll('.clickable-bot-item').forEach(item => {
                        item.addEventListener('click', (e) => {
                            if (e.target.closest('.bot-icon-btn') || e.target.closest('.bot-card-controls') || e.target.closest('.bot-exchange-link')) return;
                            const url = item.dataset.exchangeUrl;
                            if (url) window.open(url, '_blank');
                        });
                    });
                } else {
                    // Обновляем только данные в существующих карточках (только реальные боты; виртуальные обновляются при полной перерисовке)
                    console.log(`[DEBUG] Обновление данных в "Боты в работе" без перерисовки`);
                    filteredBots.forEach(bot => {
                        const botItem = detailsElement.querySelector(`.active-bot-item[data-symbol="${bot.symbol}"]:not([data-is-virtual="true"])`);
                        if (botItem) {
                            const pnlValue = (bot.unrealized_pnl_usdt || bot.unrealized_pnl || 0);
                            const pnlElement = botItem.querySelector('.bot-header > div:last-child');
                            if (pnlElement) {
                                pnlElement.textContent = `$${pnlValue.toFixed(3)}`;
                                pnlElement.style.color = pnlValue >= 0 ? '#4caf50' : '#f44336';
                            }
                            const d = this.getCompactCardData(bot);
                            const dirEl = botItem.querySelector('.bot-direction');
                            if (dirEl) {
                                dirEl.textContent = d.position;
                                dirEl.style.color = d.positionColor;
                            }
                            const rows = botItem.querySelectorAll('.compact-row');
                            if (rows.length >= 5) {
                                rows[0].querySelector('.compact-val').textContent = d.volume;
                                rows[1].querySelector('.compact-val').textContent = d.entry;
                                rows[2].querySelector('.compact-val').textContent = d.takeProfit;
                                rows[3].querySelector('.compact-val').textContent = d.currentPrice;
                                rows[4].querySelector('.compact-val').textContent = d.stopLoss;
                            }
                            const cardControls = botItem.querySelector('.bot-card-controls');
                            if (cardControls) cardControls.innerHTML = this.getBotDetailButtonsHtml(bot);
                        }
                    });
                }
            }
        }
        
        // Обновляем статистику в правой панели
        this.updateBotsSummaryStats();
        
        this.logDebug('[BotsManager] ✅ Активные боты отрисованы успешно');
    },
            async loadActiveBotsData() {
        this.logDebug('[BotsManager] 🤖 Загрузка данных активных ботов...');
        
        if (!this.serviceOnline) return;
        
        try {
            // ⚡ УБРАНО: Синхронизация позиций теперь выполняется только автоматически воркерами
            // Вызов sync-positions здесь вызывал race condition с остановкой бота
            // и перезаписывал статус PAUSED обратно на in_position_long/short
            
            // Загружаем и ботов, и конфигурацию автобота параллельно
            const [botsResponse, configResponse] = await Promise.all([
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/list`),
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`)
            ]);
            
            const botsData = await botsResponse.json();
            const configData = await configResponse.json();
            
            if (botsData.success) {
                console.log(`[DEBUG] loadActiveBotsData: получены данные ботов:`, botsData.bots);
                this.activeBots = botsData.bots;
                this.activeVirtualPositions = Array.isArray(botsData.virtual_positions) ? botsData.virtual_positions : [];
                console.log(`[DEBUG] loadActiveBotsData: this.activeBots установлен:`, this.activeBots, 'virtual:', this.activeVirtualPositions?.length);
                this.renderActiveBotsDetails();
                
                // Обновляем индикаторы активных ботов в списке монет
                this.updateCoinsListWithBotStatus();
                
                // Обновляем видимость массовых операций
                this.updateBulkControlsVisibility(botsData.bots);
            } else {
                console.log(`[DEBUG] loadActiveBotsData: ошибка загрузки ботов:`, botsData);
            }
            
            // КРИТИЧЕСКИ ВАЖНО: Синхронизируем состояние автобота ТОЛЬКО если переключатель не был изменен пользователем
            if (configData.success) {
                const autoBotEnabled = configData.config.enabled;
                
                // Обновляем глобальный переключатель автобота ТОЛЬКО если он не был изменен пользователем
                const globalAutoBotToggleEl = document.getElementById('globalAutoBotToggle');
                const hasUserChanged = globalAutoBotToggleEl?.hasAttribute('data-user-changed');
                
                this.logDebug(`[BotsManager] 🔄 Синхронизация автобота: сервер=${autoBotEnabled ? 'ВКЛ' : 'ВЫКЛ'}, UI=${globalAutoBotToggleEl?.checked ? 'ВКЛ' : 'ВЫКЛ'}, user-changed=${hasUserChanged}`);
                
                if (globalAutoBotToggleEl && !hasUserChanged) {
                    if (globalAutoBotToggleEl.checked !== autoBotEnabled) {
                        console.log(`[BotsManager] 🔄 Обновляем переключатель: ${globalAutoBotToggleEl.checked} → ${autoBotEnabled}`);
                        console.log(`[BotsManager] 🔍 data-initialized: ${globalAutoBotToggleEl.getAttribute('data-initialized')}`);
                        globalAutoBotToggleEl.checked = autoBotEnabled;
                    }
                    
                    // Обновляем визуальное состояние
                    const toggleLabel = globalAutoBotToggleEl.closest('.auto-bot-toggle')?.querySelector('.toggle-label');
                    if (toggleLabel) {
                        toggleLabel.textContent = autoBotEnabled ? '🤖 Auto Bot (ВКЛ)' : '🤖 Auto Bot (ВЫКЛ)';
                    }
                } else if (hasUserChanged) {
                    console.log(`[BotsManager] 🔒 Пропускаем синхронизацию - пользователь изменил переключатель`);
                }
                
                // Обновляем мобильный переключатель автобота ТОЛЬКО если он не был изменен пользователем
                const mobileAutoBotToggleEl = document.getElementById('mobileAutobotToggle');
                const hasMobileUserChanged = mobileAutoBotToggleEl?.hasAttribute('data-user-changed');
                
                if (mobileAutoBotToggleEl && !hasMobileUserChanged) {
                    if (mobileAutoBotToggleEl.checked !== autoBotEnabled) {
                        console.log(`[BotsManager] 🔄 Обновляем мобильный переключатель: ${mobileAutoBotToggleEl.checked} → ${autoBotEnabled}`);
                        mobileAutoBotToggleEl.checked = autoBotEnabled;
                    }
                    
                    // Обновляем визуальное состояние
                    const statusText = document.getElementById('mobileAutobotStatusText');
                    if (statusText) {
                        statusText.textContent = autoBotEnabled ? 'ВКЛ' : 'ВЫКЛ';
                        statusText.className = autoBotEnabled ? 'mobile-autobot-status enabled' : 'mobile-autobot-status';
                    }
                } else if (hasMobileUserChanged) {
                    console.log(`[BotsManager] 🔒 Пропускаем синхронизацию мобильного - пользователь изменил переключатель`);
                }
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки активных ботов:', error);
        }
    },
            async updateActiveBotsDetailed() {
        if (!this.serviceOnline) return;
        
        try {
            this.logDebug('[BotsManager] 📊 Обновление детальной информации о ботах...');
            
            // Получаем детальную информацию о всех активных ботах
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/active-detailed`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            if (data.success && data.bots) {
                this.updateBotsDetailedDisplay(data.bots);
                this.logDebug(`[BotsManager] ✅ Обновлена детальная информация для ${data.bots.length} ботов`);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка обновления детальной информации о ботах:', error);
        }
    }
    });
})();
