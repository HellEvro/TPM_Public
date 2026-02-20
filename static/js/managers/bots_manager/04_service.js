/**
 * BotsManager - 04_service
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            initializeBotControls() {
        console.log('[BotsManager] Инициализация кнопок управления ботом...');
        
        // Кнопки управления ботом
        const createBotBtn = document.getElementById('createBotBtn');
        console.log('[BotsManager] createBotBtn найдена:', !!createBotBtn);
        const startBotBtn = document.getElementById('startBotBtn');
        const stopBotBtn = document.getElementById('stopBotBtn');
        const pauseBotBtn = document.getElementById('pauseBotBtn');
        const resumeBotBtn = document.getElementById('resumeBotBtn');

        if (createBotBtn) {
            createBotBtn.addEventListener('click', () => this.createBot());
        }
        if (startBotBtn) {
            startBotBtn.addEventListener('click', () => this.startBot());
        }
        if (stopBotBtn) {
            stopBotBtn.addEventListener('click', () => this.stopBot());
        }
        if (pauseBotBtn) {
            pauseBotBtn.addEventListener('click', () => this.pauseBot());
        }
        if (resumeBotBtn) {
            resumeBotBtn.addEventListener('click', () => this.resumeBot());
        }

        // Обработчики для кнопок индивидуальных настроек
        this.initializeIndividualSettingsButtons();
        
        // Обработчики для кнопок быстрого запуска
        this.initializeQuickLaunchButtons();
    },
            async checkBotsService() {
        console.log('[BotsManager] 🔍 Проверка сервиса ботов...');
        console.log('[BotsManager] 🔗 URL:', `${this.BOTS_SERVICE_URL}/api/status`);
        
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/status`, {
                method: 'GET',
                signal: controller.signal,
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                const data = await response.json();
                console.log('[BotsManager] 📊 Ответ сервиса:', data);
                this.serviceOnline = data.status === 'online';
                
                if (this.serviceOnline) {
                    console.log('[BotsManager] ✅ Сервис ботов онлайн');
                    this.updateServiceStatus('online', 'Сервис ботов онлайн');
                    await this.loadCoinsRsiData();
                } else {
                    console.warn('[BotsManager] ⚠️ Сервис ботов недоступен (статус не online)');
                    this.updateServiceStatus('offline', window.languageUtils?.translate?.('bot_service_unavailable') || 'Сервис ботов недоступен');
                }
            } else {
                console.error('[BotsManager] ❌ HTTP ошибка:', response.status, response.statusText);
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
        } catch (error) {
            if (error.name === 'AbortError') {
                console.error('[BotsManager] ❌ Таймаут при проверке сервиса ботов (5 секунд)');
            } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                console.error('[BotsManager] ❌ Ошибка сети при проверке сервиса ботов. Проверьте:');
                console.error('[BotsManager]   1. Запущен ли bots.py?');
                console.error('[BotsManager]   2. Доступен ли порт 5001?');
                console.error('[BotsManager]   3. Нет ли блокировки CORS?');
                console.error('[BotsManager]   URL:', `${this.BOTS_SERVICE_URL}/api/status`);
            } else {
                console.error('[BotsManager] ❌ Ошибка при проверке сервиса ботов:', error);
            }
            this.serviceOnline = false;
            this.updateServiceStatus('offline', 'Сервис ботов недоступен');
            this.showServiceUnavailable();
        }
    },
            updateServiceStatus(status, message) {
        if (this._lastServiceStatus.status === status && this._lastServiceStatus.message === message) {
            return;
        }
        this._lastServiceStatus = { status, message };
        
        const statusElement = document.getElementById('botsServiceStatus');
        const statusDot = document.getElementById('rsiStatusDot');
        
        if (statusElement) {
            const indicator = statusElement.querySelector('.status-indicator');
            const text = statusElement.querySelector('.status-text');
            
            if (indicator) {
                indicator.className = `status-indicator ${status}`;
                indicator.textContent = status === 'online' ? '🟢' : '🔴';
            }
            
            if (text) {
                text.textContent = message;
            }
        }
        
        if (statusDot) {
            statusDot.style.color = status === 'online' ? '#4caf50' : '#f44336';
        }
    },
            showServiceUnavailable() {
        const coinsListElement = document.getElementById('coinsRsiList');
        if (coinsListElement) {
            coinsListElement.innerHTML = `
                <div class="service-unavailable">
                    <h3>🚫 ${window.languageUtils.translate('bot_service_unavailable')}</h3>
                    <p>${window.languageUtils.translate('bot_service_launch_instruction')}</p>
                    <code>python bots.py</code>
                    <p>${window.languageUtils.translate('bot_service_port_instruction')}</p>
                </div>
            `;
        }
    },
            async loadCoinsRsiData(forceUpdate = false) {
        if (!this.serviceOnline) {
            console.warn('[BotsManager] ⚠️ Сервис не онлайн, пропускаем загрузку');
            return;
        }

        // Получаем текущий таймфрейм для логирования
        const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
        this.logDebug(`[BotsManager] 📊 Загрузка данных RSI ${currentTimeframe.toUpperCase()}...`);
        
        // Сохраняем текущее состояние поиска
        const searchInput = document.getElementById('coinSearchInput');
        const currentSearchTerm = searchInput ? searchInput.value : '';
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/coins-with-rsi`);
            
            if (response.ok) {
            const data = await response.json();
            
            if (data.success) {
                    // ✅ ОПТИМИЗАЦИЯ: Проверяем версию данных - обновляем UI только при изменениях.
                    // При forceUpdate (например после обновления ручных позиций) всегда применяем данные.
                    const currentDataVersion = data.data_version || 0;
                    if (!forceUpdate && currentDataVersion === this.lastDataVersion && this.coinsRsiData.length > 0) {
                        this.logDebug('[BotsManager] ⏭️ Данные не изменились (version=' + currentDataVersion + '), пропускаем обновление UI');
                        return;
                    }
                    
                    this.logDebug('[BotsManager] 🔄 Данные обновились (version: ' + this.lastDataVersion + ' → ' + currentDataVersion + ')');
                    this.lastDataVersion = currentDataVersion;
                    
                    // Сохраняем флаг загрузки и статистику для отображения при пустом списке
                    this.lastUpdateInProgress = !!data.update_in_progress;
                    this.lastRsiStats = data.stats || null;
                    
                    // Преобразуем словарь в массив для совместимости с UI
                    this.logDebug('[BotsManager] 🔍 Данные от API:', data);
                    this.logDebug('[BotsManager] 🔍 Ключи coins:', Object.keys(data.coins));
                    this.coinsRsiData = Object.values(data.coins);
                    
                    // Получаем список ручных позиций
                    const manualPositions = data.manual_positions || [];
                    this.logDebug(`[BotsManager] ✋ Ручные позиции получены:`, manualPositions);
                    this.logDebug(`[BotsManager] ✋ Всего ручных позиций: ${manualPositions.length}`);
                    
                    // Помечаем монеты с ручными позициями
                    let markedCount = 0;
                    this.coinsRsiData.forEach(coin => {
                        coin.manual_position = manualPositions.includes(coin.symbol);
                        if (coin.manual_position) {
                            markedCount++;
                            this.logDebug(`[BotsManager] ✋ Монета ${coin.symbol} помечена как ручная позиция`);
                        }
                    });
                    
                    // Загружаем список зрелых монет и помечаем их
                    await this.loadMatureCoinsAndMark();
                    
                    this.logDebug(`[BotsManager] ✅ Загружено ${this.coinsRsiData.length} монет с RSI`);
                    this.logDebug(`[BotsManager] ✅ Помечено ${markedCount} монет с ручными позициями`);
                    this.logDebug('[BotsManager] 🔍 Первые 3 монеты:', this.coinsRsiData.slice(0, 3));
                    
                    // Обновляем интерфейс
                    this.renderCoinsList();
                    this.updateCoinsCounter();
                    
                    // Обновляем информацию о выбранной монете
                    if (this.selectedCoin) {
                        const updatedCoin = this.coinsRsiData.find(coin => coin.symbol === this.selectedCoin.symbol);
                        if (updatedCoin) {
                            this.selectedCoin = updatedCoin;
                            this.updateCoinInfo();
                            this.renderTradesInfo(this.selectedCoin.symbol);
                        }
                    }
                    
                    // Восстанавливаем состояние поиска
                    // ✅ ИСПРАВЛЕНИЕ: Не перезаписываем значение поля (пользователь может печатать!)
                    // Берем АКТУАЛЬНОЕ значение из поля, а не сохраненное
                    const actualSearchTerm = searchInput ? searchInput.value : '';
                    if (actualSearchTerm) {
                        // Применяем фильтр к новому списку монет
                        this.filterCoins(actualSearchTerm);
                        this.updateSmartFilterControls(actualSearchTerm);
                        this.updateClearButtonVisibility(actualSearchTerm);
                    }
                    
                    // Обновляем статус
                    this.updateServiceStatus('online', `${window.languageUtils.translate('updated')}: ${data.last_update ? new Date(data.last_update).toLocaleTimeString() : window.languageUtils.translate('unknown')}`);
                } else {
                    throw new Error(data.error || 'Ошибка загрузки данных');
                }
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки RSI данных:', error);
            this.updateServiceStatus('offline', 'Ошибка загрузки данных');
        }
    },
            async loadDelistedCoins() {
        if (!this.serviceOnline) {
            console.warn('[BotsManager] ⚠️ Сервис не онлайн, пропускаем загрузку делистинговых монет');
            return;
        }

        this.logDebug('[BotsManager] 🚨 Загрузка списка делистинговых монет...');
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/delisted-coins`);
            
            if (response.ok) {
                const data = await response.json();
                
                if (data.success) {
                    // Обновляем список делистинговых монет
                    this.delistedCoins = Object.keys(data.delisted_coins || {});
                    
                    this.logDebug(`[BotsManager] ✅ Загружено ${this.delistedCoins.length} делистинговых монет: ${this.delistedCoins.join(', ')}`);
                    
                    // Обновляем время последнего сканирования
                    if (data.last_scan) {
                        console.log(`[BotsManager] 📅 Последнее сканирование делистинга: ${new Date(data.last_scan).toLocaleString()}`);
                    }
                } else {
                    console.warn('[BotsManager] ⚠️ Ошибка загрузки делистинговых монет:', data.error);
                }
            } else {
                console.warn(`[BotsManager] ⚠️ HTTP ${response.status} при загрузке делистинговых монет`);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки делистинговых монет:', error);
        }
    },
            async loadMatureCoinsCount() {
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/mature-coins-list`);
            const data = await response.json();
            
            if (data.success) {
                const countEl = document.getElementById('matureCoinsCount');
                if (countEl) {
                    countEl.textContent = `(${data.total_count})`;
                }
            }
        } catch (error) {
            console.error('[BotsManager] Ошибка загрузки счётчика зрелых монет:', error);
        }
    }
    
    /**
     * Загружает список зрелых монет и помечает их в данных
     */,
            async loadMatureCoinsAndMark() {
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/mature-coins-list`);
            const data = await response.json();
            
            if (data.success && data.mature_coins) {
                // Помечаем зрелые монеты в данных
                let markedCount = 0;
                this.coinsRsiData.forEach(coin => {
                    coin.is_mature = data.mature_coins.includes(coin.symbol);
                    if (coin.is_mature) {
                        markedCount++;
                    }
                });
                
                // ✅ ИСПРАВЛЕНИЕ: Обновляем счетчик зрелых монет в UI
                await this.loadMatureCoinsCount();
                
                this.logDebug(`[BotsManager] 💎 Помечено ${markedCount} зрелых монет из ${data.total_count} общих`);
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки зрелых монет:', error);
        }
    }
    
    /**
     * Показывает уведомление
     */,
                updateCoinsCounter() {
        // Обновляем счетчики для новых фильтров сигналов
        this.updateSignalCounters();
        
        // Обновляем счетчик ручных позиций
        this.updateManualPositionCounter();
    }
    
    /**
     * Обновляет счетчик ручных позиций
     */
    });
})();
