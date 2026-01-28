/**
 * Главный менеджер ботов - координирует все модули
 * Модульная архитектура для упрощения разработки и поддержки
 */
class BotsManager {
    constructor() {
        // Инициализация модулей
        this.apiService = new BotsApiService();
        this.uiComponents = new BotsUIComponents();
        this.dataManager = new BotsDataManager();
        this.configManager = new BotsConfigManager();
        this.controlManager = new BotsControlManager();
        this.filtersManager = new BotsFiltersManager();

        // Состояние
        this.serviceOnline = false;
        this.updateInterval = null;
        this.monitoringTimer = null;
        this.refreshInterval = 3000; // 3 секунды
        this.trendLabelsUpdated = false;

        // Привязка контекста для модулей
        this.bindModuleContexts();

        // Инициализация при создании
        this.init();
    }

    /**
     * Привязка контекста для модулей
     */
    bindModuleContexts() {
        // Передаем ссылку на главный менеджер в модули
        this.apiService.botsManager = this;
        this.uiComponents.botsManager = this;
        this.dataManager.botsManager = this;
        this.configManager.botsManager = this;
        this.controlManager.botsManager = this;
        this.filtersManager.botsManager = this;
    }

    /**
     * Инициализация менеджера
     */
    async init() {
        console.log('[BotsManager] 🚀 Инициализация менеджера ботов...');
        
        try {
            // Инициализируем интерфейс
            this.initializeInterface();
            
            // Проверяем статус сервиса ботов
            await this.checkBotsService();
            
            // Запускаем периодическое обновление
            this.startPeriodicUpdate();
            
            // Принудительная загрузка конфигурации
            setTimeout(() => {
                console.log('[BotsManager] 🔄 Принудительная загрузка конфигурации...');
                this.loadConfigurationData();
            }, 2000);
            
            // Принудительное обновление состояния автобота и ботов
            setTimeout(() => {
                console.log('[BotsManager] 🔄 Принудительное обновление состояния автобота...');
                this.loadActiveBotsData();
            }, 1000);
            
            // Принудительное обновление подписей тренд-фильтров
            setTimeout(() => {
                console.log('[BotsManager] 🔄 Принудительное обновление подписей тренд-фильтров...');
                this.trendLabelsUpdated = false;
                this.updateTrendFilterLabels();
            }, 3000);
            
            console.log('[BotsManager] ✅ Менеджер ботов инициализирован');
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка инициализации:', error);
            this.showServiceUnavailable();
        }
    }

    /**
     * Инициализация интерфейса
     */
    initializeInterface() {
        console.log('[BotsManager] 🔧 Инициализация интерфейса...');
        
        // Инициализируем табы
        this.initializeTabs();
        
        // Инициализируем поиск
        this.initializeSearch();
        
        // Загружаем информацию о счете
        this.loadAccountInfo();
        
        // Инициализируем фильтры RSI
        this.filtersManager.initializeRsiFilters();
        
        // Инициализируем управление ботом
        this.initializeBotControls();
        
        // Инициализируем кнопки области действия
        this.initializeScopeButtons();
        
        // Инициализируем кнопки управления
        this.initializeManagementButtons();
        
        // Принудительно применяем стили для читаемости
        this.uiComponents.applyReadabilityStyles();
        
        console.log('[BotsManager] ✅ Интерфейс инициализирован');
    }

    /**
     * Инициализация табов
     */
    initializeTabs() {
        const tabButtons = document.querySelectorAll('.tab-btn');
        
        tabButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const tabName = btn.dataset.tab;
                console.log('[BotsManager] 📑 Переключение на таб:', tabName);
                this.switchTab(tabName);
            });
        });
    }

    /**
     * Переключение таба
     */
    switchTab(tabName) {
        // Обновляем кнопки табов
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        // Обновляем содержимое табов
        document.querySelectorAll('.tab-content').forEach(content => {
            const tabIdMap = {
                'management': 'managementTab',
                'filters': 'filtersTab', 
                'active-bots': 'activeBotsTab',
                'configuration': 'configurationTab'
            };
            
            const targetId = tabIdMap[tabName] || `${tabName}Tab`;
            const isActive = content.id === targetId;
            content.classList.toggle('active', isActive);
        });

        // Загружаем данные для активного таба
        this.loadTabData(tabName);
    }

    /**
     * Загрузка данных для таба
     */
    loadTabData(tabName) {
        switch (tabName) {
            case 'management':
                this.loadCoinsRsiData();
                break;
            case 'filters':
                this.filtersManager.loadFiltersData();
                break;
            case 'active-bots':
                this.loadActiveBotsData();
                break;
            case 'configuration':
                this.loadConfigurationData();
                break;
        }
    }

    /**
     * Инициализация поиска
     */
    initializeSearch() {
        const searchInput = document.getElementById('coinSearchInput');
        const clearSearchBtn = document.getElementById('clearSearchBtn');
        
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const searchTerm = e.target.value;
                this.dataManager.setSearchTerm(searchTerm);
                this.updateClearButtonVisibility(searchTerm);
                this.performSearch(searchTerm);
            });
        }
        
        if (clearSearchBtn) {
            clearSearchBtn.addEventListener('click', () => this.clearSearch());
        }
    }

    /**
     * Выполнение поиска
     */
    performSearch(searchTerm) {
        const filteredCoins = this.dataManager.searchCoins(searchTerm);
        this.filtersManager.renderFilteredCoins(filteredCoins);
        this.filtersManager.updateSmartFilterControls(searchTerm);
    }

    /**
     * Обновление видимости кнопки очистки поиска
     */
    updateClearButtonVisibility(searchTerm) {
        const clearSearchBtn = document.getElementById('clearSearchBtn');
        if (clearSearchBtn) {
            clearSearchBtn.style.display = searchTerm && searchTerm.length > 0 ? 'flex' : 'none';
        }
    }

    /**
     * Очистка поиска
     */
    clearSearch() {
        const searchInput = document.getElementById('coinSearchInput');
        if (searchInput) {
            searchInput.value = '';
            this.dataManager.setSearchTerm('');
            this.performSearch('');
        }
    }

    /**
     * Инициализация управления ботом
     */
    initializeBotControls() {
        const createBotBtn = document.getElementById('createBotBtn');
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
    }

    /**
     * Инициализация кнопок области действия
     */
    initializeScopeButtons() {
        const scopeButtons = document.querySelectorAll('.scope-btn');
        
        scopeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const scope = btn.dataset.scope;
                this.updateScope(scope);
            });
        });
    }

    /**
     * Обновление области действия
     */
    updateScope(scope) {
        // Обновляем активную кнопку
        document.querySelectorAll('.scope-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.scope === scope);
        });

        // Обновляем конфигурацию
        const config = this.configManager.getCurrentConfig();
        if (config) {
            config.scope = scope;
            this.configManager.saveAutoBotConfig(config);
        }
    }

    /**
     * Инициализация кнопок управления
     */
    initializeManagementButtons() {
        const addToWhitelistBtn = document.getElementById('addToWhitelistBtn');
        const addToBlacklistBtn = document.getElementById('addToBlacklistBtn');
        const removeFromFiltersBtn = document.getElementById('removeFromFiltersBtn');
        
        if (addToWhitelistBtn) {
            addToWhitelistBtn.addEventListener('click', () => this.filtersManager.addSelectedCoinToWhitelist());
        }
        if (addToBlacklistBtn) {
            addToBlacklistBtn.addEventListener('click', () => this.filtersManager.addSelectedCoinToBlacklist());
        }
        if (removeFromFiltersBtn) {
            removeFromFiltersBtn.addEventListener('click', () => this.filtersManager.removeSelectedCoinFromFilters());
        }
    }

    /**
     * Проверка статуса сервиса ботов
     */
    async checkBotsService() {
        console.log('[BotsManager] 🔍 Проверка сервиса ботов...');
        
        try {
            const result = await this.apiService.checkServiceStatus();
            this.serviceOnline = result.online;
            this.updateServiceStatus(this.serviceOnline ? 'online' : 'offline', result.error);
            
            if (this.serviceOnline) {
                console.log('[BotsManager] ✅ Сервис ботов онлайн');
            } else {
                console.log('[BotsManager] ❌ Сервис ботов офлайн:', result.error);
                this.showServiceUnavailable();
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка проверки сервиса ботов:', error);
            this.serviceOnline = false;
            this.updateServiceStatus('offline', error.message);
            this.showServiceUnavailable();
        }
    }

    /**
     * Обновление статуса сервиса
     */
    updateServiceStatus(status, message) {
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
                text.textContent = status === 'online' ? 'Онлайн' : 'Офлайн';
            }
        }
        
        if (statusDot) {
            statusDot.className = `status-dot ${status}`;
        }
    }

    /**
     * Показать сообщение о недоступности сервиса
     */
    showServiceUnavailable() {
        const coinsListElement = document.getElementById('coinsRsiList');
        if (coinsListElement) {
            coinsListElement.innerHTML = `
                <div class="service-unavailable">
                    <h3>🚫 ${window.languageUtils.translate('bot_service_unavailable_new')}</h3>
                    <p>${window.languageUtils.translate('bot_service_launch_terminal')}</p>
                    <code>python bots.py</code>
                    <p><small>${window.languageUtils.translate('bot_service_port_small')}</small></p>
                    <button class="btn btn-primary" onclick="location.reload()">
                        <i class="fas fa-refresh"></i> ${window.languageUtils.translate('refresh_page')}
                    </button>
                </div>
            `;
        }
        
        // Также обновляем статус в интерфейсе
        this.updateServiceStatus('offline', 'Сервис недоступен');
    }

    /**
     * Загрузка данных RSI монет
     */
    async loadCoinsRsiData() {
        if (!this.serviceOnline) {
            console.warn('[BotsManager] ⚠️ Сервис не онлайн, пропускаем загрузку');
            return;
        }

        const searchInput = document.getElementById('coinSearchInput');
        const currentSearchTerm = searchInput ? searchInput.value : '';
        
        try {
            const response = await this.apiService.getCoinsWithRsi();
            
            if (response.success) {
                this.dataManager.setCoinsRsiData(response.data);
                
                // Обновляем выбранную монету если она есть
                if (this.dataManager.getSelectedCoin()) {
                    const updatedCoin = this.dataManager.getCoinsRsiData().find(coin => 
                        coin.symbol === this.dataManager.getSelectedCoin().symbol
                    );
                    if (updatedCoin) {
                        this.dataManager.setSelectedCoin(updatedCoin);
                    }
                }
                
                this.renderCoinsList();
                console.log(`[BotsManager] ✅ Загружено ${response.data.length} монет с RSI данными`);
            } else {
                throw new Error(response.error || 'Ошибка загрузки данных RSI');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки данных RSI:', error);
            this.showServiceUnavailable();
        }
    }

    /**
     * Рендеринг списка монет
     */
    renderCoinsList() {
        const coinsListElement = document.getElementById('coinsRsiList');
        if (!coinsListElement) return;

        const coins = this.dataManager.getFilteredData();
        
        if (coins.length === 0) {
            coinsListElement.innerHTML = `
                <div class="loading-state">
                    <p>⏳ ${window.languageUtils.translate('loading_rsi_data')}</p>
                    <small>${window.languageUtils.translate('first_load_warning')}</small>
                </div>
            `;
            return;
        }

        // Получаем текущий таймфрейм для отображения данных
        const currentTimeframe = document.getElementById('systemTimeframe')?.value || '6h';
        const rsiKey = `rsi${currentTimeframe}`;
        const trendKey = `trend${currentTimeframe}`;
        
        const coinsHtml = coins.map(coin => {
            const rsiValue = coin[rsiKey] || coin.rsi6h || coin.rsi || 50;
            const trendValue = coin[trendKey] || coin.trend6h || coin.trend || 'NEUTRAL';
            const rsiClass = this.uiComponents.getRsiZoneClass(rsiValue);
            const trendClass = trendValue ? `trend-${trendValue.toLowerCase()}` : '';
            const effectiveSignal = this.uiComponents.getEffectiveSignal(coin);
            const signalClass = effectiveSignal === 'ENTER_LONG' ? 'enter-long' : 
                               effectiveSignal === 'ENTER_SHORT' ? 'enter-short' : '';

            return `
                <li class="coin-item ${rsiClass} ${trendClass} ${signalClass}" data-symbol="${coin.symbol}">
                    <div class="coin-item-content">
                        <div class="coin-header">
                            <span class="coin-symbol">${coin.symbol}</span>
                            <div class="coin-header-right">
                                ${this.uiComponents.generateWarningIndicator(coin)}
                                <span class="coin-rsi ${rsiClass}">${rsiValue}</span>
                                <a href="${this.uiComponents.createTickerLink(coin.symbol)}" 
                                   target="_blank" 
                                   class="external-link" 
                                   title="Открыть на бирже"
                                   onclick="event.stopPropagation()">
                                    <i class="fas fa-external-link-alt"></i>
                                </a>
                            </div>
                        </div>
                        <div class="coin-details">
                            <span class="coin-trend ${trendValue}">${trendValue}</span>
                            <span class="coin-price">$${coin.price?.toFixed(6) || '0'}</span>
                            ${this.uiComponents.generateEnhancedSignalInfo(coin)}
                        </div>
                    </div>
                </li>
            `;
        }).join('');

        coinsListElement.innerHTML = coinsHtml;

        // Добавляем обработчики кликов
        this.attachCoinClickHandlers();

        // Обновляем счетчики
        this.uiComponents.updateCoinsCounter();
        this.uiComponents.updateSignalCounters();
    }

    /**
     * Привязка обработчиков кликов для монет
     */
    attachCoinClickHandlers() {
        const coinItems = document.querySelectorAll('.coin-item');
        
        coinItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const symbol = item.dataset.symbol;
                this.selectCoin(symbol);
            });
        });
    }

    /**
     * Выбор монеты
     */
    selectCoin(symbol) {
        const coin = this.dataManager.getCoinBySymbol(symbol);
        if (!coin) return;

        this.dataManager.setSelectedCoin(coin);
        
        // Обновляем выделение
        document.querySelectorAll('.coin-item').forEach(item => {
            item.classList.toggle('selected', item.dataset.symbol === symbol);
        });

        // Показываем интерфейс управления ботом
        this.showBotControlInterface();
        
        // Обновляем информацию о монете
        this.updateCoinInfo();
        
        console.log(`[BotsManager] 📌 Выбрана монета: ${symbol}`);
    }

    /**
     * Показ интерфейса управления ботом
     */
    showBotControlInterface() {
        const controlInterface = document.getElementById('botControlInterface');
        if (controlInterface) {
            controlInterface.style.display = 'block';
        }
    }

    /**
     * Обновление информации о монете
     */
    updateCoinInfo() {
        const selectedCoin = this.dataManager.getSelectedCoin();
        if (!selectedCoin) return;

        const coinInfoElement = document.getElementById('selectedCoinInfo');
        if (coinInfoElement) {
            coinInfoElement.innerHTML = `
                <h4>${selectedCoin.symbol}</h4>
                <p>RSI ${currentTimeframe.toUpperCase()}: <span class="rsi-value">${selectedCoin[rsiKey] || selectedCoin.rsi6h || selectedCoin.rsi || '-'}</span></p>
                <p>Тренд: <span class="trend-value">${selectedCoin[trendKey] || selectedCoin.trend6h || selectedCoin.trend || 'NEUTRAL'}</span></p>
                <p>Цена: <span class="price-value">$${selectedCoin.price?.toFixed(6) || '0'}</span></p>
            `;
        }

        // Обновляем кнопки управления ботом
        this.uiComponents.updateBotControlButtons();
    }

    /**
     * Загрузка активных ботов
     */
    async loadActiveBotsData() {
        if (!this.serviceOnline) {
            console.warn('[BotsManager] ⚠️ Сервис не онлайн, пропускаем загрузку ботов');
            return;
        }

        try {
            // Синхронизация позиций с биржей каждые 3 секунды
            try {
                const syncResponse = await fetch(`${this.apiService.baseUrl}/sync-positions`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                const syncData = await syncResponse.json();
                if (syncData.success) {
                    console.log('[BotsManager] ✅ Позиции синхронизированы успешно');
                } else {
                    console.warn('[BotsManager] ⚠️ Ошибка синхронизации позиций:', syncData.message);
                }
            } catch (syncError) {
                console.warn('[BotsManager] ⚠️ Ошибка синхронизации позиций:', syncError);
            }

            const [botsResponse, autoBotResponse] = await Promise.all([
                this.apiService.getBotsList(),
                this.apiService.getAutoBotConfig()
            ]);

            if (botsResponse.success) {
                this.dataManager.setActiveBots(botsResponse.data);
                this.renderActiveBotsDetails();
            }

            if (autoBotResponse.success) {
                this.configManager.autoBotConfig = autoBotResponse.data;
                this.updateRsiThresholds(autoBotResponse.data);
            }

            console.log('[BotsManager] ✅ Данные активных ботов загружены');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки активных ботов:', error);
        }
    }

    /**
     * Рендеринг детальной информации об активных ботах
     */
    renderActiveBotsDetails() {
        const activeBots = this.dataManager.getActiveBots();
        const container = document.getElementById('activeBotsDetails');
        
        if (!container) return;

        if (activeBots.length === 0) {
            container.innerHTML = `
                <div class="no-bots">
                    <p>🤖 Активных ботов нет</p>
                    <small>Создайте бота на вкладке "Управление"</small>
                </div>
            `;
            return;
        }

        const botsHtml = activeBots.map(bot => {
            const positionInfo = this.controlManager.getBotPositionInfo(bot);
            const timeInfo = this.controlManager.getBotTimeInfo(bot);
            
            return `
                <div class="bot-card" data-symbol="${bot.symbol}">
                    <div class="bot-header">
                        <h4>${bot.symbol}</h4>
                        <span class="bot-status ${bot.status}">${bot.status}</span>
                    </div>
                    <div class="bot-details">
                        ${positionInfo ? `
                            <div class="position-info">
                                <p>PnL: <span class="${positionInfo.isProfitable ? 'profit' : 'loss'}">${positionInfo.pnl.toFixed(2)} USDT</span></p>
                                <p>ROI: <span class="${positionInfo.isProfitable ? 'profit' : 'loss'}">${positionInfo.roi.toFixed(2)}%</span></p>
                                <p>Сторона: ${positionInfo.side}</p>
                            </div>
                        ` : ''}
                        ${timeInfo ? `
                            <div class="time-info">
                                <p>Время работы: ${timeInfo.formatted}</p>
                            </div>
                        ` : ''}
                    </div>
                    <div class="bot-actions">
                        ${this.uiComponents.getBotControlButtonsHtml(bot)}
                        ${this.uiComponents.getBotDetailButtonsHtml(bot)}
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = botsHtml;
    }

    /**
     * Обновление пороговых значений RSI
     */
    updateRsiThresholds(config) {
        const oldLongThreshold = this.configManager.rsiLongThreshold;
        const oldShortThreshold = this.configManager.rsiShortThreshold;
        
        this.configManager.rsiLongThreshold = config.rsi_long_threshold || 29;
        this.configManager.rsiShortThreshold = config.rsi_short_threshold || 71;
        
        // Обновляем UI компоненты
        this.uiComponents.updateRsiThresholds(this.configManager.rsiLongThreshold, this.configManager.rsiShortThreshold);
        
        // Обновляем фильтры
        this.filtersManager.updateRsiFilterButtons();
        this.filtersManager.updateTrendFilterLabels();
        
        // Обновляем классы RSI для монет
        this.refreshCoinsRsiClasses();
        
        console.log(`[BotsManager] 🔄 RSI пороги обновлены: LONG=${this.configManager.rsiLongThreshold}, SHORT=${this.configManager.rsiShortThreshold}`);
    }

    /**
     * Обновление классов RSI для монет
     */
    refreshCoinsRsiClasses() {
        const coinItems = document.querySelectorAll('.coin-item');
        
        coinItems.forEach(item => {
            const symbol = item.dataset.symbol;
            const coinData = this.dataManager.getCoinBySymbol(symbol);
            
            if (coinData) {
                // Удаляем старые классы
                item.classList.remove('buy-zone', 'sell-zone', 'enter-long', 'enter-short');
                
                // Добавляем новые классы
                const rsiClass = this.uiComponents.getRsiZoneClass(coinData.rsi6h);
                if (rsiClass) {
                    item.classList.add(rsiClass);
                }
                
                const effectiveSignal = this.uiComponents.getEffectiveSignal(coinData);
                if (effectiveSignal === 'ENTER_LONG') {
                    item.classList.add('enter-long');
                } else if (effectiveSignal === 'ENTER_SHORT') {
                    item.classList.add('enter-short');
                }
            }
        });
    }

    /**
     * Запуск периодического обновления
     */
    startPeriodicUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        this.updateInterval = setInterval(() => {
            this.performPeriodicUpdate();
        }, this.refreshInterval);

        console.log(`[BotsManager] 🔄 Запущено периодическое обновление каждые ${this.refreshInterval / 1000} секунд`);
    }

    /**
     * Выполнение периодического обновления
     */
    async performPeriodicUpdate() {
        try {
            // Проверяем статус сервиса
            await this.checkBotsService();
            
            if (this.serviceOnline) {
                // Загружаем данные RSI
                await this.loadCoinsRsiData();
                
                // Загружаем активных ботов
                await this.loadActiveBotsData();
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка периодического обновления:', error);
        }
    }

    /**
     * Остановка периодического обновления
     */
    stopPeriodicUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    /**
     * Загрузка конфигурации
     */
    async loadConfigurationData() {
        await this.configManager.loadAutoBotConfig();
        await this.configManager.loadSystemConfig();
    }

    /**
     * Обновление подписей тренд-фильтров
     */
    updateTrendFilterLabels() {
        this.filtersManager.updateTrendFilterLabels();
    }

    // Методы управления ботами (делегирование к controlManager)
    async createBot() {
        const selectedCoin = this.dataManager.getSelectedCoin();
        if (!selectedCoin) {
            this.showNotification('Выберите монету для создания бота', 'warning');
            return;
        }

        return await this.controlManager.createBot(selectedCoin.symbol);
    }

    async startBot(symbol) {
        return await this.controlManager.startBot(symbol);
    }

    async stopBot(symbol) {
        return await this.controlManager.stopBot(symbol);
    }

    async pauseBot(symbol) {
        return await this.controlManager.pauseBot(symbol);
    }

    async resumeBot(symbol) {
        return await this.controlManager.resumeBot(symbol);
    }

    async deleteBot(symbol) {
        return await this.controlManager.deleteBot(symbol);
    }

    /**
     * Загрузка информации о счете
     */
    async loadAccountInfo() {
        if (!this.serviceOnline) {
            console.warn('[BotsManager] ⚠️ Сервис не онлайн, пропускаем загрузку информации о счете');
            return;
        }

        try {
            const response = await this.apiService.getAccountInfo();
            if (response.success) {
                console.log('[BotsManager] ✅ Информация о счете загружена');
                // Здесь можно обновить UI с информацией о счете
            } else {
                throw new Error(response.error || 'Ошибка загрузки информации о счете');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка запроса информации о счете:', error);
        }
    }

    /**
     * Обновление области действия
     */
    updateScope(scope) {
        // Обновляем активную кнопку
        document.querySelectorAll('.scope-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.scope === scope);
        });

        // Обновляем конфигурацию
        const config = this.configManager.getCurrentConfig();
        if (config) {
            config.scope = scope;
            this.configManager.saveAutoBotConfig(config);
        }
    }

    /**
     * Показать уведомление
     */
    showNotification(message, type = 'info') {
        this.uiComponents.showConfigNotification('Уведомление', message, type);
    }

    /**
     * Уничтожение менеджера
     */
    destroy() {
        this.stopPeriodicUpdate();
        
        if (this.monitoringTimer) {
            clearInterval(this.monitoringTimer);
        }
        
        console.log('[BotsManager] 🗑️ Менеджер ботов уничтожен');
    }
}

// BotsManager создается в bots_modules_loader.js после загрузки всех модулей
