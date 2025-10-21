/**
 * Менеджер ботов - управление торговыми ботами
 * Работает с отдельным сервисом bots.py на порту 5001
 */
class BotsManager {
    constructor() {
        this.selectedCoin = null;
        this.coinsRsiData = [];
        this.activeBots = [];
        this.serviceOnline = false;
        this.updateInterval = null;
        this.accountUpdateInterval = null;
        this.currentRsiFilter = 'all'; // Отслеживание текущего фильтра
        
        // RSI пороговые значения из конфигурации
        this.rsiLongThreshold = 29;
        this.rsiShortThreshold = 71;
        
        // Флаг для предотвращения множественных обновлений подписей
        this.trendLabelsUpdated = false;
        
        // Единый интервал обновления UI и мониторинга ботов
        this.refreshInterval = 3000; // 3 секунды по умолчанию
        this.monitoringTimer = null;
        
        // Debounce для поиска
        this.searchDebounceTimer = null;
        
        // URL сервиса ботов - используем тот же хост что и у приложения
        this.BOTS_SERVICE_URL = `${window.location.protocol}//${window.location.hostname}:5001`;
        this.apiUrl = `${window.location.protocol}//${window.location.hostname}:5001/api/bots`; // Для совместимости
        
        // Уровень логирования: 'error' - только ошибки, 'info' - важные события, 'debug' - все
        this.logLevel = 'error'; // ✅ ОТКЛЮЧЕНЫ СПАМ-ЛОГИ - только ошибки
        
        // Инициализация при создании
        this.init();
    }
    
    // Методы логирования с уровнями
    logDebug(...args) {
        if (this.logLevel === 'debug') {
            console.log(...args);
        }
    }
    
    logInfo(...args) {
        if (this.logLevel === 'info' || this.logLevel === 'debug') {
            console.log(...args);
        }
    }
    
    logError(...args) {
        console.error(...args);
    }

    async init() {
        console.log('[BotsManager] 🚀 Инициализация менеджера ботов...');
        console.log('[BotsManager] 💡 Для включения debug логов: window.botsManager.logLevel = "debug"');
        
        try {
            // Инициализируем интерфейс
            this.initializeInterface();
            
            // КРИТИЧЕСКИ ВАЖНО: Инициализируем обработчик Auto Bot переключателя
            console.log('[BotsManager] 🤖 Инициализация обработчика Auto Bot переключателя...');
            this.initializeGlobalAutoBotToggle();
            this.initializeMobileAutoBotToggle();
            
            // Проверяем статус сервиса ботов
            await this.checkBotsService();
            
            // Синхронизируем позиции при инициализации
            if (this.serviceOnline) {
                console.log('[BotsManager] 🔄 Синхронизация позиций при инициализации...');
                try {
                    const syncResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/sync-positions`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    const syncData = await syncResponse.json();
                    if (syncData.success) {
                        this.logDebug('[BotsManager] ✅ Позиции синхронизированы при инициализации');
                    } else {
                        console.warn('[BotsManager] ⚠️ Ошибка синхронизации позиций при инициализации:', syncData.message);
                    }
                } catch (syncError) {
                    console.warn('[BotsManager] ⚠️ Ошибка синхронизации позиций при инициализации:', syncError);
                }
            }
            
            // Запускаем периодическое обновление
            this.startPeriodicUpdate();
            
            // Принудительная загрузка конфигурации
            setTimeout(() => {
                console.log('[BotsManager] 🔄 Принудительная загрузка конфигурации...');
                this.loadConfigurationData();
            }, 2000);
            
            // Принудительное обновление состояния автобота и ботов (только при первой загрузке)
            setTimeout(() => {
                this.logDebug('[BotsManager] 🔄 Принудительное обновление состояния автобота...');
                this.loadActiveBotsData();
            }, 1000);
            
            // Принудительное обновление подписей тренд-фильтров
            setTimeout(() => {
                this.logDebug('[BotsManager] 🔄 Принудительное обновление подписей тренд-фильтров...');
                this.trendLabelsUpdated = false; // Сбрасываем флаг для принудительного обновления
                this.updateTrendFilterLabels();
            }, 3000);
            
            console.log('[BotsManager] ✅ Менеджер ботов инициализирован');
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка инициализации:', error);
            this.showServiceUnavailable();
        }
    }

    initializeInterface() {
        console.log('[BotsManager] 🔧 Инициализация интерфейса...');
        
        // Инициализируем табы
        this.initializeTabs();
        
        // Инициализируем поиск
        this.initializeSearch();
        
        // Загружаем информацию о счете
        this.loadAccountInfo();
        
        // Инициализируем фильтры RSI
        this.initializeRsiFilters();
        
        // Инициализируем управление ботом
        this.initializeBotControls();
        
        // Инициализируем кнопки области действия
        this.initializeScopeButtons();
        
        // Инициализируем кнопки управления
        this.initializeManagementButtons();
        
        // Инициализируем кнопки конфигурации (должны работать всегда!)
        this.initializeConfigurationButtons();
        
        // Загружаем счётчик зрелых монет
        this.loadMatureCoinsCount();
        
        // Принудительно применяем стили для читаемости
        this.applyReadabilityStyles();
        
        // Инициализируем кнопку обновления ручных позиций
        this.initializeManualPositionsControls();
        
        // Инициализируем кнопки загрузки RSI
        this.initializeRSILoadingButtons();
        
        console.log('[BotsManager] ✅ Интерфейс инициализирован');
    }

    applyReadabilityStyles() {
        // Принудительно применяем стили для select'ов и input'ов
        const applyStyles = () => {
            const selectors = [
                '.bots-page select',
                '.config-select',
                '#autoBotScope',
                '#checkInterval', 
                '#volumeModeSelect',
                '#rsiLongThreshold',
                '#rsiShortThreshold',
                '#rsiExitLong',
                '#rsiExitShort',
                '#defaultPositionSize',
                '#autoBotMaxConcurrent',
                '#autoBotRiskCap'
            ];
            
            selectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    el.style.background = '#2a2a2a';
                    el.style.color = '#ffffff';
                    el.style.border = '1px solid #404040';
                    
                    // Также применяем к option'ам
                    const options = el.querySelectorAll('option');
                    options.forEach(option => {
                        option.style.background = '#2a2a2a';
                        option.style.color = '#ffffff';
                    });
            });
        });

            console.log('[BotsManager] 🎨 Применены стили читаемости');
        };
        
        // Применяем сразу и через небольшую задержку
        applyStyles();
        setTimeout(applyStyles, 500);
        setTimeout(applyStyles, 1000);
    }

    initializeTabs() {
        console.log('[BotsManager] 🔧 Инициализация системы табов...');
        
        // Обработчики кликов по табам
        document.querySelectorAll('.bots-tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Получаем data-tab с кнопки, а не с target (который может быть span)
                const tabName = btn.dataset.tab;
                console.log('[BotsManager] 📑 Переключение на таб:', tabName);
                this.switchTab(tabName);
            });
        });

        console.log('[BotsManager] ✅ Система табов инициализирована');
    }

    switchTab(tabName) {
        console.log('[BotsManager] 🔄 Переключение на таб:', tabName);
        
        // Переключаем кнопки
        document.querySelectorAll('.bots-tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        // Переключаем контент  
        document.querySelectorAll('.bots-tab-content').forEach(content => {
            // Мапинг названий табов к их ID
            const tabIdMap = {
                'management': 'managementTab',
                'filters': 'filtersTab', 
                'config': 'configTab',
                'active-bots': 'activeBotsTab'
            };
            
            const targetId = tabIdMap[tabName] || `${tabName}Tab`;
            const isActive = content.id === targetId;
            content.classList.toggle('active', isActive);
        });

        // Загружаем данные для соответствующего таба
        switch(tabName) {
                    case 'management':
            this.loadCoinsRsiData();
            this.loadFiltersData(); // Загружаем фильтры для кнопок управления
            this.loadDuplicateSettings(); // Загружаем дублированные настройки
            break;
            case 'filters':
                this.loadFiltersData();
                break;
            case 'config':
                console.log('[BotsManager] 🎛️ Переключение на вкладку КОНФИГУРАЦИЯ');
                // Применяем стили при открытии конфигурации
                setTimeout(() => this.applyReadabilityStyles(), 100);
                // Показываем индикатор загрузки
                this.showConfigurationLoading(true);
                console.log('[BotsManager] ⏳ Индикатор загрузки включен');
                // Загружаем текущую конфигурацию с задержкой для DOM
                setTimeout(() => {
                    console.log('[BotsManager] 📋 Загружаем конфигурацию для вкладки config...');
                    this.loadConfigurationData().finally(() => {
                        console.log('[BotsManager] ✅ Загрузка конфигурации завершена, скрываем индикатор');
                        this.showConfigurationLoading(false);
                    });
                }, 200);
                break;
            case 'active-bots':
            case 'activeBotsTab':
                this.loadActiveBotsData();
                break;
        }
        
        console.log('[BotsManager] ✅ Таб переключен успешно');
    }

    initializeSearch() {
        const searchInput = document.getElementById('coinSearchInput');
        const clearSearchBtn = document.getElementById('clearSearchBtn');
        
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const searchTerm = e.target.value;
                
                // ✅ DEBOUNCE: Отменяем предыдущий таймер
                if (this.searchDebounceTimer) {
                    clearTimeout(this.searchDebounceTimer);
                }
                
                // ✅ Сразу обновляем кнопку очистки (без задержки)
                this.updateClearButtonVisibility(searchTerm);
                
                // ✅ Фильтрацию делаем с задержкой 150ms
                this.searchDebounceTimer = setTimeout(() => {
                    this.filterCoins(searchTerm);
                    this.updateSmartFilterControls(searchTerm);
                }, 150);
            });
        }
        
        if (clearSearchBtn) {
            clearSearchBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                // ✅ Отменяем любые pending фильтрации
                if (this.searchDebounceTimer) {
                    clearTimeout(this.searchDebounceTimer);
                }
                
                this.clearSearch();
            });
        }
    }
    
    updateClearButtonVisibility(searchTerm) {
        const clearSearchBtn = document.getElementById('clearSearchBtn');
        if (clearSearchBtn) {
            clearSearchBtn.style.display = searchTerm && searchTerm.length > 0 ? 'flex' : 'none';
        }
    }
    
    clearSearch() {
        console.log('[BotsManager] 🧹 Очистка поиска...');
        const searchInput = document.getElementById('coinSearchInput');
        if (searchInput) {
            // ✅ Очищаем поле
            searchInput.value = '';
            
            // ✅ Применяем пустой фильтр
            this.filterCoins('');
            this.updateSmartFilterControls('');
            this.updateClearButtonVisibility('');
            
            // ✅ Возвращаем фокус
            searchInput.focus();
            
            console.log('[BotsManager] ✅ Поиск очищен');
        }
    }

    initializeManagementButtons() {
        // Кнопки фильтров в блоке управления
        const addToWhitelistBtnMgmt = document.getElementById('addToWhitelistBtnManagement');
        const addToBlacklistBtnMgmt = document.getElementById('addToBlacklistBtnManagement');
        const removeFromFiltersBtnMgmt = document.getElementById('removeFromFiltersBtnManagement');
        
        if (addToWhitelistBtnMgmt) {
            addToWhitelistBtnMgmt.onclick = () => this.addSelectedCoinToWhitelist();
        }
        if (addToBlacklistBtnMgmt) {
            addToBlacklistBtnMgmt.onclick = () => this.addSelectedCoinToBlacklist();
        }
        if (removeFromFiltersBtnMgmt) {
            removeFromFiltersBtnMgmt.onclick = () => this.removeSelectedCoinFromFilters();
        }
        
        // Умные фильтры для найденных монет
        const addFoundToWhitelist = document.getElementById('addFoundToWhitelist');
        const addFoundToBlacklist = document.getElementById('addFoundToBlacklist');
        
        if (addFoundToWhitelist) {
            addFoundToWhitelist.onclick = () => this.addFoundCoinsToWhitelist();
        }
        if (addFoundToBlacklist) {
            addFoundToBlacklist.onclick = () => this.addFoundCoinsToBlacklist();
        }
    }

    initializeRsiFilters() {
        document.querySelectorAll('.rsi-filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                // ✅ ИСПРАВЛЕНИЕ: Используем currentTarget вместо target
                // currentTarget = сама кнопка, target = может быть вложенный элемент (эмодзи, текст)
                const clickedBtn = e.currentTarget;
                const filter = clickedBtn.dataset.filter;
                
                console.log(`[BotsManager] 🔍 Клик на фильтр: ${filter}`);
                
                // Переключаем активный фильтр
                document.querySelectorAll('.rsi-filter-btn').forEach(b => b.classList.remove('active'));
                clickedBtn.classList.add('active');
                
                // Применяем фильтр
                this.applyRsiFilter(filter);
            });
        });
        
        // Обновляем текст кнопок с текущими значениями из конфига
        this.updateRsiFilterButtons();
    }
    
    updateRsiFilterButtons() {
        // Обновляем кнопки фильтров с текущими значениями RSI
        const buyFilterBtn = document.querySelector('.rsi-filter-btn[data-filter="buy-zone"]');
        const sellFilterBtn = document.querySelector('.rsi-filter-btn[data-filter="sell-zone"]');
        
        if (buyFilterBtn) {
            buyFilterBtn.innerHTML = `🟢 ≤${this.rsiLongThreshold}`;
        }
        
        if (sellFilterBtn) {
            sellFilterBtn.innerHTML = `🔴 ≥${this.rsiShortThreshold}`;
        }
        
        // Обновляем подписи тренд-фильтров с RSI значениями
        this.updateTrendFilterLabels();
        
        console.log(`[BotsManager] 🔄 Обновлены кнопки фильтров RSI: ≤${this.rsiLongThreshold}, ≥${this.rsiShortThreshold}`);
    }
    
    updateTrendFilterLabels() {
        // Проверяем, не обновлялись ли уже подписи
        if (this.trendLabelsUpdated) {
            console.log('[BotsManager] ⏭️ Подписи тренд-фильтров уже обновлены, пропускаем');
            return;
        }
        
        // Обновляем подписи тренд-фильтров с актуальными RSI значениями
        const avoidDownTrendLabels = document.querySelectorAll('[data-translate="avoid_down_trend_label"]');
        const avoidUpTrendLabels = document.querySelectorAll('[data-translate="avoid_up_trend_label"]');
        
        console.log(`[BotsManager] 🔄 Обновление подписей тренд-фильтров: RSI LONG=${this.rsiLongThreshold}, RSI SHORT=${this.rsiShortThreshold}`);
        
        avoidDownTrendLabels.forEach(label => {
            // Заменяем статическое значение 29 на актуальное из конфигурации
            const updatedText = `Избегать нисходящий тренд когда RSI < ${this.rsiLongThreshold}`;
            label.textContent = updatedText;
            console.log(`[BotsManager] ✅ Обновленный текст для DOWN тренда: "${updatedText}"`);
        });
        
        avoidUpTrendLabels.forEach(label => {
            // Заменяем статическое значение 71 на актуальное из конфигурации
            const updatedText = `Избегать восходящий тренд когда RSI > ${this.rsiShortThreshold}`;
            label.textContent = updatedText;
            console.log(`[BotsManager] ✅ Обновленный текст для UP тренда: "${updatedText}"`);
        });
        
        // Устанавливаем флаг, что подписи обновлены
        this.trendLabelsUpdated = true;
        console.log('[BotsManager] ✅ Подписи тренд-фильтров обновлены');
    }
    
    updateRsiThresholds(config) {
        // Обновляем внутренние пороговые значения RSI
        const oldLongThreshold = this.rsiLongThreshold;
        const oldShortThreshold = this.rsiShortThreshold;
        
        this.rsiLongThreshold = config.rsi_long_threshold || 29;
        this.rsiShortThreshold = config.rsi_short_threshold || 71;
        
        // Сбрасываем флаг обновления подписей при изменении порогов
        this.trendLabelsUpdated = false;
        
        console.log(`[BotsManager] 📊 Обновлены пороги RSI: ${oldLongThreshold}→${this.rsiLongThreshold}, ${oldShortThreshold}→${this.rsiShortThreshold}`);
        
        // Обновляем кнопки фильтров
        this.updateRsiFilterButtons();
        
        // Перепересчитываем классы для существующих монет
        this.refreshCoinsRsiClasses();
        
        // Обновляем счетчики
        this.updateCoinsCounter();
        
        // Если текущий фильтр buy-zone или sell-zone, переприменяем его
        if (this.currentRsiFilter === 'buy-zone' || this.currentRsiFilter === 'sell-zone') {
            this.applyRsiFilter(this.currentRsiFilter);
        }
    }
    
    refreshCoinsRsiClasses() {
        // Перепересчитываем RSI классы для всех монет в списке
        const coinItems = document.querySelectorAll('.coin-item');
        
        coinItems.forEach(item => {
            const symbol = item.dataset.symbol;
            const coinData = this.coinsRsiData.find(c => c.symbol === symbol);
            
            if (coinData) {
                // Удаляем старые классы
                item.classList.remove('buy-zone', 'sell-zone', 'enter-long', 'enter-short');
                
                // Добавляем новые классы на основе обновленных порогов
                const rsiClass = this.getRsiZoneClass(coinData.rsi6h);
                if (rsiClass) {
                    item.classList.add(rsiClass);
                }
                
                // Используем универсальную функцию для определения сигнала
                const effectiveSignal = this.getEffectiveSignal(coinData);
                
                if (effectiveSignal === 'ENTER_LONG') {
                    item.classList.add('enter-long');
                } else if (effectiveSignal === 'ENTER_SHORT') {
                    item.classList.add('enter-short');
                }
            }
        });
        
        console.log('[BotsManager] 🔄 Обновлены RSI и сигнальные классы для всех монет');
    }

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
    }

    async checkBotsService() {
        console.log('[BotsManager] 🔍 Проверка сервиса ботов...');
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/status`, {
                method: 'GET',
                timeout: 5000
            });
            
            if (response.ok) {
                const data = await response.json();
                this.serviceOnline = data.status === 'online';
                
                if (this.serviceOnline) {
                    console.log('[BotsManager] ✅ Сервис ботов онлайн');
                    this.updateServiceStatus('online', 'Сервис ботов онлайн');
                    await this.loadCoinsRsiData();
                } else {
                    console.warn('[BotsManager] ⚠️ Сервис ботов недоступен');
                    this.updateServiceStatus('offline', 'Сервис ботов недоступен');
                }
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Сервис ботов недоступен:', error);
            this.serviceOnline = false;
            this.updateServiceStatus('offline', 'Сервис ботов недоступен');
            this.showServiceUnavailable();
        }
    }

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
                text.textContent = message;
            }
        }
        
        if (statusDot) {
            statusDot.style.color = status === 'online' ? '#4caf50' : '#f44336';
        }
    }

    showServiceUnavailable() {
        const coinsListElement = document.getElementById('coinsRsiList');
        if (coinsListElement) {
            coinsListElement.innerHTML = `
                <div class="service-unavailable">
                    <h3>🚫 Сервис ботов недоступен</h3>
                    <p>Для работы с ботами запустите:</p>
                    <code>python bots.py</code>
                    <p>Сервис должен быть доступен на порту 5001</p>
                </div>
            `;
        }
    }

    async loadCoinsRsiData() {
        if (!this.serviceOnline) {
            console.warn('[BotsManager] ⚠️ Сервис не онлайн, пропускаем загрузку');
            return;
        }

        this.logDebug('[BotsManager] 📊 Загрузка данных RSI 6H...');
        
        // Сохраняем текущее состояние поиска
        const searchInput = document.getElementById('coinSearchInput');
        const currentSearchTerm = searchInput ? searchInput.value : '';
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/coins-with-rsi`);
            
            if (response.ok) {
            const data = await response.json();
            
            if (data.success) {
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
    }

    renderCoinsList() {
        const coinsListElement = document.getElementById('coinsRsiList');
        if (!coinsListElement) {
            console.warn('[BotsManager] ⚠️ Элемент coinsRsiList не найден');
            return;
        }

        this.logDebug(`[BotsManager] 🎨 Отрисовка списка монет: ${this.coinsRsiData.length} монет`);
        
        if (this.coinsRsiData.length === 0) {
            console.warn('[BotsManager] ⚠️ Нет данных RSI для отображения');
            coinsListElement.innerHTML = `
                <div class="loading-state">
                    <p>⏳ ${window.languageUtils.translate('loading_rsi_data')}</p>
                    <small>${window.languageUtils.translate('first_load_warning')}</small>
                </div>
            `;
            return;
        }
        
        const coinsHtml = this.coinsRsiData.map(coin => {
            const rsiClass = this.getRsiZoneClass(coin.rsi6h);
            const trendClass = coin.trend6h ? `trend-${coin.trend6h.toLowerCase()}` : 'trend-none';
            
            // Используем универсальную функцию для определения сигнала
            const effectiveSignal = this.getEffectiveSignal(coin);
            const signalClass = effectiveSignal === 'ENTER_LONG' ? 'enter-long' : 
                               effectiveSignal === 'ENTER_SHORT' ? 'enter-short' : '';
            
            // Проверяем, есть ли ручная позиция
            const isManualPosition = coin.manual_position || false;
            const manualClass = isManualPosition ? 'manual-position' : '';
            
            // Проверяем, зрелая ли монета
            const isMature = coin.is_mature || false;
            const matureClass = isMature ? 'mature-coin' : '';
            
            // Убраны спам логи для лучшей отладки
            
            return `
                <li class="coin-item ${rsiClass} ${trendClass} ${signalClass} ${manualClass} ${matureClass}" data-symbol="${coin.symbol}">
                    <div class="coin-item-content">
                        <div class="coin-header">
                            <span class="coin-symbol">${coin.symbol}</span>
                            <div class="coin-header-right">
                                ${isManualPosition ? '<span class="manual-position-indicator" title="Ручная позиция">✋</span>' : ''}
                                ${isMature ? '<span class="mature-coin-indicator" title="Зрелая монета">💎</span>' : ''}
                                ${this.generateWarningIndicator(coin)}
                                <span class="coin-rsi ${this.getRsiZoneClass(coin.rsi6h)}">${coin.rsi6h}</span>
                                <a href="${this.createTickerLink(coin.symbol)}" 
                               target="_blank" 
                               class="external-link" 
                               title="Открыть на бирже"
                               onclick="event.stopPropagation()">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                                    <polyline points="15 3 21 3 21 9"></polyline>
                                    <line x1="10" y1="14" x2="21" y2="3"></line>
                                </svg>
                            </a>
                        </div>
                        </div>
                        <div class="coin-details">
                            <span class="coin-trend ${coin.trend6h}">${coin.trend6h || 'NEUTRAL'}</span>
                            <span class="coin-price">$${coin.price?.toFixed(6) || '0'}</span>
                        </div>
                        <div class="coin-signal">
                            <small class="signal-text">${effectiveSignal || 'WAIT'}</small>
                            ${this.generateEnhancedSignalInfo(coin)}
                            ${this.generateTimeFilterInfo(coin)}
                            ${this.generateAntiPumpFilterInfo(coin)}
                        </div>
                    </div>
                </li>
            `;
        }).join('');

        coinsListElement.innerHTML = coinsHtml;

        // Добавляем обработчики кликов
        coinsListElement.querySelectorAll('.coin-item').forEach(item => {
            item.addEventListener('click', () => {
                const symbol = item.dataset.symbol;
                this.selectCoin(symbol);
            });
        });
        
        // Восстанавливаем текущий фильтр и состояние кнопок
        this.restoreFilterState();
        
        // Обновляем информацию о сделках для выбранной монеты
        if (this.selectedCoin && this.selectedCoin.symbol) {
            this.renderTradesInfo(this.selectedCoin.symbol);
        }
        
        // Обновляем индикаторы активных ботов в списке
        this.updateCoinsListWithBotStatus();
    }

    generateWarningIndicator(coin) {
        // Генерирует WARNING индикатор для монеты на основе улучшенного анализа RSI
        const enhancedRsi = coin.enhanced_rsi;
        
        if (!enhancedRsi || !enhancedRsi.enabled) {
            return '';
        }
        
        const warningType = enhancedRsi.warning_type;
        const warningMessage = enhancedRsi.warning_message;
        
        if (!warningType || warningType === 'ERROR') {
            return '';
        }
        
        let warningIcon = '';
        let warningClass = '';
        let warningTitle = warningMessage || '';
        
        switch (warningType) {
            case 'EXTREME_OVERSOLD_LONG':
                warningIcon = '⚠️';
                warningClass = 'warning-extreme-oversold';
                warningTitle = `ВНИМАНИЕ: ${warningMessage}. Требуются дополнительные подтверждения для LONG`;
                break;
            case 'EXTREME_OVERBOUGHT_LONG':
                warningIcon = '⚠️';
                warningClass = 'warning-extreme-overbought';
                warningTitle = `ВНИМАНИЕ: ${warningMessage}. Требуются дополнительные подтверждения для SHORT`;
                break;
            case 'OVERSOLD':
                warningIcon = '🟢';
                warningClass = 'warning-oversold';
                warningTitle = warningMessage;
                break;
            case 'OVERBOUGHT':
                warningIcon = '🔴';
                warningClass = 'warning-overbought';
                warningTitle = warningMessage;
                break;
            default:
                return '';
        }
        
        return `<span class="enhanced-warning ${warningClass}" title="${warningTitle}">${warningIcon}</span>`;
    }
    
    generateEnhancedSignalInfo(coin) {
        // Генерирует дополнительную информацию о сигнале
        const enhancedRsi = coin.enhanced_rsi;
        let infoElements = [];
        
        // console.log(`[DEBUG] ${coin.symbol}: enhanced_rsi =`, enhancedRsi);
        
        // СТОХАСТИК - показываем ВСЕГДА если есть данные!
        let stochK = null;
        let stochD = null;
        
        // Проверяем разные источники данных стохастика
        if (coin.stoch_rsi_k !== undefined && coin.stoch_rsi_k !== null) {
            stochK = coin.stoch_rsi_k;
            stochD = coin.stoch_rsi_d || 0;
        } else if (enhancedRsi && enhancedRsi.confirmations) {
            stochK = enhancedRsi.confirmations.stoch_rsi_k;
            stochD = enhancedRsi.confirmations.stoch_rsi_d || 0;
        }
        
        if (stochK !== null && stochK !== undefined) {
            let stochIcon, stochStatus, stochDescription;
            
            // Определяем статус и описание стохастика
            if (stochK < 20) {
                stochIcon = '⬇️';
                stochStatus = 'OVERSOLD';
                stochDescription = 'Перепроданность: %K ниже 20 - возможен разворот вверх';
            } else if (stochK > 80) {
                stochIcon = '⬆️';
                stochStatus = 'OVERBOUGHT';
                stochDescription = 'Перекупленность: %K выше 80 - возможен разворот вниз';
            } else {
                stochIcon = '➡️';
                stochStatus = 'NEUTRAL';
                stochDescription = 'Нейтральная зона: %K между 20-80 - тренд продолжается';
            }
            
            // Добавляем информацию о пересечении %K и %D
            let crossoverInfo = '';
            if (stochK > stochD) {
                crossoverInfo = ' (%K выше %D - бычий сигнал)';
            } else if (stochK < stochD) {
                crossoverInfo = ' (%K ниже %D - медвежий сигнал)';
            } else {
                crossoverInfo = ' (%K = %D - нейтрально)';
            }
            
            const fullDescription = `${stochDescription}${crossoverInfo}`;
            
            // console.log(`[DEBUG] ${coin.symbol}: ГЕНЕРИРУЮ СТОХАСТИК %K=${stochK}, %D=${stochD}, статус=${stochStatus}, icon=${stochIcon}`);
            infoElements.push(`<span class="confirmation-stoch" title="${fullDescription}">${stochIcon}</span>`);
        } else {
            // console.log(`[DEBUG] ${coin.symbol}: НЕТ СТОХАСТИКА - stoch_rsi_k=${coin.stoch_rsi_k}, enhanced_rsi=${!!enhancedRsi}`);
        }
        
        // Enhanced RSI данные - только если включен
        if (enhancedRsi && enhancedRsi.enabled) {
        const extremeDuration = enhancedRsi.extreme_duration;
        const confirmations = enhancedRsi.confirmations || {};
        
        // Показываем продолжительность в экстремальной зоне
        if (extremeDuration > 0) {
            infoElements.push(`<span class="extreme-duration" title="Время в экстремальной зоне">${extremeDuration}🕐</span>`);
        }
        
        // Показываем подтверждения
        if (confirmations.volume) {
            infoElements.push(`<span class="confirmation-volume" title="Подтверждение объемом">📊</span>`);
        }
        
        if (confirmations.divergence) {
            const divIcon = confirmations.divergence === 'BULLISH_DIVERGENCE' ? '📈' : '📉';
            infoElements.push(`<span class="confirmation-divergence" title="Дивергенция: ${confirmations.divergence}">${divIcon}</span>`);
        }
        }
        
        if (infoElements.length > 0) {
            return `<div class="enhanced-info">${infoElements.join('')}</div>`;
        }
        
        return '';
    }
    
    generateTimeFilterInfo(coin) {
        // Генерирует информацию о временном фильтре RSI
        const timeFilterInfo = coin.time_filter_info;
        
        if (!timeFilterInfo) {
            // console.log(`[DEBUG] ${coin.symbol}: НЕТ time_filter_info`);
            return '';
        }
        
        // console.log(`[DEBUG] ${coin.symbol}: time_filter_info =`, timeFilterInfo);
        
        const isBlocked = timeFilterInfo.blocked;
        const reason = timeFilterInfo.reason;
        const lastExtremeCandlesAgo = timeFilterInfo.last_extreme_candles_ago;
        const calmCandles = timeFilterInfo.calm_candles;
        
        let icon = '';
        let className = '';
        let title = '';
        
        if (isBlocked) {
            // Фильтр блокирует вход
            icon = '⏰';
            className = 'time-filter-blocked';
            title = `Временной фильтр блокирует: ${reason}`;
        } else {
            // Фильтр пройден, показываем информацию
            icon = '⏱️';
            className = 'time-filter-active';
            title = `Временной фильтр: ${reason}`;
            if (lastExtremeCandlesAgo !== null) {
                title += ` (${lastExtremeCandlesAgo} свечей назад)`;
            }
            if (calmCandles !== null) {
                title += ` (${calmCandles} спокойных свечей)`;
            }
        }
        
        console.log(`[DEBUG] ${coin.symbol}: isBlocked=${isBlocked}, reason="${reason}", icon="${icon}", title="${title}"`);
        
        if (icon && title) {
            console.log(`[DEBUG] ${coin.symbol}: ГЕНЕРИРУЮ ИКОНКУ RSI Time Filter`);
            return `<div class="time-filter-info ${className}" title="${title}">${icon}</div>`;
        }
        
        console.log(`[DEBUG] ${coin.symbol}: НЕ ГЕНЕРИРУЮ ИКОНКУ - icon="${icon}", title="${title}"`);
        return '';
    }
    
    generateExitScamFilterInfo(coin) {
        // Генерирует информацию об ExitScam фильтре
        const exitScamInfo = coin.exit_scam_info;
        
        if (!exitScamInfo) {
            return '';
        }
        
        const isBlocked = exitScamInfo.blocked;
        const reason = exitScamInfo.reason;
        
        let icon = '';
        let className = '';
        let title = '';
        
        if (isBlocked) {
            // Фильтр блокирует вход
            icon = '🛡️';
            className = 'exit-scam-blocked';
            title = `ExitScam фильтр блокирует: ${reason}`;
        } else {
            // Фильтр пройден
            icon = '✅';
            className = 'exit-scam-passed';
            title = `ExitScam фильтр: ${reason}`;
        }
        
        if (icon && title) {
            return `<div class="exit-scam-info ${className}" title="${title}">${icon}</div>`;
        }
        
        return '';
    }
    
    // Алиас для обратной совместимости
    generateAntiPumpFilterInfo(coin) {
        return this.generateExitScamFilterInfo(coin);
    }

    getRsiZoneClass(rsi) {
        if (rsi <= this.rsiLongThreshold) return 'buy-zone';
        if (rsi >= this.rsiShortThreshold) return 'sell-zone';
        return '';
    }

    createTickerLink(symbol) {
        try {
            // Получаем текущую биржу из exchangeManager
            let currentExchange = 'bybit'; // значение по умолчанию
            
            // Проверяем наличие exchangeManager и его метода
            const exchangeManager = window.app?.exchangeManager;
            if (exchangeManager && typeof exchangeManager.getSelectedExchange === 'function') {
                currentExchange = exchangeManager.getSelectedExchange();
            }
            
            return this.getExchangeLink(symbol, currentExchange);
        } catch (error) {
            console.warn('Error in createTickerLink:', error);
            return this.getExchangeLink(symbol, 'bybit');
        }
    }

    getExchangeLink(symbol, exchange = 'bybit') {
        // Удаляем USDT из символа для корректной ссылки
        const cleanSymbol = symbol.replace('USDT', '');
        
        // Создаем ссылки в зависимости от биржи
        switch (exchange.toLowerCase()) {
            case 'binance':
                return `https://www.binance.com/ru/futures/${cleanSymbol}USDT`;
            case 'bybit':
                return `https://www.bybit.com/trade/usdt/${cleanSymbol}USDT`;
            case 'okx':
                return `https://www.okx.com/ru/trade-swap/${cleanSymbol.toLowerCase()}-usdt-swap`;
            default:
                return `https://www.bybit.com/trade/usdt/${cleanSymbol}USDT`; // По умолчанию Bybit
        }
    }

        updateCoinsCounter() {
        // Обновляем счетчики для новых фильтров сигналов
        this.updateSignalCounters();
        
        // Обновляем счетчик ручных позиций
        this.updateManualPositionCounter();
    }
    
    /**
     * Обновляет счетчик ручных позиций
     */
    updateManualPositionCounter() {
        const manualCountElement = document.getElementById('manualCount');
        if (manualCountElement) {
            const manualCount = this.coinsRsiData.filter(coin => coin.manual_position).length;
            manualCountElement.textContent = `(${manualCount})`;
        }
    }
    
    /**
     * Универсальная функция для определения эффективного сигнала монеты
     * Используется и автоботом, и фильтрами для единообразия
     * @param {Object} coin - Данные монеты
     * @returns {string} - Эффективный сигнал (ENTER_LONG, ENTER_SHORT, WAIT)
     */
    getEffectiveSignal(coin) {
        // Если API уже предоставил эффективный сигнал, используем его
        if (coin.effective_signal) {
            return coin.effective_signal;
        }
        
        // Иначе вычисляем самостоятельно (fallback для совместимости)
        if (coin.enhanced_rsi && coin.enhanced_rsi.enabled && coin.enhanced_rsi.enhanced_signal) {
            return coin.enhanced_rsi.enhanced_signal;
        }
        
        // Иначе используем стандартный сигнал
        return coin.signal || 'WAIT';
    }

    updateSignalCounters() {
        // Подсчитываем все категории
        const allCount = this.coinsRsiData.length;
        const longCount = this.coinsRsiData.filter(coin => this.getEffectiveSignal(coin) === 'ENTER_LONG').length;
        const shortCount = this.coinsRsiData.filter(coin => this.getEffectiveSignal(coin) === 'ENTER_SHORT').length;
        const buyZoneCount = this.coinsRsiData.filter(coin => coin.rsi6h && coin.rsi6h <= 29).length;
        const sellZoneCount = this.coinsRsiData.filter(coin => coin.rsi6h && coin.rsi6h >= 71).length;
        const trendUpCount = this.coinsRsiData.filter(coin => coin.trend6h === 'UP').length;
        const trendDownCount = this.coinsRsiData.filter(coin => coin.trend6h === 'DOWN').length;
        const manualPositionCount = this.coinsRsiData.filter(coin => coin.manual_position === true).length;
        
        // Обновляем счетчики в HTML (фильтры)
        const allCountEl = document.getElementById('filterAllCount');
        const buyZoneCountEl = document.getElementById('filterBuyZoneCount');
        const sellZoneCountEl = document.getElementById('filterSellZoneCount');
        
        // Если элементы не найдены, создаем их динамически
        if (!buyZoneCountEl || !sellZoneCountEl) {
            // Попробуем найти кнопки фильтров и добавить элементы динамически
            const buyFilterBtn = document.querySelector('button[data-filter="buy-zone"]');
            const sellFilterBtn = document.querySelector('button[data-filter="sell-zone"]');
            
            if (buyFilterBtn && !buyFilterBtn.querySelector('#filterBuyZoneCount')) {
                const buySpan = document.createElement('span');
                buySpan.id = 'filterBuyZoneCount';
                buySpan.textContent = ` (${buyZoneCount})`;
                buyFilterBtn.appendChild(buySpan);
            }
            
            if (sellFilterBtn && !sellFilterBtn.querySelector('#filterSellZoneCount')) {
                const sellSpan = document.createElement('span');
                sellSpan.id = 'filterSellZoneCount';
                sellSpan.textContent = ` (${sellZoneCount})`;
                sellFilterBtn.appendChild(sellSpan);
            }
        }
        
        const trendUpCountEl = document.getElementById('filterTrendUpCount');
        const trendDownCountEl = document.getElementById('filterTrendDownCount');
        const longCountEl = document.getElementById('filterLongCount');
        const shortCountEl = document.getElementById('filterShortCount');
        const manualCountEl = document.getElementById('manualCount');
        
        
        // Обновляем счетчики фильтров
        if (allCountEl) allCountEl.textContent = allCount;
        
        if (buyZoneCountEl) buyZoneCountEl.textContent = ` (${buyZoneCount})`;
        if (sellZoneCountEl) sellZoneCountEl.textContent = ` (${sellZoneCount})`;
        if (trendUpCountEl) trendUpCountEl.textContent = trendUpCount;
        if (trendDownCountEl) trendDownCountEl.textContent = trendDownCount;
        if (longCountEl) longCountEl.textContent = longCount;
        if (shortCountEl) shortCountEl.textContent = shortCount;
        if (manualCountEl) manualCountEl.textContent = `(${manualPositionCount})`;
        
        this.logDebug(`[BotsManager] 📊 Счетчики фильтров: ALL=${allCount}, BUY=${buyZoneCount}, SELL=${sellZoneCount}, UP=${trendUpCount}, DOWN=${trendDownCount}, LONG=${longCount}, SHORT=${shortCount}, MANUAL=${manualPositionCount}`);
    }

    selectCoin(symbol) {
        this.logDebug('[BotsManager] 🎯 Выбрана монета:', symbol);
        this.logDebug('[BotsManager] 🔍 Доступные монеты в RSI данных:', this.coinsRsiData.length);
        this.logDebug('[BotsManager] 🔍 Первые 5 монет:', this.coinsRsiData.slice(0, 5).map(c => c.symbol));
        
        // Находим данные монеты
        const coinData = this.coinsRsiData.find(coin => coin.symbol === symbol);
        this.logDebug('[BotsManager] 🔍 Найденные данные монеты:', coinData);
        
        if (!coinData) {
            console.warn('[BotsManager] ⚠️ Монета не найдена в RSI данных:', symbol);
            return;
        }

        this.selectedCoin = coinData;
        
        // Обновляем выделение в списке
        document.querySelectorAll('.coin-item').forEach(item => {
            item.classList.toggle('selected', item.dataset.symbol === symbol);
        });
        
        // Показываем интерфейс управления ботом
        this.showBotControlInterface();
        
        // Обновляем информацию о монете
        this.updateCoinInfo();
        
        // Обновляем статус и кнопки бота для выбранной монеты
        this.updateBotStatus();
        this.updateBotControlButtons();
        
        // Загружаем индивидуальные настройки для выбранной монеты
        this.loadAndApplyIndividualSettings(symbol);
        
        // Показываем блок фильтров и обновляем статус
        this.showFilterControls(symbol);
        this.updateFilterStatus(symbol);
        
        // Рендерим информацию о сделках
        this.renderTradesInfo(symbol);
    }

    showBotControlInterface() {
        console.log('[BotsManager] 🎨 Показ интерфейса управления ботом...');
        
        const promptElement = document.getElementById('selectCoinPrompt');
        const controlElement = document.getElementById('botControlInterface');
        const tradesSection = document.getElementById('tradesInfoSection');
        
        console.log('[BotsManager] 🔍 Найденные элементы:', {
            promptElement: !!promptElement,
            controlElement: !!controlElement,
            tradesSection: !!tradesSection
        });
        
        // Проверяем родительский элемент
        const parentPanel = document.querySelector('.bot-control-panel');
        console.log('[BotsManager] 🔍 Родительская панель:', {
            exists: !!parentPanel,
            display: parentPanel ? window.getComputedStyle(parentPanel).display : 'N/A',
            visibility: parentPanel ? window.getComputedStyle(parentPanel).visibility : 'N/A',
            height: parentPanel ? window.getComputedStyle(parentPanel).height : 'N/A',
            clientHeight: parentPanel ? parentPanel.clientHeight : 'N/A',
            offsetHeight: parentPanel ? parentPanel.offsetHeight : 'N/A'
        });
        
        if (promptElement) {
            promptElement.style.display = 'none';
            console.log('[BotsManager] ✅ Скрыт prompt элемент');
        } else {
            console.warn('[BotsManager] ⚠️ Элемент selectCoinPrompt не найден');
        }
        
        if (controlElement) {
            controlElement.style.display = 'block';
            console.log('[BotsManager] ✅ Показан control элемент');
            console.log('[BotsManager] 🔍 Стили control элемента:', {
                display: controlElement.style.display,
                visibility: window.getComputedStyle(controlElement).visibility,
                opacity: window.getComputedStyle(controlElement).opacity,
                position: window.getComputedStyle(controlElement).position,
                zIndex: window.getComputedStyle(controlElement).zIndex,
                height: window.getComputedStyle(controlElement).height,
                minHeight: window.getComputedStyle(controlElement).minHeight,
                width: window.getComputedStyle(controlElement).width,
                clientHeight: controlElement.clientHeight,
                offsetHeight: controlElement.offsetHeight
            });
            
            // Проверяем содержимое элемента
            console.log('[BotsManager] 🔍 Содержимое control элемента:', {
                innerHTML: controlElement.innerHTML.substring(0, 200) + '...',
                childrenCount: controlElement.children.length,
                firstChild: controlElement.firstChild ? controlElement.firstChild.tagName : 'null'
            });
        } else {
            console.warn('[BotsManager] ⚠️ Элемент botControlInterface не найден');
        }
        
        if (tradesSection) {
            tradesSection.style.display = 'block';
            console.log('[BotsManager] ✅ Показана trades секция');
        } else {
            console.warn('[BotsManager] ⚠️ Элемент tradesInfoSection не найден');
        }
    }

    updateCoinInfo() {
        if (!this.selectedCoin) return;

        const coin = this.selectedCoin;
        console.log('[BotsManager] 🔄 Обновление информации о монете:', coin);
        
        // Обновляем основную информацию
        const symbolElement = document.getElementById('selectedCoinSymbol');
        const priceElement = document.getElementById('selectedCoinPrice');
        const rsiElement = document.getElementById('selectedCoinRSI');
        const trendElement = document.getElementById('selectedCoinTrend');
        const emaElement = document.getElementById('selectedCoinEMA');
        const zoneElement = document.getElementById('selectedCoinZone');
        const signalElement = document.getElementById('selectedCoinSignal');
        const changeElement = document.getElementById('selectedCoinChange');

        console.log('[BotsManager] 🔍 Найденные элементы:', {
            symbolElement: !!symbolElement,
            priceElement: !!priceElement,
            rsiElement: !!rsiElement,
            trendElement: !!trendElement,
            emaElement: !!emaElement,
            zoneElement: !!zoneElement,
            signalElement: !!signalElement,
            changeElement: !!changeElement
        });

        if (symbolElement) {
            const exchangeUrl = this.getExchangeLink(coin.symbol, 'bybit');
            symbolElement.innerHTML = `
                🪙 ${coin.symbol} 
                <a href="${exchangeUrl}" target="_blank" class="exchange-link" title="Открыть на Bybit">
                    🔗
                </a>
            `;
            console.log('[BotsManager] ✅ Символ обновлен:', coin.symbol);
        }
        
        // Используем правильные поля из RSI данных
        if (priceElement) {
            const price = coin.current_price || coin.mark_price || coin.last_price || coin.price || 0;
            priceElement.textContent = `$${price.toFixed(6)}`;
            console.log('[BotsManager] ✅ Цена обновлена:', price);
        }
        
        if (rsiElement) {
            const rsi = coin.enhanced_rsi?.rsi_6h || coin.rsi6h || '-';
            rsiElement.textContent = rsi;
            rsiElement.className = `value rsi-indicator ${this.getRsiZoneClass(rsi)}`;
            console.log('[BotsManager] ✅ RSI обновлен:', rsi);
        }
        
        if (trendElement) {
            const trend = coin.trend6h || 'NEUTRAL';
            trendElement.textContent = trend;
            trendElement.className = `value trend-indicator ${trend}`;
            console.log('[BotsManager] ✅ Тренд обновлен:', trend);
        }
        
        if (emaElement) {
            const emaText = coin.ema_periods ? `EMA(${coin.ema_periods.ema_short},${coin.ema_periods.ema_long})` : '-';
            emaElement.textContent = emaText;
            emaElement.className = 'value ema-indicator';
            console.log('[BotsManager] ✅ EMA обновлен:', emaText);
        }
        
        if (zoneElement) {
            const zone = coin.rsi_zone || 'NEUTRAL';
            zoneElement.textContent = zone;
            console.log('[BotsManager] ✅ Зона обновлена:', zone);
        }
        
        if (signalElement) {
            const signal = coin.effective_signal || coin.signal || 'WAIT';
            signalElement.textContent = signal;
            signalElement.className = `value signal-indicator ${signal}`;
            console.log('[BotsManager] ✅ Сигнал обновлен:', signal);
        }
        
        if (changeElement) {
            const change = coin.change24h || 0;
            changeElement.textContent = `${change > 0 ? '+' : ''}${change}%`;
            changeElement.style.color = change >= 0 ? 'var(--green-color)' : 'var(--red-color)';
            console.log('[BotsManager] ✅ Изменение обновлено:', change);
        }
        
        console.log('[BotsManager] ✅ Информация о монете обновлена полностью');
        
        // Обновляем активные иконки монеты
        this.updateActiveCoinIcons();
        
        // ПРИНУДИТЕЛЬНО ПОКАЗЫВАЕМ СТАТУС БОТА
        setTimeout(() => {
            const botStatusItem = document.getElementById('botStatusItem');
            if (botStatusItem) {
                botStatusItem.style.display = 'flex';
                console.log('[BotsManager] 🔧 ПРИНУДИТЕЛЬНО ПОКАЗАН СТАТУС БОТА');
            }
        }, 100);
    }
    
    updateActiveCoinIcons() {
        if (!this.selectedCoin) return;
        
        const coin = this.selectedCoin;
        const activeStatusData = {};
        
        // Тренд убираем - он уже показан выше в ТРЕНД 6Н
        
        // Зону RSI убираем - она уже показана выше в ЗОНА RSI
        
        // 2. Статус бота - проверяем активные боты
        let botStatus = 'Нет бота';
        if (this.activeBots && this.activeBots.length > 0) {
            const bot = this.activeBots.find(bot => bot.symbol === coin.symbol);
            if (bot) {
                // Используем bot_status из API, если есть
                if (bot.bot_status) {
                    botStatus = bot.bot_status;
                } else if (bot.status === 'running') {
                    botStatus = 'Активен';
                } else if (bot.status === 'waiting') {
                    botStatus = 'Ожидание сигнала';
                } else if (bot.status === 'in_position_long') {
                    botStatus = 'Активен';
                } else if (bot.status === 'in_position_short') {
                    botStatus = 'Активен';
                } else {
                    botStatus = bot.status || 'Нет бота';
                }
            }
        }
        activeStatusData.bot = botStatus;
        
        // 3. ФИЛЬТРЫ - проверяем ВСЕ возможные поля
        
        // Подтверждение объемом (Volume Confirmation) - проверяем разные поля
        if (coin.volume_confirmation && coin.volume_confirmation !== 'NONE' && coin.volume_confirmation !== null) {
            activeStatusData.volume_confirmation = coin.volume_confirmation;
        } else if (coin.volume_confirmation_status && coin.volume_confirmation_status !== 'NONE') {
            activeStatusData.volume_confirmation = coin.volume_confirmation_status;
        } else if (coin.volume_status && coin.volume_status !== 'NONE') {
            activeStatusData.volume_confirmation = coin.volume_status;
        }
        
        // Стохастик (Stochastic) - проверяем разные поля
        let stochValue = null;
        if (coin.stochastic_rsi && coin.stochastic_rsi !== 'NONE' && coin.stochastic_rsi !== null) {
            stochValue = coin.stochastic_rsi;
        } else if (coin.stochastic_status && coin.stochastic_status !== 'NONE') {
            stochValue = coin.stochastic_status;
        } else if (coin.stochastic && coin.stochastic !== 'NONE') {
            stochValue = coin.stochastic;
        } else if (coin.stoch_rsi_k !== undefined && coin.stoch_rsi_k !== null) {
            // Используем числовые значения стохастика с подробным описанием
            const stochK = coin.stoch_rsi_k;
            const stochD = coin.stoch_rsi_d || 0;
            let stochStatus = '';
            let crossoverInfo = '';
            
            if (stochK < 20) {
                stochStatus = 'OVERSOLD';
                crossoverInfo = stochK > stochD ? ' (%K выше %D - бычий сигнал)' : ' (%K ниже %D - медвежий сигнал)';
                stochValue = `<span style="color: #44ff44;">Перепроданность: %K=${stochK.toFixed(1)} (ниже 20) - возможен разворот цены вверх.</span><br><span style="color: ${stochK > stochD ? '#44ff44' : '#ff4444'};">${stochK > stochD ? 'Бычий сигнал' : 'Медвежий сигнал'}: %D=${stochD.toFixed(1)} (${stochK > stochD ? '%K выше %D' : '%K ниже %D'})</span>`;
            } else if (stochK > 80) {
                stochStatus = 'OVERBOUGHT';
                crossoverInfo = stochK > stochD ? ' (%K выше %D - бычий сигнал)' : ' (%K ниже %D - медвежий сигнал)';
                stochValue = `<span style="color: #ff4444;">Перекупленность: %K=${stochK.toFixed(1)} (выше 80) - возможен разворот цены вниз.</span><br><span style="color: ${stochK > stochD ? '#44ff44' : '#ff4444'};">${stochK > stochD ? 'Бычий сигнал' : 'Медвежий сигнал'}: %D=${stochD.toFixed(1)} (${stochK > stochD ? '%K выше %D' : '%K ниже %D'})</span>`;
            } else {
                stochStatus = 'NEUTRAL';
                crossoverInfo = stochK > stochD ? ' (%K выше %D - бычий сигнал)' : ' (%K ниже %D - медвежий сигнал)';
                stochValue = `<span style="color: #ffff44;">Нейтральная зона: %K=${stochK.toFixed(1)} (между 20-80) - тренд продолжается.</span><br><span style="color: ${stochK > stochD ? '#44ff44' : '#ff4444'};">${stochK > stochD ? 'Бычий сигнал' : 'Медвежий сигнал'}: %D=${stochD.toFixed(1)} (${stochK > stochD ? '%K выше %D' : '%K ниже %D'})</span>`;
            }
        } else if (coin.enhanced_rsi && coin.enhanced_rsi.confirmations) {
            const stochK = coin.enhanced_rsi.confirmations.stoch_rsi_k;
            const stochD = coin.enhanced_rsi.confirmations.stoch_rsi_d || 0;
            if (stochK !== undefined && stochK !== null) {
                let stochStatus = '';
                let crossoverInfo = '';
                
                if (stochK < 20) {
                    stochStatus = 'OVERSOLD';
                    crossoverInfo = stochK > stochD ? ' (%K выше %D - бычий сигнал)' : ' (%K ниже %D - медвежий сигнал)';
                    stochValue = `<span style="color: #44ff44;">Перепроданность: %K=${stochK.toFixed(1)} (ниже 20) - возможен разворот цены вверх.</span><br><span style="color: ${stochK > stochD ? '#44ff44' : '#ff4444'};">${stochK > stochD ? 'Бычий сигнал' : 'Медвежий сигнал'}: %D=${stochD.toFixed(1)} (${stochK > stochD ? '%K выше %D' : '%K ниже %D'})</span>`;
                } else if (stochK > 80) {
                    stochStatus = 'OVERBOUGHT';
                    crossoverInfo = stochK > stochD ? ' (%K выше %D - бычий сигнал)' : ' (%K ниже %D - медвежий сигнал)';
                    stochValue = `<span style="color: #ff4444;">Перекупленность: %K=${stochK.toFixed(1)} (выше 80) - возможен разворот цены вниз.</span><br><span style="color: ${stochK > stochD ? '#44ff44' : '#ff4444'};">${stochK > stochD ? 'Бычий сигнал' : 'Медвежий сигнал'}: %D=${stochD.toFixed(1)} (${stochK > stochD ? '%K выше %D' : '%K ниже %D'})</span>`;
                } else {
                    stochStatus = 'NEUTRAL';
                    crossoverInfo = stochK > stochD ? ' (%K выше %D - бычий сигнал)' : ' (%K ниже %D - медвежий сигнал)';
                    stochValue = `<span style="color: #ffff44;">Нейтральная зона: %K=${stochK.toFixed(1)} (между 20-80) - тренд продолжается.</span><br><span style="color: ${stochK > stochD ? '#44ff44' : '#ff4444'};">${stochK > stochD ? 'Бычий сигнал' : 'Медвежий сигнал'}: %D=${stochD.toFixed(1)} (${stochK > stochD ? '%K выше %D' : '%K ниже %D'})</span>`;
                }
            }
        }
        
        if (stochValue) {
            activeStatusData.stochastic_rsi = stochValue;
        }
        
        // ExitScam защита (ExitScam Protection) - проверяем разные поля
        if (coin.exit_scam_status && coin.exit_scam_status !== 'NONE' && coin.exit_scam_status !== null) {
            activeStatusData.exit_scam = coin.exit_scam_status;
        } else if (coin.exit_scam && coin.exit_scam !== 'NONE') {
            activeStatusData.exit_scam = coin.exit_scam;
        } else if (coin.scam_status && coin.scam_status !== 'NONE') {
            activeStatusData.exit_scam = coin.scam_status;
        }
        
        // RSI Time Filter - проверяем разные поля
        if (coin.rsi_time_filter && coin.rsi_time_filter !== 'NONE' && coin.rsi_time_filter !== null) {
            activeStatusData.rsi_time_filter = coin.rsi_time_filter;
        } else if (coin.time_filter && coin.time_filter !== 'NONE') {
            activeStatusData.rsi_time_filter = coin.time_filter;
        } else if (coin.rsi_time_status && coin.rsi_time_status !== 'NONE') {
            activeStatusData.rsi_time_filter = coin.rsi_time_status;
        }
        
        // Enhanced RSI Warning (если есть)
        if (coin.enhanced_rsi?.warning_type && coin.enhanced_rsi.warning_type !== 'ERROR') {
            activeStatusData.enhanced_warning = coin.enhanced_rsi.warning_type;
        }
        
        // Manual Position (если есть)
        if (coin.is_manual_position) {
            activeStatusData.manual_position = 'MANUAL';
        }
        
        // Maturity (зрелость монеты)
        if (coin.is_mature === true) {
            const actualCandles = coin.candles_count || 'N/A';
            const minCandles = this.autoBotConfig?.min_candles_for_maturity || 400;
            activeStatusData.maturity = `Зрелая монета: ${actualCandles} > ${minCandles} свечей - достаточно истории для торговли`;
        } else if (coin.is_mature === false) {
            const minCandles = this.autoBotConfig?.min_candles_for_maturity || 400;
            activeStatusData.maturity = `Незрелая монета, менее ${minCandles} свечей - недостаточно истории`;
        }
        
        console.log('[BotsManager] 🎯 Обновление активных иконок:', activeStatusData);
        console.log('[BotsManager] 🔍 ВСЕ ДАННЫЕ МОНЕТЫ:', coin);
        
        // Обновляем иконки в верхнем блоке
        this.updateCoinStatusIcons(activeStatusData);
        
        // ОТЛАДКА: Принудительно показываем ВСЕ фильтры для тестирования
        this.forceShowAllFilters();
    }
    
    getRsiZone(rsi) {
        if (rsi === '-' || rsi === null || rsi === undefined) return 'NEUTRAL';
        if (rsi <= 30) return 'OVERSOLD';
        if (rsi >= 70) return 'OVERBOUGHT';
        return 'NEUTRAL';
    }
    
    updateCoinStatusIcons(activeStatusData) {
        // Обновляем основные иконки
        this.updateStatusIcon('rsiIcon', activeStatusData.zone);
        this.updateStatusIcon('trendIcon', activeStatusData.trend);
        this.updateStatusIcon('zoneIcon', activeStatusData.zone);
        this.updateStatusIcon('signalIcon', activeStatusData.signal);
        
        // Обновляем дополнительные фильтры
        this.updateFilterItem('volumeConfirmationItem', 'selectedCoinVolumeConfirmation', 'volumeConfirmationIcon', 
                             activeStatusData.volume_confirmation, 'Подтверждение объемом');
        
        this.updateFilterItem('stochasticItem', 'selectedCoinStochastic', 'stochasticIcon', 
                             activeStatusData.stochastic_rsi, 'Стохастик');
        
        this.updateFilterItem('exitScamItem', 'selectedCoinExitScam', 'exitScamIcon', 
                             activeStatusData.exit_scam, 'ExitScam защита');
        
        this.updateFilterItem('rsiTimeFilterItem', 'selectedCoinRsiTimeFilter', 'rsiTimeFilterIcon', 
                             activeStatusData.rsi_time_filter, 'RSI Time Filter');
        
        this.updateFilterItem('maturityDiamondItem', 'selectedCoinMaturityDiamond', 'maturityDiamondIcon', 
                             activeStatusData.maturity, 'Зрелость монеты');
        
        this.updateFilterItem('botStatusItem', 'selectedCoinBotStatus', 'botStatusIcon', 
                             activeStatusData.bot, 'Статус бота');
    }
    
    updateStatusIcon(iconId, statusValue) {
        const iconElement = document.getElementById(iconId);
        if (iconElement && statusValue) {
            const icon = this.getStatusIcon('zone', statusValue); // Используем зону как базовую
            iconElement.textContent = icon;
            iconElement.style.display = 'inline';
        } else if (iconElement) {
            iconElement.style.display = 'none';
        }
    }
    
    updateFilterItem(itemId, valueId, iconId, statusValue, label) {
        const itemElement = document.getElementById(itemId);
        const valueElement = document.getElementById(valueId);
        const iconElement = document.getElementById(iconId);
        
        if (itemElement && valueElement && iconElement) {
            if (statusValue && statusValue !== 'NONE' && statusValue !== null && statusValue !== undefined) {
                itemElement.style.display = 'flex';
                valueElement.textContent = statusValue;
                
                // Получаем правильную иконку для каждого типа статуса
                let icon = '❓';
                let description = '';
                
                if (label === 'Подтверждение объемом') {
                    if (statusValue.includes('CONFIRMED')) { icon = '📊'; description = 'Объем подтвержден'; }
                    else if (statusValue.includes('NOT_CONFIRMED')) { icon = '❌'; description = 'Объем не подтвержден'; }
                    else if (statusValue.includes('LOW_VOLUME')) { icon = '⚠️'; description = 'Низкий объем'; }
                    else if (statusValue.includes('HIGH_VOLUME')) { icon = '📈'; description = 'Высокий объем'; }
                }
                else if (label === 'Стохастик') {
                    // Специальная обработка для стохастика с HTML и цветами
                    if (statusValue.includes('<br>') || statusValue.includes('<span')) {
                        // Это HTML контент с цветовым кодированием
                        valueElement.innerHTML = statusValue;
                        return; // Выходим рано для HTML контента
                    }
                    
                    if (statusValue.includes('OVERSOLD')) { icon = '🔴'; description = 'Stochastic перепродан'; }
                    else if (statusValue.includes('OVERBOUGHT')) { icon = '🟢'; description = 'Stochastic перекуплен'; }
                    else if (statusValue.includes('NEUTRAL')) { icon = '🟡'; description = 'Stochastic нейтральный'; }
                    else if (statusValue.includes('BULLISH')) { icon = '📈'; description = 'Stochastic бычий сигнал'; }
                    else if (statusValue.includes('BEARISH')) { icon = '📉'; description = 'Stochastic медвежий сигнал'; }
                }
                else if (label === 'ExitScam защита') {
                    // Специальная обработка для ExitScam с цветами
                    if (statusValue.includes('Блокирует:')) {
                        valueElement.innerHTML = `<span style="color: #ff4444;">${statusValue}</span>`;
                        return; // Выходим рано для цветного контента
                    } else if (statusValue.includes('Безопасно:')) {
                        valueElement.innerHTML = `<span style="color: #44ff44;">${statusValue}</span>`;
                        return; // Выходим рано для цветного контента
                    }
                    
                    if (statusValue.includes('SAFE')) { icon = '🛡️'; description = 'ExitScam: Безопасно'; }
                    else if (statusValue.includes('RISK')) { icon = '⚠️'; description = 'ExitScam: Риск обнаружен'; }
                    else if (statusValue.includes('SCAM')) { icon = '🚨'; description = 'ExitScam: Возможный скам'; }
                    else if (statusValue.includes('CHECKING')) { icon = '🔍'; description = 'ExitScam: Проверка'; }
                }
                else if (label === 'RSI Time Filter') {
                    if (statusValue.includes('ALLOWED')) { icon = '✅'; description = 'RSI Time Filter разрешен'; }
                    else if (statusValue.includes('BLOCKED')) { icon = '❌'; description = 'RSI Time Filter заблокирован'; }
                    else if (statusValue.includes('WAITING')) { icon = '⏳'; description = 'RSI Time Filter ожидание'; }
                    else if (statusValue.includes('TIMEOUT')) { icon = '⏰'; description = 'RSI Time Filter таймаут'; }
                }
                else if (label === 'Статус бота') {
                    if (statusValue === 'Нет бота') { 
                        icon = '❓'; 
                        description = 'Бот не создан';
                        
                        // Показываем кнопку "Включить" только для монет с сигналами LONG/SHORT
                        const enableBotBtn = document.getElementById('enableBotBtn');
                        if (enableBotBtn && this.selectedCoin) {
                            const signal = this.selectedCoin.signal;
                            if (signal === 'ENTER_LONG' || signal === 'ENTER_SHORT') {
                                enableBotBtn.style.display = 'inline-block';
                            } else {
                                enableBotBtn.style.display = 'none';
                            }
                        }
                    }
                    else if (statusValue.includes('running')) { 
                        icon = '🟢'; 
                        description = 'Бот активен и работает';
                        // Скрываем кнопку для активных ботов
                        const enableBotBtn = document.getElementById('enableBotBtn');
                        if (enableBotBtn) enableBotBtn.style.display = 'none';
                    }
                    else if (statusValue.includes('waiting')) { icon = '🔵'; description = 'Ожидание сигнала'; }
                    else if (statusValue.includes('error')) { icon = '🔴'; description = 'Ошибка в работе'; }
                    else if (statusValue.includes('stopped')) { icon = '🔴'; description = 'Бот остановлен'; }
                    else if (statusValue.includes('in_position')) { icon = '🟣'; description = 'В позиции'; }
                    else if (statusValue.includes('paused')) { icon = '⚪'; description = 'Приостановлен'; }
                }
                
                iconElement.textContent = icon;
                iconElement.title = `${label}: ${description || statusValue}`;
                valueElement.title = `${label}: ${description || statusValue}`;
            } else {
                itemElement.style.display = 'none';
            }
        }
    }
    
    getStatusIcon(statusType, statusValue) {
        const iconMap = {
            'OVERSOLD': '🔴',
            'OVERBOUGHT': '🟢',
            'NEUTRAL': '🟡',
            'UP': '📈',
            'DOWN': '📉'
        };
        
        return iconMap[statusValue] || '';
    }
    
    forceShowAllFilters() {
        console.log('[BotsManager] 🔧 ПРИНУДИТЕЛЬНО ПОКАЗЫВАЕМ ВСЕ ФИЛЬТРЫ');
        
        if (!this.selectedCoin) return;
        const coin = this.selectedCoin;
        
        // Получаем РЕАЛЬНЫЕ данные из объекта coin и конфига
        const realFilters = [];
        
        // 1. Ручная позиция
        if (coin.is_manual_position) {
            realFilters.push({
                itemId: 'manualPositionItem',
                valueId: 'selectedCoinManualPosition',
                iconId: 'manualPositionIcon',
                value: 'Ручная позиция',
                icon: '',
                description: 'Монета в ручной позиции'
            });
        }
        
        // 2. Зрелость монеты
        if (coin.is_mature) {
            const actualCandles = coin.candles_count || 'N/A';
            const minCandles = this.autoBotConfig?.min_candles_for_maturity || 400;
            realFilters.push({
                itemId: 'maturityDiamondItem',
                valueId: 'selectedCoinMaturityDiamond',
                iconId: 'maturityDiamondIcon',
                value: `Зрелая монета: ${actualCandles} > ${minCandles} свечей - достаточно истории для торговли`,
                icon: '',
                description: 'Монета имеет достаточно истории для надежного анализа'
            });
        } else if (coin.is_mature === false) {
            const minCandles = this.autoBotConfig?.min_candles_for_maturity || 400;
            realFilters.push({
                itemId: 'maturityDiamondItem',
                valueId: 'selectedCoinMaturityDiamond',
                iconId: 'maturityDiamondIcon',
                value: `Незрелая монета, менее ${minCandles} свечей - недостаточно истории`,
                icon: '',
                description: 'Монета не имеет достаточно истории для надежного анализа'
            });
        }
        
        // 3. Enhanced RSI данные
        if (coin.enhanced_rsi && coin.enhanced_rsi.enabled) {
            const enhancedRsi = coin.enhanced_rsi;
            
            // Время в экстремальной зоне
            if (enhancedRsi.extreme_duration > 0) {
                realFilters.push({
                    itemId: 'extremeDurationItem',
                    valueId: 'selectedCoinExtremeDuration',
                    iconId: 'extremeDurationIcon',
                    value: `${enhancedRsi.extreme_duration}🕐`,
                    icon: '',
                    description: 'Время в экстремальной зоне RSI'
                });
            }
            
            // Подтверждения
            if (enhancedRsi.confirmations) {
                const conf = enhancedRsi.confirmations;
                
                // Подтверждение объемом
                if (conf.volume) {
                    realFilters.push({
                        itemId: 'volumeConfirmationItem',
                        valueId: 'selectedCoinVolumeConfirmation',
                        iconId: 'volumeConfirmationIcon',
                        value: 'Подтвержден объемом',
                        icon: '📊',
                        description: 'Объем подтверждает сигнал'
                    });
                }
                
                // Дивергенция
                if (conf.divergence) {
                    const divIcon = conf.divergence === 'BULLISH_DIVERGENCE' ? '📈' : '📉';
                    realFilters.push({
                        itemId: 'divergenceItem',
                        valueId: 'selectedCoinDivergence',
                        iconId: 'divergenceIcon',
                        value: conf.divergence,
                        icon: divIcon,
                        description: `Дивергенция: ${conf.divergence}`
                    });
                }
                
                // Stochastic RSI
                if (conf.stoch_rsi_k !== undefined && conf.stoch_rsi_k !== null) {
                    const stochK = conf.stoch_rsi_k;
                    const stochD = conf.stoch_rsi_d || 0;
                    
                    let stochIcon, stochStatus, stochDescription;
                    
                    // Определяем статус и описание
                    if (stochK < 20) {
                        stochIcon = '⬇️';
                        stochStatus = 'OVERSOLD';
                        stochDescription = 'Перепроданность: %K ниже 20 - возможен разворот вверх';
                    } else if (stochK > 80) {
                        stochIcon = '⬆️';
                        stochStatus = 'OVERBOUGHT';
                        stochDescription = 'Перекупленность: %K выше 80 - возможен разворот вниз';
                    } else {
                        stochIcon = '➡️';
                        stochStatus = 'NEUTRAL';
                        stochDescription = 'Нейтральная зона: %K между 20-80 - тренд продолжается';
                    }
                    
                    // Добавляем информацию о пересечении
                    let crossoverInfo = '';
                    if (stochK > stochD) {
                        crossoverInfo = ' (%K выше %D - бычий сигнал)';
                    } else if (stochK < stochD) {
                        crossoverInfo = ' (%K ниже %D - медвежий сигнал)';
                    } else {
                        crossoverInfo = ' (%K = %D - нейтрально)';
                    }
                    
                    const fullDescription = `Stochastic RSI: ${stochDescription}${crossoverInfo}`;
                    
                    // Создаем подробное описание для отображения на странице
                    let detailedValue = '';
                    
                    // Определяем сигнал пересечения с цветами
                    let signalInfo = '';
                    if (stochK > stochD) {
                        signalInfo = `<span style="color: #44ff44;">Бычий сигнал: %D=${stochD.toFixed(1)} (%K выше %D)</span>`;
                    } else if (stochK < stochD) {
                        signalInfo = `<span style="color: #ff4444;">Медвежий сигнал: %D=${stochD.toFixed(1)} (%K ниже %D)</span>`;
                    } else {
                        signalInfo = `<span style="color: #ffff44;">Нейтральный сигнал: %D=${stochD.toFixed(1)} (%K = %D)</span>`;
                    }
                    
                    if (stochStatus === 'OVERSOLD') {
                        detailedValue = `<span style="color: #44ff44;">Перепроданность: %K=${stochK.toFixed(1)} (ниже 20) - возможен разворот цены вверх.</span><br>${signalInfo}`;
                    } else if (stochStatus === 'OVERBOUGHT') {
                        detailedValue = `<span style="color: #ff4444;">Перекупленность: %K=${stochK.toFixed(1)} (выше 80) - возможен разворот цены вниз.</span><br>${signalInfo}`;
                    } else {
                        detailedValue = `<span style="color: #ffff44;">Нейтральная зона: %K=${stochK.toFixed(1)} (между 20-80) - тренд продолжается.</span><br>${signalInfo}`;
                    }
                    
                    realFilters.push({
                        itemId: 'stochasticRsiItem',
                        valueId: 'selectedCoinStochasticRsi',
                        iconId: 'stochasticRsiIcon',
                        value: detailedValue,
                        icon: '',
                        description: fullDescription
                    });
                }
            }
            
            // Warning типы
            if (enhancedRsi.warning_type && enhancedRsi.warning_type !== 'ERROR') {
                const warningType = enhancedRsi.warning_type;
                const warningMessage = enhancedRsi.warning_message || '';
                
                if (warningType === 'EXTREME_OVERSOLD_LONG') {
                    realFilters.push({
                        itemId: 'extremeOversoldItem',
                        valueId: 'selectedCoinExtremeOversold',
                        iconId: 'extremeOversoldIcon',
                        value: 'EXTREME_OVERSOLD_LONG',
                        icon: '⚠️',
                        description: `ВНИМАНИЕ: ${warningMessage}. Требуются дополнительные подтверждения для LONG`
                    });
                } else if (warningType === 'EXTREME_OVERBOUGHT_LONG') {
                    realFilters.push({
                        itemId: 'extremeOverboughtItem',
                        valueId: 'selectedCoinExtremeOverbought',
                        iconId: 'extremeOverboughtIcon',
                        value: 'EXTREME_OVERBOUGHT_LONG',
                        icon: '⚠️',
                        description: `ВНИМАНИЕ: ${warningMessage}. Требуются дополнительные подтверждения для SHORT`
                    });
                } else if (warningType === 'OVERSOLD') {
                    realFilters.push({
                        itemId: 'oversoldWarningItem',
                        valueId: 'selectedCoinOversoldWarning',
                        iconId: 'oversoldWarningIcon',
                        value: 'OVERSOLD',
                        icon: '🟢',
                        description: warningMessage
                    });
                } else if (warningType === 'OVERBOUGHT') {
                    realFilters.push({
                        itemId: 'overboughtWarningItem',
                        valueId: 'selectedCoinOverboughtWarning',
                        iconId: 'overboughtWarningIcon',
                        value: 'OVERBOUGHT',
                        icon: '🔴',
                        description: warningMessage
                    });
                }
            }
        }
        
        // 4. RSI Time Filter
        if (coin.time_filter_info) {
            const timeFilter = coin.time_filter_info;
            const isBlocked = timeFilter.blocked;
            const reason = timeFilter.reason || '';
            const calmCandles = timeFilter.calm_candles || 0;
            
            realFilters.push({
                itemId: 'rsiTimeFilterItem',
                valueId: 'selectedCoinRsiTimeFilter',
                iconId: 'rsiTimeFilterIcon',
                value: isBlocked ? `Блокирует: ${reason}` : `Разрешено: ${reason}`,
                icon: isBlocked ? '⏰' : '⏱️',
                description: `RSI Time Filter: ${reason}${calmCandles > 0 ? ` (${calmCandles} спокойных свечей)` : ''}`
            });
        }
        
        // 5. ExitScam фильтр
        if (coin.exit_scam_info) {
            const exitScam = coin.exit_scam_info;
            const isBlocked = exitScam.blocked;
            const reason = exitScam.reason || '';
            
            // Добавляем цветовое кодирование
            let coloredValue = '';
            if (isBlocked) {
                coloredValue = `<span style="color: #ff4444;">Блокирует: ${reason}</span>`;
            } else {
                coloredValue = `<span style="color: #44ff44;">Безопасно: ${reason}</span>`;
            }
            
            realFilters.push({
                itemId: 'exitScamItem',
                valueId: 'selectedCoinExitScam',
                iconId: 'exitScamIcon',
                value: coloredValue,
                icon: '',
                description: `ExitScam фильтр: ${reason}`
            });
        }
        
        realFilters.forEach(filter => {
            const itemElement = document.getElementById(filter.itemId);
            const valueElement = document.getElementById(filter.valueId);
            const iconElement = document.getElementById(filter.iconId);
            
            if (itemElement && valueElement && iconElement) {
                itemElement.style.display = 'flex';
                // Используем innerHTML для поддержки цветного HTML контента
                valueElement.innerHTML = filter.value;
                iconElement.textContent = '';
                iconElement.title = filter.description;
                valueElement.title = filter.description;
                console.log(`[BotsManager] ✅ Показан фильтр: ${filter.itemId}`);
            }
        });
    }

    filterCoins(searchTerm) {
        const items = document.querySelectorAll('.coin-item');
        const term = searchTerm.toLowerCase();
        
        items.forEach(item => {
            const symbol = item.dataset.symbol.toLowerCase();
            const visible = symbol.includes(term);
            item.style.display = visible ? 'block' : 'none';
        });
    }

    applyRsiFilter(filter) {
        // Сохраняем текущий фильтр
        this.currentRsiFilter = filter;
        
        const items = document.querySelectorAll('.coin-item');
        
        items.forEach(item => {
            let visible = true;
            
            switch(filter) {
                case 'buy-zone':
                    visible = item.classList.contains('buy-zone');
                    break;
                case 'sell-zone':
                    visible = item.classList.contains('sell-zone');
                    break;
                case 'trend-up':
                    visible = item.classList.contains('trend-up');
                    break;
                case 'trend-down':
                    visible = item.classList.contains('trend-down');
                    break;
                case 'enter-long':
                    visible = item.classList.contains('enter-long');
                    break;
                case 'enter-short':
                    visible = item.classList.contains('enter-short');
                    break;
                case 'manual-position':
                    visible = item.classList.contains('manual-position');
                    break;
                case 'mature-coins':
                    visible = item.classList.contains('mature-coin');
                    break;
                case 'all':
                default:
                    visible = true;
                    break;
            }
            
            item.style.display = visible ? 'block' : 'none';
        });
        
        this.logDebug(`[BotsManager] 🔍 Применен фильтр: ${filter}`);
    }

    restoreFilterState() {
        // Восстанавливаем активную кнопку фильтра
        document.querySelectorAll('.rsi-filter-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.filter === this.currentRsiFilter) {
                btn.classList.add('active');
            }
        });
        
        // Применяем сохраненный фильтр
        this.applyRsiFilter(this.currentRsiFilter);
        
        this.logDebug(`[BotsManager] 🔄 Восстановлен фильтр: ${this.currentRsiFilter}`);
    }

    // Методы управления ботами
    async createBot() {
        console.log('[BotsManager] 🚀 Запуск создания бота...');
        
        if (!this.selectedCoin) {
            console.log('[BotsManager] ❌ Нет выбранной монеты!');
            this.showNotification('⚠️ ' + this.translate('select_coin_to_create_bot'), 'warning');
            return;
        }
        
        console.log(`[BotsManager] 🤖 Создание бота для ${this.selectedCoin.symbol}`);
        console.log(`[BotsManager] 📊 RSI текущий: ${this.selectedCoin.rsi6h || 'неизвестно'}`);
        
        // Показываем уведомление о начале процесса
        this.showNotification(`🔄 ${this.translate('creating_bot_for')} ${this.selectedCoin.symbol}...`, 'info');
        
        try {
            // Собираем дублированные настройки
            const duplicateSettings = this.collectDuplicateSettings();
            
            const config = {
                volume_mode: document.getElementById('volumeModeSelect')?.value || 'usdt',
                volume_value: parseFloat(document.getElementById('volumeValueInput')?.value || '10'),
                // Добавляем индивидуальные настройки бота
                ...duplicateSettings
            };
            
            console.log('[BotsManager] 📊 Полная конфигурация бота:', config);
            console.log('[BotsManager] 🌐 Отправка запроса на создание бота...');
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: this.selectedCoin.symbol,
                    config: config
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
                
            } else {
                console.error('[BotsManager] ❌ Ошибка создания бота:', data.error);
                this.showNotification(`❌ Ошибка создания бота: ${data.error}`, 'error');
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка создания бота:', error);
            this.showNotification('❌ ' + this.translate('connection_error_bot_service'), 'error');
        }
    }
    
    collectDuplicateSettings() {
        console.log('[BotsManager] 📋 Сбор дублированных настроек...');
        
        const settings = {};
        
        // RSI настройки
        const rsiLongEl = document.getElementById('rsiLongThresholdDup');
        if (rsiLongEl && rsiLongEl.value) settings.rsi_long_threshold = parseInt(rsiLongEl.value);
        
        const rsiShortEl = document.getElementById('rsiShortThresholdDup');
        if (rsiShortEl && rsiShortEl.value) settings.rsi_short_threshold = parseInt(rsiShortEl.value);
        
        const rsiExitLongEl = document.getElementById('rsiExitLongDup');
        if (rsiExitLongEl && rsiExitLongEl.value) settings.rsi_exit_long = parseInt(rsiExitLongEl.value);
        
        const rsiExitShortEl = document.getElementById('rsiExitShortDup');
        if (rsiExitShortEl && rsiExitShortEl.value) settings.rsi_exit_short = parseInt(rsiExitShortEl.value);
        
        // Защитные механизмы
        const maxLossEl = document.getElementById('maxLossPercentDup');
        if (maxLossEl && maxLossEl.value) settings.max_loss_percent = parseFloat(maxLossEl.value);
        
        const trailingActivationEl = document.getElementById('trailingStopActivationDup');
        if (trailingActivationEl && trailingActivationEl.value) settings.trailing_stop_activation = parseFloat(trailingActivationEl.value);
        
        const trailingDistanceEl = document.getElementById('trailingStopDistanceDup');
        if (trailingDistanceEl && trailingDistanceEl.value) settings.trailing_stop_distance = parseFloat(trailingDistanceEl.value);
        
        const maxHoursEl = document.getElementById('maxPositionHoursDup');
        if (maxHoursEl) {
            const minutes = parseInt(maxHoursEl.value) || 0;
            // Конвертируем минуты в секунды
            settings.max_position_hours = minutes * 60;
        }
        
        const breakEvenEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenEl) settings.break_even_protection = breakEvenEl.checked;
        
        const breakEvenTriggerEl = document.getElementById('breakEvenTriggerDup');
        if (breakEvenTriggerEl && breakEvenTriggerEl.value) settings.break_even_trigger = parseFloat(breakEvenTriggerEl.value);
        
        console.log('[BotsManager] 📋 Собранные настройки:', settings);
        return settings;
    }

    // Методы для работы с индивидуальными настройками монет
    async loadIndividualSettings(symbol) {
        if (!symbol) return null;
        
        try {
            console.log(`[BotsManager] 📥 Загрузка индивидуальных настроек для ${symbol}`);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}`);
            
            // 404 - это нормально, значит настроек нет
            if (response.status === 404) {
                console.log(`[BotsManager] ℹ️ Индивидуальных настроек для ${symbol} не найдено (404)`);
                return null;
            }
            
            const data = await response.json();
            
            if (data.success) {
                console.log(`[BotsManager] ✅ Индивидуальные настройки для ${symbol} загружены:`, data.settings);
                return data.settings;
            } else {
                console.log(`[BotsManager] ℹ️ Индивидуальных настроек для ${symbol} не найдено`);
                return null;
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка загрузки индивидуальных настроек для ${symbol}:`, error);
            return null;
        }
    }

    async saveIndividualSettings(symbol, settings) {
        if (!symbol || !settings) return false;
        
        try {
            console.log(`[BotsManager] 💾 Сохранение индивидуальных настроек для ${symbol}:`, settings);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            
            const data = await response.json();
            if (data.success) {
                console.log(`[BotsManager] ✅ Индивидуальные настройки для ${symbol} сохранены`);
                this.showNotification(`✅ Настройки для ${symbol} сохранены`, 'success');
                return true;
            } else {
                console.error(`[BotsManager] ❌ Ошибка сохранения настроек: ${data.error}`);
                this.showNotification(`❌ Ошибка сохранения: ${data.error}`, 'error');
                return false;
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка сохранения индивидуальных настроек для ${symbol}:`, error);
            this.showNotification('❌ Ошибка соединения при сохранении', 'error');
            return false;
        }
    }

    async deleteIndividualSettings(symbol) {
        if (!symbol) return false;
        
        try {
            console.log(`[BotsManager] 🗑️ Удаление индивидуальных настроек для ${symbol}`);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            if (data.success) {
                console.log(`[BotsManager] ✅ Индивидуальные настройки для ${symbol} удалены`);
                this.showNotification(`✅ Настройки для ${symbol} сброшены к общим`, 'success');
                return true;
            } else {
                console.error(`[BotsManager] ❌ Ошибка удаления настроек: ${data.error}`);
                this.showNotification(`❌ Ошибка удаления: ${data.error}`, 'error');
                return false;
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка удаления индивидуальных настроек для ${symbol}:`, error);
            this.showNotification('❌ Ошибка соединения при удалении', 'error');
            return false;
        }
    }

    async copySettingsToAllCoins(symbol) {
        if (!symbol) return false;
        
        try {
            console.log(`[BotsManager] 📋 Копирование настроек ${symbol} ко всем монетам`);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}/copy-to-all`, {
                method: 'POST'
            });
            
            const data = await response.json();
            if (data.success) {
                console.log(`[BotsManager] ✅ Настройки ${symbol} скопированы к ${data.copied_count} монетам`);
                this.showNotification(`✅ Настройки применены к ${data.copied_count} монетам`, 'success');
                return true;
            } else {
                console.error(`[BotsManager] ❌ Ошибка копирования настроек: ${data.error}`);
                this.showNotification(`❌ Ошибка копирования: ${data.error}`, 'error');
                return false;
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка копирования настроек ${symbol}:`, error);
            this.showNotification('❌ Ошибка соединения при копировании', 'error');
            return false;
        }
    }

    applyIndividualSettingsToUI(settings) {
        if (!settings) return;
        
        console.log('[BotsManager] 🎨 Применение индивидуальных настроек к UI:', settings);
        
        // RSI настройки
        const rsiLongEl = document.getElementById('rsiLongThresholdDup');
        if (rsiLongEl && settings.rsi_long_threshold !== undefined) {
            rsiLongEl.value = settings.rsi_long_threshold;
        }
        
        const rsiShortEl = document.getElementById('rsiShortThresholdDup');
        if (rsiShortEl && settings.rsi_short_threshold !== undefined) {
            rsiShortEl.value = settings.rsi_short_threshold;
        }
        
        const rsiExitLongEl = document.getElementById('rsiExitLongDup');
        if (rsiExitLongEl && settings.rsi_exit_long !== undefined) {
            rsiExitLongEl.value = settings.rsi_exit_long;
        }
        
        const rsiExitShortEl = document.getElementById('rsiExitShortDup');
        if (rsiExitShortEl && settings.rsi_exit_short !== undefined) {
            rsiExitShortEl.value = settings.rsi_exit_short;
        }
        
        // Защитные механизмы
        const maxLossEl = document.getElementById('maxLossPercentDup');
        if (maxLossEl && settings.max_loss_percent !== undefined) {
            maxLossEl.value = settings.max_loss_percent;
        }
        
        const trailingActivationEl = document.getElementById('trailingStopActivationDup');
        if (trailingActivationEl && settings.trailing_stop_activation !== undefined) {
            trailingActivationEl.value = settings.trailing_stop_activation;
        }
        
        const trailingDistanceEl = document.getElementById('trailingStopDistanceDup');
        if (trailingDistanceEl && settings.trailing_stop_distance !== undefined) {
            trailingDistanceEl.value = settings.trailing_stop_distance;
        }
        
        const maxHoursEl = document.getElementById('maxPositionHoursDup');
        if (maxHoursEl && settings.max_position_hours !== undefined) {
            // Конвертируем секунды в минуты
            maxHoursEl.value = Math.round(settings.max_position_hours / 60);
        }
        
        const breakEvenEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenEl && settings.break_even_protection !== undefined) {
            breakEvenEl.checked = settings.break_even_protection;
        }
        
        const breakEvenTriggerEl = document.getElementById('breakEvenTriggerDup');
        if (breakEvenTriggerEl && settings.break_even_trigger !== undefined) {
            breakEvenTriggerEl.value = settings.break_even_trigger;
        }
        
        // Трендовые настройки
        const avoidDownTrendEl = document.getElementById('avoidDownTrendDup');
        if (avoidDownTrendEl && settings.avoid_down_trend !== undefined) {
            avoidDownTrendEl.checked = settings.avoid_down_trend;
        }
        
        const avoidUpTrendEl = document.getElementById('avoidUpTrendDup');
        if (avoidUpTrendEl && settings.avoid_up_trend !== undefined) {
            avoidUpTrendEl.checked = settings.avoid_up_trend;
        }
        
        const enableMaturityEl = document.getElementById('enableMaturityCheckDup');
        if (enableMaturityEl && settings.enable_maturity_check !== undefined) {
            enableMaturityEl.checked = settings.enable_maturity_check;
        }
        
        // Объем торговли
        const volumeModeEl = document.getElementById('volumeModeSelect');
        if (volumeModeEl && settings.volume_mode !== undefined) {
            volumeModeEl.value = settings.volume_mode;
        }
        
        const volumeValueEl = document.getElementById('volumeValueInput');
        if (volumeValueEl && settings.volume_value !== undefined) {
            volumeValueEl.value = settings.volume_value;
        }
        
        console.log('[BotsManager] ✅ Индивидуальные настройки применены к UI');
    }

    updateIndividualSettingsStatus(hasSettings) {
        const statusEl = document.getElementById('individualSettingsStatus');
        if (statusEl) {
            if (hasSettings) {
                statusEl.innerHTML = '<span style="color: #4CAF50;">✅ Есть индивидуальные настройки</span>';
            } else {
                statusEl.innerHTML = '<span style="color: #888;">Нет индивидуальных настроек для этой монеты</span>';
            }
        }
    }

    async loadAndApplyIndividualSettings(symbol) {
        if (!symbol) return;
        
        try {
            console.log(`[BotsManager] 📥 Загрузка и применение индивидуальных настроек для ${symbol}`);
            const settings = await this.loadIndividualSettings(symbol);
            
            if (settings) {
                // Применяем настройки к UI
                this.applyIndividualSettingsToUI(settings);
                this.updateIndividualSettingsStatus(true);
                console.log(`[BotsManager] ✅ Индивидуальные настройки для ${symbol} применены`);
            } else {
                // Сбрасываем UI к общим настройкам
                this.resetToGeneralSettings();
                this.updateIndividualSettingsStatus(false);
                console.log(`[BotsManager] ℹ️ Используются общие настройки для ${symbol}`);
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка загрузки индивидуальных настроек для ${symbol}:`, error);
            this.updateIndividualSettingsStatus(false);
        }
    }

    resetToGeneralSettings() {
        // Сбрасываем все поля к значениям по умолчанию из общей конфигурации
        // Это можно реализовать, загрузив общие настройки и применив их к UI
        console.log('[BotsManager] 🔄 Сброс к общим настройкам');
        // Здесь можно добавить логику сброса к общим настройкам
    }

    initializeIndividualSettingsButtons() {
        console.log('[BotsManager] 🔧 Инициализация кнопок индивидуальных настроек...');
        
        // Кнопка сохранения индивидуальных настроек
        const saveIndividualBtn = document.getElementById('saveIndividualSettingsBtn');
        if (saveIndividualBtn) {
            saveIndividualBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для сохранения настроек', 'warning');
                    return;
                }
                
                const settings = this.collectDuplicateSettings();
                await this.saveIndividualSettings(this.selectedCoin.symbol, settings);
            });
        }
        
        // Кнопка загрузки индивидуальных настроек
        const loadIndividualBtn = document.getElementById('loadIndividualSettingsBtn');
        if (loadIndividualBtn) {
            loadIndividualBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для загрузки настроек', 'warning');
                    return;
                }
                
                await this.loadAndApplyIndividualSettings(this.selectedCoin.symbol);
            });
        }
        
        // Кнопка сброса к общим настройкам
        const resetIndividualBtn = document.getElementById('resetIndividualSettingsBtn');
        if (resetIndividualBtn) {
            resetIndividualBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для сброса настроек', 'warning');
                    return;
                }
                
                await this.deleteIndividualSettings(this.selectedCoin.symbol);
                this.resetToGeneralSettings();
                this.updateIndividualSettingsStatus(false);
            });
        }
        
        // Кнопка копирования настроек ко всем монетам
        const copyToAllBtn = document.getElementById('copyToAllCoinsBtn');
        if (copyToAllBtn) {
            copyToAllBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для копирования настроек', 'warning');
                    return;
                }
                
                const confirmed = confirm(`Вы уверены, что хотите применить настройки ${this.selectedCoin.symbol} ко всем монетам?`);
                if (confirmed) {
                    await this.copySettingsToAllCoins(this.selectedCoin.symbol);
                }
            });
        }
        
        console.log('[BotsManager] ✅ Кнопки индивидуальных настроек инициализированы');
    }

    initializeQuickLaunchButtons() {
        console.log('[BotsManager] 🚀 Инициализация кнопок быстрого запуска...');
        
        // Кнопка быстрого запуска LONG
        const quickStartLongBtn = document.getElementById('quickStartLongBtn');
        if (quickStartLongBtn) {
            quickStartLongBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для запуска', 'warning');
                    return;
                }
                
                await this.quickLaunchBot('LONG');
            });
        }
        
        // Кнопка быстрого запуска SHORT
        const quickStartShortBtn = document.getElementById('quickStartShortBtn');
        if (quickStartShortBtn) {
            quickStartShortBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для запуска', 'warning');
                    return;
                }
                
                await this.quickLaunchBot('SHORT');
            });
        }
        
        // Кнопка быстрой остановки
        const quickStopBtn = document.getElementById('quickStopBtn');
        if (quickStopBtn) {
            quickStopBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для остановки', 'warning');
                    return;
                }
                
                await this.stopBot();
            });
        }
        
        console.log('[BotsManager] ✅ Кнопки быстрого запуска инициализированы');
    }

    async quickLaunchBot(direction) {
        if (!this.selectedCoin) return;
        
        try {
            console.log(`[BotsManager] 🚀 Быстрый запуск ${direction} бота для ${this.selectedCoin.symbol}`);
            
            // Собираем настройки
            const settings = this.collectDuplicateSettings();
            settings.volume_mode = document.getElementById('volumeModeSelect')?.value || 'usdt';
            settings.volume_value = parseFloat(document.getElementById('volumeValueInput')?.value || '10');
            
            // Создаем бота с настройками
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: this.selectedCoin.symbol,
                    config: settings
                })
            });
            
            const data = await response.json();
            if (data.success) {
                this.showNotification(`✅ ${direction} бот для ${this.selectedCoin.symbol} создан и запущен`, 'success');
                
                // Обновляем UI
                await this.loadActiveBotsData();
                this.updateBotControlButtons();
                this.updateBotStatus();
                this.updateCoinsListWithBotStatus();
                this.renderActiveBotsDetails();
            } else {
                this.showNotification(`❌ Ошибка создания ${direction} бота: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка быстрого запуска ${direction} бота:`, error);
            this.showNotification('❌ Ошибка соединения при создании бота', 'error');
        }
    }

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
    }

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
                // Обновляем UI после успешной остановки
                this.updateBotStatusInUI(targetSymbol, 'stopped');
                
                // Обновляем локальные данные бота
                if (this.activeBots) {
                    const botIndex = this.activeBots.findIndex(bot => bot.symbol === targetSymbol);
                    if (botIndex >= 0) {
                        this.activeBots[botIndex].status = 'paused'; // или 'stopped'
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
    }

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
    }

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
    }

    // Немедленное обновление статуса бота в UI
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
                    statusElement.textContent = 'Запуск...';
                    statusElement.className = 'bot-status status-starting';
                    if (startButton) startButton.disabled = true;
                    if (stopButton) stopButton.disabled = true;
                    break;
                case 'active':
                    statusElement.textContent = 'Активен';
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
                    statusElement.textContent = 'Ожидание';
                    statusElement.className = 'bot-status status-idle';
                    if (startButton) startButton.disabled = false;
                    if (stopButton) stopButton.disabled = true;
                    break;
                case 'stopped':
                    statusElement.textContent = 'Остановлен';
                    statusElement.className = 'bot-status status-stopped';
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
    }

    // Удаление бота из UI
    removeBotFromUI(symbol) {
        const botCard = document.querySelector(`[data-symbol="${symbol}"]`);
        if (botCard) {
            botCard.remove();
        }
    }

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
    }

    getBotControlButtonsHtml(bot) {
        // Бот активен если running, idle, или в позиции
        const isRunning = bot.status === 'running' || bot.status === 'idle' || 
                         bot.status === 'in_position_long' || bot.status === 'in_position_short';
        const isStopped = bot.status === 'stopped' || bot.status === 'paused';
        
        let buttons = [];
        
        if (isRunning) {
            // Если бот работает - показываем кнопку СТОП
            buttons.push(`<button onclick="window.app.botsManager.stopBot('${bot.symbol}')" style="padding: 4px 8px; background: #f44336; border: none; border-radius: 3px; color: white; cursor: pointer; font-size: 10px;">⏹️ Стоп</button>`);
        } else if (isStopped) {
            // Если бот остановлен - показываем кнопку СТАРТ
            buttons.push(`<button onclick="window.app.botsManager.startBot('${bot.symbol}')" style="padding: 4px 8px; background: #4caf50; border: none; border-radius: 3px; color: white; cursor: pointer; font-size: 10px;">▶️ Старт</button>`);
        }
        
        // Кнопка удаления всегда доступна
        buttons.push(`<button onclick="window.app.botsManager.deleteBot('${bot.symbol}')" style="padding: 4px 8px; background: #9e9e9e; border: none; border-radius: 3px; color: white; cursor: pointer; font-size: 10px;">🗑️ Удалить</button>`);
        
        return buttons.join('');
    }

    getBotDetailButtonsHtml(bot) {
        // Бот активен если running, idle, или в позиции
        const isRunning = bot.status === 'running' || bot.status === 'idle' || 
                         bot.status === 'in_position_long' || bot.status === 'in_position_short';
        const isStopped = bot.status === 'stopped' || bot.status === 'paused';
        
        let buttons = [];
        
        if (isRunning) {
            // Если бот работает - показываем кнопку СТОП
            buttons.push(`<button onclick="window.app.botsManager.stopBot('${bot.symbol}')" style="padding: 5px 10px; background: #f44336; border: none; border-radius: 4px; color: white; cursor: pointer; font-size: 12px;">⏹️ Стоп</button>`);
        } else if (isStopped) {
            // Если бот остановлен - показываем кнопку СТАРТ  
            buttons.push(`<button onclick="window.app.botsManager.startBot('${bot.symbol}')" style="padding: 5px 10px; background: #4caf50; border: none; border-radius: 4px; color: white; cursor: pointer; font-size: 12px;">▶️ Старт</button>`);
        }
        
        // Кнопка удаления
        buttons.push(`<button onclick="window.app.botsManager.deleteBot('${bot.symbol}')" style="padding: 5px 10px; background: #9e9e9e; border: none; border-radius: 4px; color: white; cursor: pointer; font-size: 12px;">🗑️ Удалить</button>`);
        
        return buttons.join('');
    }

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
                        statusText.textContent = 'Бот создан (ожидает)';
                        break;
                    case 'running':
                        statusText.textContent = 'Бот активен';
                        break;
                    case 'in_position_long':
                        statusText.textContent = 'Бот активен (LONG)';
                        break;
                    case 'in_position_short':
                        statusText.textContent = 'Бот активен (SHORT)';
                        break;
                    case 'stopped':
                        statusText.textContent = 'Бот остановлен';
                        break;
                    case 'paused':
                        statusText.textContent = 'Бот на паузе';
                        break;
                    default:
                        statusText.textContent = 'Бот создан';
                }
            } else {
                statusText.textContent = 'Бот не создан';
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
    }

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
        
        // Проверяем есть ли бот для выбранной монеты
        const selectedBot = this.selectedCoin && this.activeBots ? 
                           this.activeBots.find(bot => bot.symbol === this.selectedCoin.symbol) : null;
        
        console.log(`[BotsManager] 🔍 Выбранная монета: ${this.selectedCoin?.symbol}`);
        console.log(`[BotsManager] 🤖 Найден бот:`, selectedBot);
        
        if (selectedBot) {
            // Есть бот для выбранной монеты
            const isRunning = selectedBot.status === 'running';
            const isStopped = selectedBot.status === 'idle' || selectedBot.status === 'stopped' || selectedBot.status === 'paused';
            
            if (createBtn) createBtn.style.display = 'none';
            
            if (isRunning) {
                // Бот работает - показываем только Стоп
                if (startBtn) startBtn.style.display = 'none';
                if (stopBtn) stopBtn.style.display = 'inline-block';
                if (pauseBtn) pauseBtn.style.display = 'none';
                if (resumeBtn) resumeBtn.style.display = 'none';
                
                // Кнопки быстрого запуска скрыты, показываем только быструю остановку
                if (quickStartLongBtn) quickStartLongBtn.style.display = 'none';
                if (quickStartShortBtn) quickStartShortBtn.style.display = 'none';
                if (quickStopBtn) quickStopBtn.style.display = 'inline-block';
            } else if (isStopped) {
                // Бот остановлен - показываем Старт
                if (startBtn) startBtn.style.display = 'inline-block';
                if (stopBtn) stopBtn.style.display = 'none';
                if (pauseBtn) pauseBtn.style.display = 'none';
                if (resumeBtn) resumeBtn.style.display = 'none';
                
                // Кнопки быстрого запуска скрыты, показываем только быструю остановку
                if (quickStartLongBtn) quickStartLongBtn.style.display = 'none';
                if (quickStartShortBtn) quickStartShortBtn.style.display = 'none';
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
            
            // Показываем кнопки быстрого запуска
            if (quickStartLongBtn) quickStartLongBtn.style.display = 'inline-block';
            if (quickStartShortBtn) quickStartShortBtn.style.display = 'inline-block';
            if (quickStopBtn) quickStopBtn.style.display = 'none';
            
            console.log(`[BotsManager] 🆕 Нет бота, показаны кнопки создания и быстрого запуска`);
        }
    }

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
    }

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
    }

    async loadFiltersData() {
        console.log('[BotsManager] 🔧 Загрузка данных фильтров...');
        
        if (!this.serviceOnline) return;
        
        try {
            // Получаем конфигурацию Auto Bot с фильтрами
            const response = await fetch(`${this.apiUrl}/auto-bot`);
            const data = await response.json();
            
            if (data.success && data.config) {
                this.filtersData = {
                    whitelist: data.config.whitelist || [],
                    blacklist: data.config.blacklist || []
                };
                
                this.renderFilters();
                this.initializeFilterControls();
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки фильтров:', error);
        }
    }

    renderFilters() {
        this.renderWhitelist();
        this.renderBlacklist();
    }

    renderWhitelist() {
        const container = document.getElementById('whitelistContainer');
        const countElement = document.getElementById('whitelistCount');
        
        if (!container || !countElement) return;
        
        const whitelist = this.filtersData?.whitelist || [];
        countElement.textContent = whitelist.length;
        
        if (whitelist.length === 0) {
            const currentLang = document.documentElement.lang || 'ru';
            const whiteListEmptyText = TRANSLATIONS[currentLang]['white_list_empty_text'] || 'Белый список пуст';
            const addCoinsForTradingText = TRANSLATIONS[currentLang]['add_coins_for_auto_trading'] || 'Добавьте монеты для автоматической торговли';
            
        container.innerHTML = `
                <div class="empty-filter-state">
                    <p>${whiteListEmptyText}</p>
                    <small>${addCoinsForTradingText}</small>
            </div>
        `;
        } else {
            container.innerHTML = whitelist.map(symbol => `
                <div class="filter-item" data-symbol="${symbol}">
                <span class="filter-item-symbol">${symbol}</span>
                    <button class="filter-item-remove" onclick="window.botsManager.removeFromWhitelist('${symbol}')">
                        ❌ Удалить
                    </button>
            </div>
        `).join('');
        }
    }

    renderBlacklist() {
        const container = document.getElementById('blacklistContainer');
        const countElement = document.getElementById('blacklistCount');
        
        if (!container || !countElement) return;
        
        const blacklist = this.filtersData?.blacklist || [];
        countElement.textContent = blacklist.length;
        
        if (blacklist.length === 0) {
            const currentLang = document.documentElement.lang || 'ru';
            const blackListEmptyText = TRANSLATIONS[currentLang]['black_list_empty_text'] || 'Черный список пуст';
            const addCoinsForExclusionText = TRANSLATIONS[currentLang]['add_coins_for_exclusion'] || 'Добавьте монеты для исключения';
            
            container.innerHTML = `
                <div class="empty-filter-state">
                    <p>${blackListEmptyText}</p>
                    <small>${addCoinsForExclusionText}</small>
            </div>
        `;
        } else {
            container.innerHTML = blacklist.map(symbol => `
                <div class="filter-item" data-symbol="${symbol}">
                    <span class="filter-item-symbol">${symbol}</span>
                    <button class="filter-item-remove" onclick="window.botsManager.removeFromBlacklist('${symbol}')">
                        ❌ Удалить
                    </button>
            </div>
        `).join('');
        }
    }

    initializeFilterControls() {
        // Новый поиск на вкладке фильтров
        const filtersSearchInput = document.getElementById('filtersSearchInput');
        if (filtersSearchInput) {
            filtersSearchInput.addEventListener('input', (e) => {
                this.performFiltersSearch(e.target.value);
            });
        }
    }

    async addToWhitelist() {
        const input = document.getElementById('whitelistInput');
        if (!input) return;
        
        const symbol = input.value.trim().toUpperCase();
        if (!symbol) return;
        
        // Проверяем что монета существует
        if (!this.validateCoinSymbol(symbol)) {
            this.showNotification('❌ Монета не найдена среди доступных пар', 'error');
            return;
        }
        
        // Проверяем что монеты еще нет в списке
        const whitelist = this.filtersData?.whitelist || [];
        if (whitelist.includes(symbol)) {
            this.showNotification('⚠️ Монета уже в белом списке', 'warning');
            return;
        }
        
        try {
            whitelist.push(symbol);
            await this.updateFilters({ whitelist });
            input.value = '';
            this.showNotification(`✅ ${symbol} добавлена в белый список`, 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка добавления в белый список:', error);
            this.showNotification('❌ Ошибка добавления в белый список', 'error');
        }
    }

    async addToBlacklist() {
        const input = document.getElementById('blacklistInput');
        if (!input) return;
        
        const symbol = input.value.trim().toUpperCase();
        if (!symbol) return;
        
        // Проверяем что монета существует
        if (!this.validateCoinSymbol(symbol)) {
            this.showNotification('❌ Монета не найдена среди доступных пар', 'error');
            return;
        }

        // Проверяем что монеты еще нет в списке
        const blacklist = this.filtersData?.blacklist || [];
        if (blacklist.includes(symbol)) {
            this.showNotification('⚠️ Монета уже в черном списке', 'warning');
            return;
        }

        try {
            blacklist.push(symbol);
            await this.updateFilters({ blacklist });
        input.value = '';
            this.showNotification(`✅ ${symbol} добавлена в черный список`, 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка добавления в черный список:', error);
            this.showNotification(`❌ Ошибка добавления в черный список: ${error.message}`, 'error');
        }
    }

    async removeFromWhitelist(symbol) {
        try {
            const whitelist = (this.filtersData?.whitelist || []).filter(s => s !== symbol);
            await this.updateFilters({ whitelist });
            this.showNotification(`✅ ${symbol} удалена из белого списка`, 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка удаления из белого списка:', error);
            this.showNotification('❌ Ошибка удаления из белого списка', 'error');
        }
    }

    async removeFromBlacklist(symbol) {
        try {
            const blacklist = (this.filtersData?.blacklist || []).filter(s => s !== symbol);
            await this.updateFilters({ blacklist });
            this.showNotification(`✅ ${symbol} удалена из черного списка`, 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка удаления из черного списка:', error);
            this.showNotification('❌ Ошибка удаления из черного списка', 'error');
        }
    }

    async updateFilters(updates) {
        // Убеждаемся что filtersData инициализирован
        if (!this.filtersData) {
            this.filtersData = { whitelist: [], blacklist: [] };
        }
        
        // Обновляем локальные данные
        if (updates.whitelist !== undefined) {
            this.filtersData.whitelist = updates.whitelist;
        }
        if (updates.blacklist !== undefined) {
            this.filtersData.blacklist = updates.blacklist;
        }
        
        // Отправляем на сервер
        const response = await fetch(`${this.apiUrl}/auto-bot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updates)
        });
        
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Ошибка обновления фильтров');
        }
        
        // Перерисовываем интерфейс
        this.renderFilters();
    }

    validateCoinSymbol(symbol) {
        // Проверяем что монета есть в списке доступных пар
        return this.coinsRsiData && this.coinsRsiData.some(coin => coin.symbol === symbol);
    }

    // Вспомогательная функция для перевода сообщений
    translate(key, params = {}) {
        if (window.languageUtils && typeof languageUtils.translate === 'function') {
            let text = languageUtils.translate(key);
            // Замена параметров в строке
            Object.keys(params).forEach(param => {
                text = text.replace(`{${param}}`, params[param]);
            });
            return text;
        }
        // Fallback на ключ если система переводов не доступна
        return key;
    }

    showNotification(message, type = 'info') {
        console.log(`[BotsManager] 📢 ${message}`);
        
        // Используем новую toast систему
        if (window.toastManager) {
            switch(type) {
                case 'success':
                    window.toastManager.success(message);
                    break;
                case 'error':
                    window.toastManager.error(message);
                    break;
                case 'warning':
                    window.toastManager.warning(message);
                    break;
                case 'info':
                default:
                    window.toastManager.info(message);
                    break;
            }
        } else {
            // Fallback для совместимости
            console.log(`[NOTIFICATION] ${type.toUpperCase()}: ${message}`);
        }
    }

    // ============ НОВЫЕ МЕТОДЫ ДЛЯ УЛУЧШЕННЫХ ФИЛЬТРОВ ============

    showFilterControls(symbol) {
        const filterSection = document.getElementById('filterControlsSection');
        if (filterSection && symbol) {
            filterSection.style.display = 'block';
        }
    }

    updateFilterStatus(symbol) {
        const statusText = document.getElementById('filterStatusText');
        if (!statusText || !symbol) return;

        const whitelist = this.filtersData?.whitelist || [];
        const blacklist = this.filtersData?.blacklist || [];

        statusText.className = 'filter-status-text';

        if (blacklist.includes(symbol)) {
            statusText.textContent = '🔴 В черном списке';
            statusText.classList.add('in-blacklist');
        } else if (whitelist.includes(symbol)) {
            statusText.textContent = '🟢 В белом списке';
            statusText.classList.add('in-whitelist');
        } else {
            statusText.textContent = 'Не в фильтрах';
        }
    }

    async addSelectedCoinToWhitelist() {
        if (!this.selectedCoin) {
            return;
        }

        // Убеждаемся что фильтры загружены
        if (!this.filtersData) {
            await this.loadFiltersData();
        }

        const symbol = this.selectedCoin.symbol;
        const whitelist = this.filtersData?.whitelist || [];
        const blacklist = this.filtersData?.blacklist || [];

        // Если уже в белом списке - подсвечиваем
        if (whitelist.includes(symbol)) {
            this.highlightFilterStatus(symbol, 'whitelist');
            return;
        }

        try {
            whitelist.push(symbol);
            
            // УБИРАЕМ ИЗ ЧЕРНОГО СПИСКА если там была
            const newBlacklist = blacklist.filter(s => s !== symbol);
            
            await this.updateFilters({ 
                whitelist: whitelist,
                blacklist: newBlacklist 
            });
            
            this.updateFilterStatus(symbol);
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка добавления в белый список:', error);
        }
    }

    async addSelectedCoinToBlacklist() {
        if (!this.selectedCoin) {
            return;
        }

        // Убеждаемся что фильтры загружены
        if (!this.filtersData) {
            await this.loadFiltersData();
        }

        const symbol = this.selectedCoin.symbol;
        const whitelist = this.filtersData?.whitelist || [];
        const blacklist = this.filtersData?.blacklist || [];

        // Если уже в черном списке - подсвечиваем
        if (blacklist.includes(symbol)) {
            this.highlightFilterStatus(symbol, 'blacklist');
            return;
        }

        try {
            blacklist.push(symbol);
            
            // УБИРАЕМ ИЗ БЕЛОГО СПИСКА если там была
            const newWhitelist = whitelist.filter(s => s !== symbol);
            
            await this.updateFilters({ 
                whitelist: newWhitelist,
                blacklist: blacklist 
            });
            
            this.updateFilterStatus(symbol);
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка добавления в черный список:', error);
        }
    }

    async removeSelectedCoinFromFilters() {
        if (!this.selectedCoin) {
            return;
        }

        // Убеждаемся что фильтры загружены
        if (!this.filtersData) {
            await this.loadFiltersData();
        }

        const symbol = this.selectedCoin.symbol;
        const whitelist = this.filtersData?.whitelist || [];
        const blacklist = this.filtersData?.blacklist || [];

        try {
            // Удаляем из обоих списков
            const newWhitelist = whitelist.filter(s => s !== symbol);
            const newBlacklist = blacklist.filter(s => s !== symbol);
            
            await this.updateFilters({ 
                whitelist: newWhitelist,
                blacklist: newBlacklist 
            });
            
            this.updateFilterStatus(symbol);
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка удаления из фильтров:', error);
        }
    }

    updateSmartFilterControls(searchTerm) {
        const controlsDiv = document.getElementById('smartFilterControls');
        const foundCountSpan = document.getElementById('foundCount');
        
        if (!controlsDiv || !foundCountSpan) return;

        if (!searchTerm || searchTerm.length < 2) {
            controlsDiv.style.display = 'none';
            return;
        }

        // Подсчитываем найденные монеты
        const foundCoins = this.getFoundCoins(searchTerm);
        
        if (foundCoins.length === 0) {
            controlsDiv.style.display = 'none';
            return;
        }

        foundCountSpan.textContent = `${foundCoins.length} найдено`;
        controlsDiv.style.display = 'block';
        
        // Сохраняем найденные монеты для массового добавления
        this.foundCoins = foundCoins;
    }

    getFoundCoins(searchTerm) {
        if (!this.coinsRsiData || !searchTerm) return [];

        const term = searchTerm.toLowerCase();
        return this.coinsRsiData.filter(coin => 
            coin.symbol.toLowerCase().includes(term) ||
            coin.symbol.toLowerCase().startsWith(term)
        );
    }

    async addFoundCoinsToWhitelist() {
        if (!this.foundCoins || this.foundCoins.length === 0) {
            this.showNotification('⚠️ Нет найденных монет для добавления', 'warning');
            return;
        }

        try {
            const whitelist = this.filtersData?.whitelist || [];
            const newCoins = this.foundCoins
                .map(coin => coin.symbol)
                .filter(symbol => !whitelist.includes(symbol));

            if (newCoins.length === 0) {
                this.showNotification('⚠️ Все найденные монеты уже в белом списке', 'warning');
                return;
            }

            whitelist.push(...newCoins);
            await this.updateFilters({ whitelist });
            
            // Очищаем поиск
            const searchInput = document.getElementById('coinSearchInput');
            if (searchInput) searchInput.value = '';
            this.filterCoins('');
            this.updateSmartFilterControls('');

            this.showNotification(`✅ Добавлено ${newCoins.length} монет в белый список`, 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка массового добавления в белый список:', error);
            this.showNotification('❌ Ошибка добавления в белый список', 'error');
        }
    }

    async addFoundCoinsToBlacklist() {
        if (!this.foundCoins || this.foundCoins.length === 0) {
            this.showNotification('⚠️ Нет найденных монет для добавления', 'warning');
            return;
        }

        try {
            const blacklist = this.filtersData?.blacklist || [];
            const newCoins = this.foundCoins
                .map(coin => coin.symbol)
                .filter(symbol => !blacklist.includes(symbol));

            if (newCoins.length === 0) {
                this.showNotification('⚠️ Все найденные монеты уже в черном списке', 'warning');
                return;
            }

            blacklist.push(...newCoins);
            await this.updateFilters({ blacklist });
            
            // Очищаем поиск
            const searchInput = document.getElementById('coinSearchInput');
            if (searchInput) searchInput.value = '';
            this.filterCoins('');
            this.updateSmartFilterControls('');

            this.showNotification(`✅ Добавлено ${newCoins.length} монет в черный список`, 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка массового добавления в черный список:', error);
            this.showNotification('❌ Ошибка добавления в черный список', 'error');
        }
    }

    // ============ ПОИСК НА ВКЛАДКЕ ФИЛЬТРОВ ============

    performFiltersSearch(searchTerm) {
        const resultsContainer = document.getElementById('filtersSearchResults');
        if (!resultsContainer) return;

        console.log(`[BotsManager] 🔍 Поиск фильтров: "${searchTerm}"`);

        if (!searchTerm || searchTerm.length < 2) {
            resultsContainer.innerHTML = `
                <div class="search-prompt">
                    <p>💡 Введите минимум 2 символа для поиска</p>
                    <small>Будут показаны все монеты содержащие введенный текст</small>
                </div>
            `;
            return;
        }

        // Фильтруем монеты
        const foundCoins = this.searchCoins(searchTerm);
        
        if (foundCoins.length === 0) {
            resultsContainer.innerHTML = `
                <div class="search-prompt">
                    <p>🔍 Ничего не найдено по запросу "${searchTerm}"</p>
                    <small>Попробуйте другие символы</small>
                </div>
            `;
            return;
        }

        // Показываем результаты
        this.renderSearchResults(foundCoins, resultsContainer);
    }

    searchCoins(searchTerm) {
        if (!this.coinsRsiData || !searchTerm) return [];

        const term = searchTerm.toLowerCase();
        return this.coinsRsiData.filter(coin => 
            coin.symbol.toLowerCase().includes(term)
        ).slice(0, 50); // Ограничиваем 50 результатами
    }

    renderSearchResults(coins, container) {
        const whitelist = this.filtersData?.whitelist || [];
        const blacklist = this.filtersData?.blacklist || [];

        const resultsHtml = coins.map(coin => {
            const inWhitelist = whitelist.includes(coin.symbol);
            const inBlacklist = blacklist.includes(coin.symbol);
            const inAnyList = inWhitelist || inBlacklist;
            
            let statusHtml = '';
            
            if (inBlacklist) {
                statusHtml = '<div class="search-result-status in-blacklist">В черном списке</div>';
            } else if (inWhitelist) {
                statusHtml = '<div class="search-result-status in-whitelist">В белом списке</div>';
            }

            return `
                <div class="search-result-item">
                    <div class="search-result-info">
                        <div class="search-result-symbol">${coin.symbol}</div>
                        ${statusHtml}
                    </div>
                    <div class="search-result-buttons">
                        <button class="btn-search-white" 
                                onclick="window.botsManager.addCoinToWhitelistFromSearch('${coin.symbol}')">
                            🟢 Белый
                        </button>
                        <button class="btn-search-black" 
                                onclick="window.botsManager.addCoinToBlacklistFromSearch('${coin.symbol}')">
                            🔴 Черный
                        </button>
                        <button class="btn-search-remove" 
                                onclick="window.botsManager.removeCoinFromFiltersFromSearch('${coin.symbol}')">
                            🗑️ Удалить
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = `
            <div style="padding: 12px; background: var(--bg-tertiary); border-bottom: 1px solid var(--border-color); font-size: 14px; color: var(--text-muted);">
                📊 Найдено ${coins.length} монет
            </div>
            ${resultsHtml}
        `;
    }

    async addCoinToWhitelistFromSearch(symbol) {
        // Убеждаемся что фильтры загружены
        if (!this.filtersData) {
            await this.loadFiltersData();
        }

        const whitelist = this.filtersData?.whitelist || [];
        const blacklist = this.filtersData?.blacklist || [];

        // Если уже в белом списке - подсвечиваем
        if (whitelist.includes(symbol)) {
            this.highlightStatus(symbol, 'whitelist');
            return;
        }

        try {
            whitelist.push(symbol);
            
            // УБИРАЕМ ИЗ ЧЕРНОГО СПИСКА если там была
            const newBlacklist = blacklist.filter(s => s !== symbol);
            
            await this.updateFilters({ 
                whitelist: whitelist,
                blacklist: newBlacklist 
            });
            
            // Обновляем поиск для показа нового статуса
            const searchInput = document.getElementById('filtersSearchInput');
            if (searchInput && searchInput.value) {
                this.performFiltersSearch(searchInput.value);
            }
            
            // ОБНОВЛЯЕМ СПИСКИ СПРАВА
            this.renderFilters();
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка добавления в белый список:', error);
        }
    }

    async addCoinToBlacklistFromSearch(symbol) {
        // Убеждаемся что фильтры загружены
        if (!this.filtersData) {
            await this.loadFiltersData();
        }

        const whitelist = this.filtersData?.whitelist || [];
        const blacklist = this.filtersData?.blacklist || [];

        // Если уже в черном списке - подсвечиваем
        if (blacklist.includes(symbol)) {
            this.highlightStatus(symbol, 'blacklist');
            return;
        }

        try {
            blacklist.push(symbol);
            
            // УБИРАЕМ ИЗ БЕЛОГО СПИСКА если там была
            const newWhitelist = whitelist.filter(s => s !== symbol);
            
            await this.updateFilters({ 
                whitelist: newWhitelist,
                blacklist: blacklist 
            });
            
            // Обновляем поиск для показа нового статуса
            const searchInput = document.getElementById('filtersSearchInput');
            if (searchInput && searchInput.value) {
                this.performFiltersSearch(searchInput.value);
            }
            
            // ОБНОВЛЯЕМ СПИСКИ СПРАВА
            this.renderFilters();
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка добавления в черный список:', error);
        }
    }

    async removeCoinFromFiltersFromSearch(symbol) {
        console.log(`[BotsManager] 🗑️ Удаление ${symbol} из фильтров через поиск`);
        
        // Убеждаемся что фильтры загружены
        if (!this.filtersData) {
            await this.loadFiltersData();
        }

        const whitelist = this.filtersData?.whitelist || [];
        const blacklist = this.filtersData?.blacklist || [];
        
        let removed = false;
        let listType = '';

        try {
            // Удаляем из белого списка если там есть
            if (whitelist.includes(symbol)) {
                const newWhitelist = whitelist.filter(s => s !== symbol);
                await this.updateFilters({ whitelist: newWhitelist });
                removed = true;
                listType = 'белого списка';
            }
            // Удаляем из черного списка если там есть  
            else if (blacklist.includes(symbol)) {
                const newBlacklist = blacklist.filter(s => s !== symbol);
                await this.updateFilters({ blacklist: newBlacklist });
                removed = true;
                listType = 'черного списка';
            }

            if (removed) {
                // Обновляем поиск для показа нового статуса
                const searchInput = document.getElementById('filtersSearchInput');
                if (searchInput && searchInput.value) {
                    this.performFiltersSearch(searchInput.value);
                }
                
                // ОБНОВЛЯЕМ СПИСКИ СПРАВА
                this.renderFilters();
                
                // ТИХОЕ УДАЛЕНИЕ - БЕЗ УВЕДОМЛЕНИЙ!
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка удаления из фильтров:', error);
            this.showNotification(`❌ Ошибка удаления из фильтров: ${error.message}`, 'error');
        }
    }

    highlightStatus(symbol, listType) {
        // Найти элемент с символом и подсветить статус
        const searchResults = document.getElementById('filtersSearchResults');
        if (!searchResults) return;

        const items = searchResults.querySelectorAll('.search-result-item');
        items.forEach(item => {
            const symbolElement = item.querySelector('.search-result-symbol');
            if (symbolElement && symbolElement.textContent === symbol) {
                const statusElement = item.querySelector('.search-result-status');
                if (statusElement) {
                    // Добавляем класс для анимации подсветки
                    statusElement.classList.add('highlight-flash');
                    
                    // Убираем через 1 секунду
                    setTimeout(() => {
                        statusElement.classList.remove('highlight-flash');
                    }, 1000);
                }
            }
        });
    }

    highlightFilterStatus(symbol, listType) {
        // Подсветка статуса на вкладке управления
        const statusElement = document.getElementById('filterStatusText');
        if (statusElement) {
            statusElement.classList.add('highlight-flash');
            
            // Убираем через 1 секунду
            setTimeout(() => {
                statusElement.classList.remove('highlight-flash');
            }, 1000);
        }
    }

    async loadActiveBotsData() {
        this.logDebug('[BotsManager] 🤖 Загрузка данных активных ботов...');
        
        if (!this.serviceOnline) return;
        
        try {
            // Синхронизация позиций с биржей каждые 3 секунды
            try {
                const syncResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/sync-positions`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                const syncData = await syncResponse.json();
                if (syncData.success) {
                    this.logDebug('[BotsManager] ✅ Позиции синхронизированы успешно');
                } else {
                    console.warn('[BotsManager] ⚠️ Ошибка синхронизации позиций:', syncData.message);
                }
            } catch (syncError) {
                console.warn('[BotsManager] ⚠️ Ошибка синхронизации позиций:', syncError);
            }
            
            // Затем загружаем и ботов, и конфигурацию автобота параллельно
            const [botsResponse, configResponse] = await Promise.all([
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/list`),
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`)
            ]);
            
            const botsData = await botsResponse.json();
            const configData = await configResponse.json();
            
            if (botsData.success) {
                console.log(`[DEBUG] loadActiveBotsData: получены данные ботов:`, botsData.bots);
                this.activeBots = botsData.bots;
                console.log(`[DEBUG] loadActiveBotsData: this.activeBots установлен:`, this.activeBots);
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
    }

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
        
        // Проверяем, нужно ли полностью перерисовывать HTML
        const existingBots = scrollListElement ? Array.from(scrollListElement.querySelectorAll('.active-bot-item')).map(item => item.dataset.symbol) : [];
        const currentBots = hasActiveBots ? this.activeBots.map(bot => bot.symbol) : [];
        const needsFullRedraw = JSON.stringify(existingBots.sort()) !== JSON.stringify(currentBots.sort());
        
        console.log(`[DEBUG] Проверка перерисовки:`, { existingBots, currentBots, needsFullRedraw });

        // Обновляем правую панель (вкладка "Управление")
        if (emptyStateElement && scrollListElement) {
            if (hasActiveBots) {
                emptyStateElement.style.display = 'none';
                scrollListElement.style.display = 'block';
                
                // Если список ботов изменился - полная перерисовка
                if (needsFullRedraw) {
                    console.log(`[DEBUG] Полная перерисовка правой панели`);
                    // Отображаем список активных ботов в правой панели
                    const rightPanelHtml = this.activeBots.map(bot => {
                    // Определяем статус бота (активен если running, idle, или в позиции)
                    const isActive = bot.status === 'running' || bot.status === 'idle' || 
                                    bot.status === 'in_position_long' || bot.status === 'in_position_short' ||
                                    bot.status === 'armed_up' || bot.status === 'armed_down';
                    
                    const statusColor = isActive ? '#4caf50' : '#ff5722';
                    const statusText = isActive ? 'Активен' : (bot.status === 'paused' ? 'Приостановлен' : (bot.status === 'idle' ? 'Ожидание' : 'Остановлен'));
                    
                    // Определяем информацию о позиции
                    console.log(`[DEBUG] renderActiveBotsDetails для ${bot.symbol}:`, {
                        position_side: bot.position_side,
                        entry_price: bot.entry_price,
                        current_price: bot.current_price,
                        rsi_data: bot.rsi_data
                    });
                    
                    const positionInfo = this.getBotPositionInfo(bot);
                    const timeInfo = this.getBotTimeInfo(bot);
                    
                    console.log(`[DEBUG] positionInfo для ${bot.symbol}:`, positionInfo);
                    console.log(`[DEBUG] timeInfo для ${bot.symbol}:`, timeInfo);
                    
                    const htmlResult = `
                        <div class="active-bot-item clickable-bot-item" data-symbol="${bot.symbol}" style="border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 8px 0; background: #2a2a2a; cursor: pointer;" onmouseover="this.style.backgroundColor='#333'" onmouseout="this.style.backgroundColor='#2a2a2a'">
                            <div class="bot-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="color: #fff; font-weight: bold; font-size: 16px;">${bot.symbol}</span>
                                    <span style="background: ${statusColor}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px;">${statusText}</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <div style="text-align: right;">
                                        <div style="color: ${(bot.unrealized_pnl || bot.unrealized_pnl_usdt || 0) >= 0 ? '#4caf50' : '#f44336'}; font-weight: bold; font-size: 14px;">$${(bot.unrealized_pnl_usdt || bot.unrealized_pnl || 0).toFixed(3)}</div>
                                    </div>
                                    <button class="collapse-btn" onclick="event.stopPropagation(); const details = this.parentElement.parentElement.parentElement.querySelector('.bot-details'); const isCurrentlyCollapsed = details.style.display === 'none'; details.style.display = isCurrentlyCollapsed ? 'block' : 'none'; this.textContent = isCurrentlyCollapsed ? '▲' : '▼'; window.botsManager && window.botsManager.saveCollapseState(this.parentElement.parentElement.parentElement.dataset.symbol, !isCurrentlyCollapsed);" style="background: none; border: none; color: #888; font-size: 12px; cursor: pointer; padding: 4px;">▼</button>
                                </div>
                            </div>
                                
                            <div class="bot-details" style="font-size: 12px; color: #ccc; margin-bottom: 8px; display: none;">
                                <div style="margin-bottom: 4px;">💰 Объем: ${parseFloat(((bot.position_size || 0) * (bot.entry_price || 0)).toFixed(2))} USDT</div>
                                ${positionInfo}
                                ${timeInfo}
                            </div>
                            
                            <div class="bot-controls" style="display: flex; gap: 8px; justify-content: center;">
                                ${this.getBotControlButtonsHtml(bot)}
                            </div>
                        </div>
                    `;
                    
                    console.log(`[DEBUG] Финальный HTML для ${bot.symbol}:`, htmlResult);
                    return htmlResult;
                }).join('');
                
                console.log(`[DEBUG] Вставляем HTML в DOM:`, rightPanelHtml);
                console.log(`[DEBUG] Элемент для вставки:`, scrollListElement);
                
                scrollListElement.innerHTML = rightPanelHtml;
                
                // Восстанавливаем состояние сворачивания ПОСЛЕ обновления HTML
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
                            // Обновляем PnL
                            const pnlElement = botItem.querySelector('.bot-header > div:last-child > div > div:first-child');
                            if (pnlElement) {
                                const pnlValue = (bot.unrealized_pnl_usdt || bot.unrealized_pnl || 0);
                                pnlElement.textContent = `$${pnlValue.toFixed(3)}`;
                                pnlElement.style.color = pnlValue >= 0 ? '#4caf50' : '#f44336';
                            }
                            
                            // Обновляем данные в деталях (если развернуто)
                            const details = botItem.querySelector('.bot-details');
                            if (details && details.style.display !== 'none') {
                                const positionInfo = this.getBotPositionInfo(bot);
                                const timeInfo = this.getBotTimeInfo(bot);
                                // Обновляем только изменяемые части, не трогая структуру
                                // TODO: можно оптимизировать дальше, обновляя только конкретные значения
                            }
                        }
                    });
                }
            } else {
                emptyStateElement.style.display = 'block';
                scrollListElement.style.display = 'none';
            }
        }

        // Обновляем вкладку "Боты в работе"
        if (detailsElement) {
            if (!hasActiveBots) {
                const currentLang = document.documentElement.lang || 'ru';
                const noActiveBotsText = TRANSLATIONS[currentLang]['no_active_bots'] || 'Нет активных ботов';
                const createBotsText = TRANSLATIONS[currentLang]['create_bots_for_trading'] || 'Создайте ботов для торговли';
                
                detailsElement.innerHTML = `
                    <div class="empty-bots-state" style="text-align: center; padding: 20px; color: #888;">
                        <div style="font-size: 48px; margin-bottom: 10px;">🤖</div>
                        <p style="margin: 10px 0; font-size: 16px;">${noActiveBotsText}</p>
                        <small style="color: #666;">${createBotsText}</small>
                    </div>
                `;
            } else {
                // Если список ботов изменился - полная перерисовка
                if (needsFullRedraw) {
                    console.log(`[DEBUG] Полная перерисовка вкладки "Боты в работе"`);
                    
                    const rightPanelHtml = this.activeBots.map(bot => {
                    // Определяем статус бота (активен если running, idle, или в позиции)
                    const isActive = bot.status === 'running' || bot.status === 'idle' || 
                                    bot.status === 'in_position_long' || bot.status === 'in_position_short' ||
                                    bot.status === 'armed_up' || bot.status === 'armed_down';
                    
                    const statusColor = isActive ? '#4caf50' : '#ff5722';
                    const statusText = isActive ? 'Активен' : (bot.status === 'paused' ? 'Приостановлен' : (bot.status === 'idle' ? 'Ожидание' : 'Остановлен'));
                    
                    // Определяем информацию о позиции
                    console.log(`[DEBUG] renderActiveBotsDetails для ${bot.symbol}:`, {
                        position_side: bot.position_side,
                        entry_price: bot.entry_price,
                        current_price: bot.current_price,
                        rsi_data: bot.rsi_data
                    });
                    
                    const positionInfo = this.getBotPositionInfo(bot);
                    const timeInfo = this.getBotTimeInfo(bot);
                    
                    console.log(`[DEBUG] positionInfo для ${bot.symbol}:`, positionInfo);
                    console.log(`[DEBUG] timeInfo для ${bot.symbol}:`, timeInfo);
                    
                    const htmlResult = `
                        <div class="active-bot-item clickable-bot-item" data-symbol="${bot.symbol}" style="border: 1px solid #333; border-radius: 12px; padding: 16px; margin: 12px 0; background: linear-gradient(135deg, #252525 0%, #2a2a2a 100%); cursor: pointer; transition: all 0.3s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.3);" onmouseover="this.style.backgroundColor='#2a2a2a'; this.style.borderColor='#555'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.4)'" onmouseout="this.style.backgroundColor='#252525'; this.style.borderColor='#333'; this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.3)'">
                            <div class="bot-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #333;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="color: #fff; font-weight: bold; font-size: 18px; text-shadow: 0 1px 2px rgba(0,0,0,0.5);">${bot.symbol}</span>
                                    <span style="background: ${statusColor}; color: white; padding: 4px 10px; border-radius: 16px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">${statusText}</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <div style="text-align: right;">
                                        <div style="color: ${(bot.unrealized_pnl || bot.unrealized_pnl_usdt || 0) >= 0 ? '#4caf50' : '#f44336'}; font-weight: bold; font-size: 16px; text-shadow: 0 1px 2px rgba(0,0,0,0.5);">$${(bot.unrealized_pnl_usdt || bot.unrealized_pnl || 0).toFixed(3)}</div>
                                        <div style="color: #888; font-size: 10px; margin-top: 2px;">PnL</div>
                                    </div>
                                    <button class="collapse-btn" onclick="event.stopPropagation(); const details = this.parentElement.parentElement.parentElement.querySelector('.bot-details'); const isCurrentlyCollapsed = details.style.display === 'none'; details.style.display = isCurrentlyCollapsed ? 'grid' : 'none'; this.textContent = isCurrentlyCollapsed ? '▲' : '▼'; window.botsManager && window.botsManager.saveCollapseState(this.parentElement.parentElement.parentElement.dataset.symbol, !isCurrentlyCollapsed);" style="background: none; border: none; color: #888; font-size: 14px; cursor: pointer; padding: 4px;">▼</button>
                                </div>
                            </div>
                            
                            <div class="bot-details" style="display: none; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 13px; color: #ccc; margin-bottom: 16px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                                    <span style="color: #888;">💰 Объем</span>
                                    <span style="color: #fff; font-weight: 600;">${bot.position_size || bot.volume_value} ${(bot.volume_mode || 'USDT').toUpperCase()}</span>
                                </div>
                                
                                ${positionInfo}
                                ${timeInfo}
                            </div>
                            
                            <div class="bot-controls" style="display: flex; gap: 8px; justify-content: center; padding-top: 12px; border-top: 1px solid #333;">
                                ${this.getBotControlButtonsHtml(bot)}
                            </div>
                        </div>
                    `;
                    
                    console.log(`[DEBUG] Финальный HTML для ${bot.symbol}:`, htmlResult);
                    return htmlResult;
                }).join('');

                    console.log(`[DEBUG] Вставляем ПОЛНЫЙ HTML в detailsElement:`, rightPanelHtml);
                    detailsElement.innerHTML = rightPanelHtml;
                    
                    // Восстанавливаем состояние сворачивания ПОСЛЕ обновления HTML
                    this.preserveCollapseState(detailsElement);
                } else {
                    // Обновляем только данные в существующих карточках
                    console.log(`[DEBUG] Обновление данных в "Боты в работе" без перерисовки`);
                    this.activeBots.forEach(bot => {
                        const botItem = detailsElement.querySelector(`.active-bot-item[data-symbol="${bot.symbol}"]`);
                        if (botItem) {
                            // Обновляем PnL
                            const pnlElement = botItem.querySelector('.bot-header > div:last-child > div > div:first-child');
                            if (pnlElement) {
                                const pnlValue = (bot.unrealized_pnl_usdt || bot.unrealized_pnl || 0);
                                pnlElement.textContent = `$${pnlValue.toFixed(3)}`;
                                pnlElement.style.color = pnlValue >= 0 ? '#4caf50' : '#f44336';
                            }
                        }
                    });
                }
            }
        }
        
        // Обновляем статистику в правой панели
        this.updateBotsSummaryStats();
        
        this.logDebug('[BotsManager] ✅ Активные боты отрисованы успешно');
    }

    updateBotsSummaryStats() {
        this.logDebug('[BotsManager] 📊 Обновление статистики ботов...');
        this.logDebug('[BotsManager] 📊 Активные боты:', this.activeBots);
        
        // Вычисляем общий PnL
        let totalPnL = 0;
        let inPositionCount = 0;
        
        if (this.activeBots && this.activeBots.length > 0) {
            this.activeBots.forEach(bot => {
                // Добавляем PnL бота к общему
                const botPnL = parseFloat(bot.unrealized_pnl || 0);
                totalPnL += botPnL;
                
                console.log(`[BotsManager] 📊 Бот ${bot.symbol}: PnL=$${botPnL}, Статус=${bot.status}`);
                
                // Считаем ботов в позиции
                if (bot.status === 'in_position_long' || bot.status === 'in_position_short') {
                    inPositionCount++;
                }
            });
        }
        
        // Обновляем элементы статистики
        const totalPnLElement = document.getElementById('totalPnLValue');
        if (totalPnLElement) {
            totalPnLElement.textContent = `$${totalPnL.toFixed(2)}`;
            totalPnLElement.style.color = totalPnL >= 0 ? '#4caf50' : '#f44336';
            this.logDebug(`[BotsManager] 📊 Обновлен элемент totalPnLValue: $${totalPnL.toFixed(2)}`);
        } else {
            console.warn('[BotsManager] ⚠️ Элемент totalPnLValue не найден!');
        }
        
        this.logDebug(`[BotsManager] 📊 Статистика обновлена: PnL=$${totalPnL.toFixed(2)}, В позиции=${inPositionCount}`);
    }

    startPeriodicUpdate() {
        // Обновляем данные с единым интервалом
        this.updateInterval = setInterval(() => {
            if (this.serviceOnline) {
                this.logDebug('[BotsManager] 🔄 Автообновление данных...');
                
                // Обновляем основные данные
                this.loadCoinsRsiData();
                this.loadAccountInfo();
                
                // КРИТИЧЕСКИ ВАЖНО: Всегда обновляем состояние автобота и ботов
                this.loadActiveBotsData();
        } else {
                this.checkBotsService();
            }
        }, this.refreshInterval);
        
        // Отдельный интервал для обновления данных аккаунта каждую секунду
        this.accountUpdateInterval = setInterval(() => {
            if (this.serviceOnline) {
                this.logDebug('[BotsManager] 💰 Обновление данных аккаунта...');
                this.loadAccountInfo();
            }
        }, 1000); // Каждую секунду
        
        console.log(`[BotsManager] ⏰ Запущено периодическое обновление (${this.refreshInterval/1000} сек)`);
        
        // Запускаем мониторинг активных ботов с тем же интервалом
        this.startBotMonitoring();
    }
    
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
    }
    
    stopBotMonitoring() {
        if (this.monitoringTimer) {
            clearInterval(this.monitoringTimer);
            this.monitoringTimer = null;
            console.log('[BotsManager] ⏹️ Мониторинг активных ботов остановлен');
        }
    }
    
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
    
    updateBotsDetailedDisplay(bots) {
        // Обновляем отображение каждого бота с детальной информацией
        bots.forEach(bot => {
            this.updateSingleBotDisplay(bot);
        });
    }
    
    updateSingleBotDisplay(bot) {
        // Находим элемент бота в списке
        const botElement = document.querySelector(`[data-bot-symbol="${bot.symbol}"]`);
        if (!botElement) return;
        
        // Обновляем PnL
        const pnlElement = botElement.querySelector('.bot-pnl');
        if (pnlElement) {
            const pnl = bot.pnl || 0;
            pnlElement.textContent = `PnL: $${pnl.toFixed(2)}`;
            pnlElement.style.color = pnl >= 0 ? 'var(--green-color)' : 'var(--red-color)';
        }
        
        // Обновляем цену
        const priceElement = botElement.querySelector('.bot-price');
        if (priceElement && bot.current_price) {
            priceElement.textContent = `$${bot.current_price.toFixed(6)}`;
        }
        
        // Обновляем направление позиции
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
        
        // Обновляем статус трейлинг стопа
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
            const timeLeft = this.calculateTimeLeft(bot.position_start_time, bot.max_position_hours);
            timeElement.textContent = `Время: ${timeLeft}`;
            timeElement.style.color = timeLeft.includes('0:00') ? 'var(--red-color)' : 'var(--blue-color)';
        } else if (timeElement) {
            timeElement.textContent = 'Время: ∞';
            timeElement.style.color = 'var(--gray-color)';
        }
    }
    
    calculateTimeLeft(startTime, maxHours) {
        const start = new Date(startTime);
        const now = new Date();
        const elapsed = now - start;
        const maxMs = maxHours * 60 * 60 * 1000;
        const remaining = maxMs - elapsed;
        
        if (remaining <= 0) {
            return '0:00';
        }
        
        const hours = Math.floor(remaining / (60 * 60 * 1000));
        const minutes = Math.floor((remaining % (60 * 60 * 1000)) / (60 * 1000));
        
        return `${hours}:${minutes.toString().padStart(2, '0')}`;
    }

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
    
    // ==========================================
    // ИНИЦИАЛИЗАЦИЯ КНОПОК ОБЛАСТИ ДЕЙСТВИЯ
    // ==========================================
    
    initializeScopeButtons() {
        const scopeButtons = document.querySelectorAll('.scope-btn');
        const scopeInput = document.getElementById('autoBotScope');
        
        if (!scopeButtons.length || !scopeInput) return;
        
        scopeButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Убираем активность со всех кнопок
                scopeButtons.forEach(btn => btn.classList.remove('active'));
                
                // Добавляем активность на нажатую кнопку
                button.classList.add('active');
                
                // Обновляем скрытое поле
                const value = button.getAttribute('data-value');
                scopeInput.value = value;
                
                console.log('[BotsManager] 🎯 Область действия изменена на:', value);
            });
        });
        
        console.log('[BotsManager] ✅ Кнопки области действия инициализированы');
    }
    
    // ==========================================
    // ЗАГРУЗКА КОНФИГУРАЦИИ
    // ==========================================
    
    async loadConfigurationData() {
        console.log('[BotsManager] 📋 ЗАГРУЗКА КОНФИГУРАЦИИ НАЧАТА...');
        console.log('[BotsManager] 🌐 Отправка запросов к API...');
        
        try {
            // Параллельная загрузка данных Auto Bot и системных настроек
            const [autoBotResponse, systemResponse] = await Promise.all([
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`),
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`)
            ]);
            
            console.log('[BotsManager] 📡 Ответы API получены');
            console.log('   Auto Bot status:', autoBotResponse.status);
            console.log('   System config status:', systemResponse.status);
            
            if (!autoBotResponse.ok || !systemResponse.ok) {
                throw new Error(`HTTP ${autoBotResponse.status} или ${systemResponse.status}`);
            }
            
            const autoBotData = await autoBotResponse.json();
            const systemData = await systemResponse.json();
            
            console.log('[BotsManager] 📋 Данные получены:');
            console.log('   Auto Bot:', autoBotData);
            console.log('   System:', systemData);
            
            if (autoBotData.success && systemData.success) {
                const config = {
                    autoBot: autoBotData.config,
                    system: systemData.config
                };
                
                console.log('[BotsManager] 📋 Заполнение формы данными...');
                console.log('[BotsManager] 🚀 ВЫЗОВ populateConfigurationForm с config:', config);
                this.populateConfigurationForm(config);
                console.log('[BotsManager] 🎯 populateConfigurationForm завершена');
                
                // КРИТИЧЕСКИ ВАЖНО: Инициализируем глобальный переключатель Auto Bot
                console.log('[BotsManager] 🤖 Инициализация глобального переключателя Auto Bot...');
                this.initializeGlobalAutoBotToggle();
            this.initializeMobileAutoBotToggle();
                
                console.log('[BotsManager] ✅ Конфигурация загружена и применена');
                return config;
            } else {
                throw new Error(`API ошибка: ${autoBotData.message || systemData.message}`);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки конфигурации:', error);
            this.showNotification('❌ Ошибка загрузки конфигурации', 'error');
            throw error;
        }
    }
    
    populateConfigurationForm(config) {
        this.logDebug('[BotsManager] 🔧 Заполнение формы конфигурации:', config);
        this.logDebug('[BotsManager] 🔍 DOM готовность:', document.readyState);
        this.logDebug('[BotsManager] 🔍 Элемент positionSyncInterval существует:', !!document.getElementById('positionSyncInterval'));
        this.logDebug('[BotsManager] 🔍 Детали конфигурации:');
        this.logDebug('   autoBot:', config.autoBot);
        this.logDebug('   system:', config.system);
        
        const autoBotConfig = config.autoBot || config;
        
        // ==========================================
        // КОНФИГУРАЦИЯ AUTO BOT
        // ==========================================
        
        // Основные настройки Auto Bot (включение/выключение управляется на основной вкладке)
        
        const maxConcurrentEl = document.getElementById('autoBotMaxConcurrent');
        if (maxConcurrentEl) {
            maxConcurrentEl.value = autoBotConfig.max_concurrent || 5;
            console.log('[BotsManager] 👥 Макс. одновременных ботов:', maxConcurrentEl.value);
        }
        
        const riskCapEl = document.getElementById('autoBotRiskCap');
        if (riskCapEl) {
            riskCapEl.value = autoBotConfig.risk_cap_percent || 10;
            console.log('[BotsManager] ⚠️ Лимит риска:', riskCapEl.value);
        }
        
        // Область действия
        const scopeEl = document.getElementById('autoBotScope');
        if (scopeEl) {
            const scopeValue = autoBotConfig.scope;
            if (scopeValue !== undefined) {
                scopeEl.value = scopeValue;
                console.log('[BotsManager] 🎯 Область действия:', scopeValue);
                
                const scopeButtons = document.querySelectorAll('.scope-btn');
                console.log('[BotsManager] 🔍 Найдено кнопок области:', scopeButtons.length);
                
                scopeButtons.forEach(btn => {
                    btn.classList.remove('active');
                    const btnValue = btn.getAttribute('data-value');
                    if (btnValue === scopeEl.value) {
                        btn.classList.add('active');
                        console.log('[BotsManager] ✅ Активирована кнопка:', btnValue);
                    }
                });
            } else {
                console.warn('[BotsManager] ⚠️ Область действия не найдена в API, оставляем поле пустым');
            }
        } else {
            console.error('[BotsManager] ❌ Элемент autoBotScope не найден!');
        }
        
        // Торговые параметры
        const rsiLongEl = document.getElementById('rsiLongThreshold');
        if (rsiLongEl) {
            rsiLongEl.value = autoBotConfig.rsi_long_threshold || 29;
            console.log('[BotsManager] 📈 RSI LONG порог:', rsiLongEl.value);
        }
        
        const rsiShortEl = document.getElementById('rsiShortThreshold');
        if (rsiShortEl) {
            rsiShortEl.value = autoBotConfig.rsi_short_threshold || 71;
            console.log('[BotsManager] 📉 RSI SHORT порог:', rsiShortEl.value);
        }
        
        const positionSizeEl = document.getElementById('defaultPositionSize');
        if (positionSizeEl) {
            positionSizeEl.value = autoBotConfig.default_position_size || 10;
            console.log('[BotsManager] 💰 Размер позиции:', positionSizeEl.value);
        }
        
        const checkIntervalEl = document.getElementById('checkInterval');
        if (checkIntervalEl && autoBotConfig.check_interval !== undefined) {
            checkIntervalEl.value = autoBotConfig.check_interval;
            console.log('[BotsManager] ⏱️ Интервал проверки установлен:', autoBotConfig.check_interval, '(из API)');
        } else if (checkIntervalEl) {
            console.warn('[BotsManager] ⚠️ Интервал проверки не найден в API, оставляем поле пустым');
        }
        

        
        const rsiExitLongEl = document.getElementById('rsiExitLong');
        if (rsiExitLongEl) {
            rsiExitLongEl.value = autoBotConfig.rsi_exit_long || 65;
            console.log('[BotsManager] 🟢 RSI выход LONG:', rsiExitLongEl.value);
        }
        
        const rsiExitShortEl = document.getElementById('rsiExitShort');
        if (rsiExitShortEl) {
            rsiExitShortEl.value = autoBotConfig.rsi_exit_short || 35;
            console.log('[BotsManager] 🔴 RSI выход SHORT:', rsiExitShortEl.value);
        }
        
        // Торговые настройки (торговля включена по умолчанию в backend)
        
        const useTestServerEl1 = document.getElementById('useTestServer');
        if (useTestServerEl1) {
            useTestServerEl1.checked = autoBotConfig.use_test_server || false;
            console.log('[BotsManager] 🧪 Тестовый сервер:', useTestServerEl1.checked);
        }
        
        const maxRiskEl = document.getElementById('maxRiskPerTrade');
        if (maxRiskEl) {
            maxRiskEl.value = autoBotConfig.max_risk_per_trade || 2.0;
            console.log('[BotsManager] ⚠️ Макс. риск на сделку:', maxRiskEl.value);
        }
        
        // ==========================================
        // ЗАЩИТНЫЕ МЕХАНИЗМЫ
        // ==========================================
        
        const maxLossPercentEl = document.getElementById('maxLossPercent');
        if (maxLossPercentEl) {
            maxLossPercentEl.value = autoBotConfig.max_loss_percent || 15.0;
            console.log('[BotsManager] 🛡️ Макс. убыток (стоп-лосс):', maxLossPercentEl.value);
        }
        
        const trailingStopActivationEl = document.getElementById('trailingStopActivation');
        if (trailingStopActivationEl) {
            trailingStopActivationEl.value = autoBotConfig.trailing_stop_activation || 300.0;
            console.log('[BotsManager] 📈 Активация trailing stop:', trailingStopActivationEl.value);
        }
        
        const trailingStopDistanceEl = document.getElementById('trailingStopDistance');
        if (trailingStopDistanceEl) {
            trailingStopDistanceEl.value = autoBotConfig.trailing_stop_distance || 150.0;
            console.log('[BotsManager] 📉 Расстояние trailing stop:', trailingStopDistanceEl.value);
        }
        
        const maxPositionHoursEl = document.getElementById('maxPositionHours');
        if (maxPositionHoursEl) {
            maxPositionHoursEl.value = autoBotConfig.max_position_hours || 0;
            console.log('[BotsManager] ⏰ Макс. время позиции (часов):', maxPositionHoursEl.value);
        }
        
        const breakEvenProtectionEl = document.getElementById('breakEvenProtection');
        if (breakEvenProtectionEl) {
            breakEvenProtectionEl.checked = autoBotConfig.break_even_protection !== false;
            console.log('[BotsManager] 🛡️ Защита безубыточности:', breakEvenProtectionEl.checked);
        }
        
        const breakEvenTriggerEl = document.getElementById('breakEvenTrigger');
        if (breakEvenTriggerEl) {
            breakEvenTriggerEl.value = autoBotConfig.break_even_trigger || 100.0;
            console.log('[BotsManager] 🎯 Триггер безубыточности:', breakEvenTriggerEl.value);
        }
        
        // ==========================================
        // ФИЛЬТРЫ ПО ТРЕНДУ
        // ==========================================
        
        const avoidDownTrendEl = document.getElementById('avoidDownTrend');
        if (avoidDownTrendEl) {
            avoidDownTrendEl.checked = autoBotConfig.avoid_down_trend !== false;
            console.log('[BotsManager] 📉 Избегать DOWN тренд:', avoidDownTrendEl.checked);
        }
        
        const avoidUpTrendEl = document.getElementById('avoidUpTrend');
        if (avoidUpTrendEl) {
            avoidUpTrendEl.checked = autoBotConfig.avoid_up_trend !== false;
            console.log('[BotsManager] 📈 Избегать UP тренд:', avoidUpTrendEl.checked);
        }
        
        // ==========================================
        // СИСТЕМНЫЕ НАСТРОЙКИ
        // ==========================================
        const systemConfig = config.system || {};
        
        // Интервалы обновления - ТОЛЬКО из API, без значений по умолчанию
        const rsiUpdateIntervalEl = document.getElementById('rsiUpdateInterval');
        if (rsiUpdateIntervalEl && systemConfig.rsi_update_interval !== undefined) {
            rsiUpdateIntervalEl.value = systemConfig.rsi_update_interval;
            console.log('[BotsManager] 🔄 RSI интервал установлен:', systemConfig.rsi_update_interval, '(из API)');
        } else if (rsiUpdateIntervalEl) {
            console.warn('[BotsManager] ⚠️ RSI интервал не найден в API, оставляем поле пустым');
        } else {
            console.error('[BotsManager] ❌ Элемент rsiUpdateInterval не найден!');
        }
        
        const autoSaveIntervalEl = document.getElementById('autoSaveInterval');
        if (autoSaveIntervalEl && systemConfig.auto_save_interval !== undefined) {
            autoSaveIntervalEl.value = systemConfig.auto_save_interval;
            console.log('[BotsManager] 💾 Автосохранение интервал установлен:', systemConfig.auto_save_interval, '(из API)');
        } else if (autoSaveIntervalEl) {
            console.warn('[BotsManager] ⚠️ Автосохранение интервал не найден в API, оставляем поле пустым');
        } else {
            console.error('[BotsManager] ❌ Элемент autoSaveInterval не найден!');
        }
        
        // Режим отладки
        const debugModeEl = document.getElementById('debugMode');
        if (debugModeEl) {
            debugModeEl.checked = systemConfig.debug_mode || false;
            console.log('[BotsManager] 🐛 Режим отладки:', debugModeEl.checked);
        }
        
        // Автообновление UI
        const autoRefreshUIEl = document.getElementById('autoRefreshUI');
        if (autoRefreshUIEl) {
            autoRefreshUIEl.checked = systemConfig.auto_refresh_ui !== false;
            console.log('[BotsManager] 🔄 Автообновление UI:', autoRefreshUIEl.checked);
        }
        
        // Интервал обновления UI
        const refreshIntervalEl = document.getElementById('refreshInterval');
        if (refreshIntervalEl && systemConfig.refresh_interval !== undefined) {
            refreshIntervalEl.value = systemConfig.refresh_interval;
            this.refreshInterval = systemConfig.refresh_interval * 1000;
            console.log('[BotsManager] 🔄 Интервал обновления UI установлен:', systemConfig.refresh_interval, 'сек (из API)');
        } else if (refreshIntervalEl) {
            refreshIntervalEl.value = 3; // Значение по умолчанию
            this.refreshInterval = 3000; // 3 секунды по умолчанию
            console.log('[BotsManager] 🔄 Интервал обновления UI установлен по умолчанию: 3 сек');
        }
        
        // ==========================================
        // ИНТЕРВАЛЫ СИНХРОНИЗАЦИИ И ОЧИСТКИ
        // ==========================================
        
        // Интервал синхронизации позиций
        const positionSyncIntervalEl = document.getElementById('positionSyncInterval');
        console.log('[BotsManager] 🔍 Поиск элемента positionSyncInterval:', positionSyncIntervalEl);
        console.log('[BotsManager] 🔍 systemConfig.position_sync_interval:', systemConfig.position_sync_interval);
        if (positionSyncIntervalEl && systemConfig.position_sync_interval !== undefined) {
            positionSyncIntervalEl.value = systemConfig.position_sync_interval;
            console.log('[BotsManager] 🔄 Position Sync интервал установлен:', systemConfig.position_sync_interval, 'сек (из API)');
        } else if (positionSyncIntervalEl) {
            positionSyncIntervalEl.value = 600; // 10 минут по умолчанию
            console.log('[BotsManager] 🔄 Position Sync интервал установлен по умолчанию: 600 сек');
        } else {
            console.error('[BotsManager] ❌ Элемент positionSyncInterval не найден!');
        }
        
        // Интервал очистки неактивных ботов
        const inactiveBotCleanupIntervalEl = document.getElementById('inactiveBotCleanupInterval');
        if (inactiveBotCleanupIntervalEl && systemConfig.inactive_bot_cleanup_interval !== undefined) {
            inactiveBotCleanupIntervalEl.value = systemConfig.inactive_bot_cleanup_interval;
            console.log('[BotsManager] 🧹 Inactive Bot Cleanup интервал установлен:', systemConfig.inactive_bot_cleanup_interval, 'сек (из API)');
        } else if (inactiveBotCleanupIntervalEl) {
            inactiveBotCleanupIntervalEl.value = 600; // 10 минут по умолчанию
            console.log('[BotsManager] 🧹 Inactive Bot Cleanup интервал установлен по умолчанию: 600 сек');
        }
        
        // Таймаут неактивных ботов
        const inactiveBotTimeoutEl = document.getElementById('inactiveBotTimeout');
        if (inactiveBotTimeoutEl && systemConfig.inactive_bot_timeout !== undefined) {
            inactiveBotTimeoutEl.value = systemConfig.inactive_bot_timeout;
            console.log('[BotsManager] ⏰ Inactive Bot Timeout установлен:', systemConfig.inactive_bot_timeout, 'сек (из API)');
        } else if (inactiveBotTimeoutEl) {
            inactiveBotTimeoutEl.value = 600; // 10 минут по умолчанию
            console.log('[BotsManager] ⏰ Inactive Bot Timeout установлен по умолчанию: 600 сек');
        }
        
        // Интервал настройки стоп-лоссов
        const stopLossSetupIntervalEl = document.getElementById('stopLossSetupInterval');
        if (stopLossSetupIntervalEl && systemConfig.stop_loss_setup_interval !== undefined) {
            stopLossSetupIntervalEl.value = systemConfig.stop_loss_setup_interval;
            console.log('[BotsManager] 🛡️ Stop Loss Setup интервал установлен:', systemConfig.stop_loss_setup_interval, 'сек (из API)');
        } else if (stopLossSetupIntervalEl) {
            stopLossSetupIntervalEl.value = 300; // 5 минут по умолчанию
            console.log('[BotsManager] 🛡️ Stop Loss Setup интервал установлен по умолчанию: 300 сек');
        }
        
        // ==========================================
        // RSI ВРЕМЕННОЙ ФИЛЬТР
        // ==========================================
        
        const rsiTimeFilterEnabledEl = document.getElementById('rsiTimeFilterEnabled');
        if (rsiTimeFilterEnabledEl) {
            rsiTimeFilterEnabledEl.checked = autoBotConfig.rsi_time_filter_enabled !== false;
            console.log('[BotsManager] ⏰ RSI временной фильтр:', rsiTimeFilterEnabledEl.checked);
        }
        
        const rsiTimeFilterCandlesEl = document.getElementById('rsiTimeFilterCandles');
        if (rsiTimeFilterCandlesEl) {
            rsiTimeFilterCandlesEl.value = autoBotConfig.rsi_time_filter_candles || 8;
            console.log('[BotsManager] 🕐 RSI временной фильтр (свечей):', rsiTimeFilterCandlesEl.value);
        }
        
        const rsiTimeFilterUpperEl = document.getElementById('rsiTimeFilterUpper');
        if (rsiTimeFilterUpperEl) {
            rsiTimeFilterUpperEl.value = autoBotConfig.rsi_time_filter_upper || 65;
            console.log('[BotsManager] 📈 RSI временной фильтр (верхняя граница):', rsiTimeFilterUpperEl.value);
        }
        
        const rsiTimeFilterLowerEl = document.getElementById('rsiTimeFilterLower');
        if (rsiTimeFilterLowerEl) {
            rsiTimeFilterLowerEl.value = autoBotConfig.rsi_time_filter_lower || 35;
            console.log('[BotsManager] 📉 RSI временной фильтр (нижняя граница):', rsiTimeFilterLowerEl.value);
        }
        
        // ==========================================
        // EXITSCAM ФИЛЬТР
        // ==========================================
        
        const exitScamEnabledEl = document.getElementById('exitScamEnabled');
        if (exitScamEnabledEl) {
            exitScamEnabledEl.checked = autoBotConfig.exit_scam_enabled !== false;
            console.log('[BotsManager] 🛡️ ExitScam фильтр:', exitScamEnabledEl.checked);
        }
        
        const exitScamCandlesEl = document.getElementById('exitScamCandles');
        if (exitScamCandlesEl) {
            exitScamCandlesEl.value = autoBotConfig.exit_scam_candles || 10;
            console.log('[BotsManager] 📊 ExitScam анализ свечей:', exitScamCandlesEl.value);
        }
        
        const exitScamSingleCandlePercentEl = document.getElementById('exitScamSingleCandlePercent');
        if (exitScamSingleCandlePercentEl) {
            exitScamSingleCandlePercentEl.value = autoBotConfig.exit_scam_single_candle_percent || 15.0;
            console.log('[BotsManager] ⚡ ExitScam лимит одной свечи:', exitScamSingleCandlePercentEl.value);
        }
        
        const exitScamMultiCandleCountEl = document.getElementById('exitScamMultiCandleCount');
        if (exitScamMultiCandleCountEl) {
            exitScamMultiCandleCountEl.value = autoBotConfig.exit_scam_multi_candle_count || 4;
            console.log('[BotsManager] 📈 ExitScam свечей для анализа:', exitScamMultiCandleCountEl.value);
        }
        
        const exitScamMultiCandlePercentEl = document.getElementById('exitScamMultiCandlePercent');
        if (exitScamMultiCandlePercentEl) {
            exitScamMultiCandlePercentEl.value = autoBotConfig.exit_scam_multi_candle_percent || 50.0;
            console.log('[BotsManager] 📊 ExitScam суммарный лимит:', exitScamMultiCandlePercentEl.value);
        }
        
        // ==========================================
        // НАСТРОЙКИ ЗРЕЛОСТИ МОНЕТ
        // ==========================================
        
        const enableMaturityCheckEl = document.getElementById('enableMaturityCheck');
        if (enableMaturityCheckEl) {
            enableMaturityCheckEl.checked = autoBotConfig.enable_maturity_check !== false;
            console.log('[BotsManager] 🔍 Проверка зрелости:', enableMaturityCheckEl.checked);
        }
        
        const minCandlesForMaturityEl = document.getElementById('minCandlesForMaturity');
        if (minCandlesForMaturityEl) {
            minCandlesForMaturityEl.value = autoBotConfig.min_candles_for_maturity || 200;
            console.log('[BotsManager] 📊 Мин. свечей для зрелости:', minCandlesForMaturityEl.value);
        }
        
        const minRsiLowEl = document.getElementById('minRsiLow');
        if (minRsiLowEl) {
            minRsiLowEl.value = autoBotConfig.min_rsi_low || 35;
            console.log('[BotsManager] 📉 Мин. RSI low:', minRsiLowEl.value);
        }
        
        const maxRsiHighEl = document.getElementById('maxRsiHigh');
        if (maxRsiHighEl) {
            maxRsiHighEl.value = autoBotConfig.max_rsi_high || 65;
            console.log('[BotsManager] 📈 Макс. RSI high:', maxRsiHighEl.value);
        }
        
        // ==========================================
        // ENHANCED RSI (УЛУЧШЕННАЯ СИСТЕМА RSI)
        // ==========================================
        
        const enhancedRsiEnabledEl = document.getElementById('enhancedRsiEnabled');
        if (enhancedRsiEnabledEl) {
            enhancedRsiEnabledEl.checked = systemConfig.enhanced_rsi_enabled || false;
            console.log('[BotsManager] 🧠 Enhanced RSI включен:', enhancedRsiEnabledEl.checked);
        }
        
        const enhancedRsiVolumeConfirmEl = document.getElementById('enhancedRsiVolumeConfirm');
        if (enhancedRsiVolumeConfirmEl) {
            enhancedRsiVolumeConfirmEl.checked = systemConfig.enhanced_rsi_require_volume_confirmation || false;
            console.log('[BotsManager] 📊 Enhanced RSI требует подтверждение объёмом:', enhancedRsiVolumeConfirmEl.checked);
        }
        
        const enhancedRsiDivergenceConfirmEl = document.getElementById('enhancedRsiDivergenceConfirm');
        if (enhancedRsiDivergenceConfirmEl) {
            enhancedRsiDivergenceConfirmEl.checked = systemConfig.enhanced_rsi_require_divergence_confirmation || false;
            console.log('[BotsManager] 📈 Enhanced RSI требует дивергенцию:', enhancedRsiDivergenceConfirmEl.checked);
        }
        
        const enhancedRsiUseStochRsiEl = document.getElementById('enhancedRsiUseStochRsi');
        if (enhancedRsiUseStochRsiEl) {
            enhancedRsiUseStochRsiEl.checked = systemConfig.enhanced_rsi_use_stoch_rsi || false;
            console.log('[BotsManager] 📊 Enhanced RSI использует Stoch RSI:', enhancedRsiUseStochRsiEl.checked);
        }
        
        const rsiExtremeZoneTimeoutEl = document.getElementById('rsiExtremeZoneTimeout');
        if (rsiExtremeZoneTimeoutEl) {
            rsiExtremeZoneTimeoutEl.value = systemConfig.rsi_extreme_zone_timeout || 3;
            console.log('[BotsManager] ⏰ RSI экстремальная зона таймаут:', rsiExtremeZoneTimeoutEl.value);
        }
        
        const rsiExtremeOversoldEl = document.getElementById('rsiExtremeOversold');
        if (rsiExtremeOversoldEl) {
            rsiExtremeOversoldEl.value = systemConfig.rsi_extreme_oversold || 20;
            console.log('[BotsManager] 📉 RSI экстремальный oversold:', rsiExtremeOversoldEl.value);
        }
        
        const rsiExtremeOverboughtEl = document.getElementById('rsiExtremeOverbought');
        if (rsiExtremeOverboughtEl) {
            rsiExtremeOverboughtEl.value = systemConfig.rsi_extreme_overbought || 80;
            console.log('[BotsManager] 📈 RSI экстремальный overbought:', rsiExtremeOverboughtEl.value);
        }
        
        const rsiVolumeMultiplierEl = document.getElementById('rsiVolumeMultiplier');
        if (rsiVolumeMultiplierEl) {
            rsiVolumeMultiplierEl.value = systemConfig.rsi_volume_confirmation_multiplier || 1.2;
            console.log('[BotsManager] 📊 RSI множитель объёма:', rsiVolumeMultiplierEl.value);
        }
        
        const rsiDivergenceLookbackEl = document.getElementById('rsiDivergenceLookback');
        if (rsiDivergenceLookbackEl) {
            rsiDivergenceLookbackEl.value = systemConfig.rsi_divergence_lookback || 10;
            console.log('[BotsManager] 🔍 RSI период поиска дивергенций:', rsiDivergenceLookbackEl.value);
        }
        
        // ==========================================
        // ПАРАМЕТРЫ ОПРЕДЕЛЕНИЯ ТРЕНДА
        // ==========================================
        
        const trendConfirmationBarsEl = document.getElementById('trendConfirmationBars');
        if (trendConfirmationBarsEl && systemConfig.trend_confirmation_bars !== undefined) {
            trendConfirmationBarsEl.value = systemConfig.trend_confirmation_bars;
            console.log('[BotsManager] 📊 Количество свечей для подтверждения тренда:', trendConfirmationBarsEl.value);
        }
        
        const trendMinConfirmationsEl = document.getElementById('trendMinConfirmations');
        if (trendMinConfirmationsEl && systemConfig.trend_min_confirmations !== undefined) {
            trendMinConfirmationsEl.value = systemConfig.trend_min_confirmations;
            console.log('[BotsManager] ✅ Минимум подтверждений тренда:', trendMinConfirmationsEl.value);
        }
        
        const trendRequireSlopeEl = document.getElementById('trendRequireSlope');
        if (trendRequireSlopeEl) {
            trendRequireSlopeEl.checked = systemConfig.trend_require_slope || false;
            console.log('[BotsManager] 📈 Требовать наклон EMA:', trendRequireSlopeEl.checked);
        }
        
        const trendRequirePriceEl = document.getElementById('trendRequirePrice');
        if (trendRequirePriceEl) {
            trendRequirePriceEl.checked = systemConfig.trend_require_price !== false;
            console.log('[BotsManager] 💰 Требовать позицию цены:', trendRequirePriceEl.checked);
        }
        
        const trendRequireCandlesEl = document.getElementById('trendRequireCandles');
        if (trendRequireCandlesEl) {
            trendRequireCandlesEl.checked = systemConfig.trend_require_candles !== false;
            console.log('[BotsManager] 🕯️ Требовать подтверждение свечами:', trendRequireCandlesEl.checked);
        }
        
        console.log('[BotsManager] ✅ Форма заполнена данными из API');
    }
    
    // ==========================================
    // ИНДИКАТОР ЗАГРУЗКИ КОНФИГУРАЦИИ
    // ==========================================
    
    showConfigurationLoading(show) {
        const configContainer = document.getElementById('configTab');
        if (!configContainer) return;
        
        if (show) {
            // Добавляем класс загрузки
            configContainer.classList.add('loading');
            
            // Отключаем все поля ввода
            const inputs = configContainer.querySelectorAll('input, select, button:not(.scope-btn)');
            inputs.forEach(input => {
                input.disabled = true;
                input.style.opacity = '0.6';
            });
            
            console.log('[BotsManager] ⏳ Конфигурация загружается...');
        } else {
            // Убираем класс загрузки
            configContainer.classList.remove('loading');
            
            // Включаем все поля ввода
            const inputs = configContainer.querySelectorAll('input, select, button');
            inputs.forEach(input => {
                input.disabled = false;
                input.style.opacity = '1';
            });
            
            console.log('[BotsManager] ✅ Конфигурация загружена');
        }
    }
    
    // ==========================================
    // МЕТОДЫ РАБОТЫ С КОНФИГУРАЦИЕЙ
    // ==========================================
    
    async loadConfigurationData() {
        this.logDebug('[BotsManager] 📋 Загрузка конфигурации...');
        
        try {
            this.logDebug('[BotsManager] 🌐 Запрос Auto Bot конфигурации...');
            // Загружаем конфигурацию Auto Bot
            const autoBotResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`);
            this.logDebug('[BotsManager] 📡 Auto Bot response status:', autoBotResponse.status);
            const autoBotData = await autoBotResponse.json();
            this.logDebug('[BotsManager] 🤖 Auto Bot data:', autoBotData);
            
            this.logDebug('[BotsManager] 🌐 Запрос системных настроек...');
            // Загружаем системные настройки
            const systemResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`);
            this.logDebug('[BotsManager] 📡 System response status:', systemResponse.status);
            const systemData = await systemResponse.json();
            this.logDebug('[BotsManager] ⚙️ System data:', systemData);
            
            if (autoBotData.success && systemData.success) {
                this.populateConfigurationForm({
                    autoBot: autoBotData.config,
                    system: systemData.config
                });
                
                // Обновляем RSI пороги из конфигурации
                this.updateRsiThresholds(autoBotData.config);
                
                console.log('[BotsManager] ✅ Конфигурация загружена');
                console.log('[BotsManager] Auto Bot config:', autoBotData.config);
                console.log('[BotsManager] System config:', systemData.config);
            } else {
                const errorMsg = !autoBotData.success ? autoBotData.message : systemData.message;
                console.error('[BotsManager] ❌ Ошибка загрузки конфигурации:', errorMsg);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка запроса конфигурации:', error);
        }
    }
    
    async saveDefaultConfiguration(defaultConfig) {
        console.log('[BotsManager] 💾 Сохранение конфигурации по умолчанию...');
        
        try {
            // Сохраняем Auto Bot настройки
            const autoBotResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(defaultConfig.autoBot)
            });
            
            // Сохраняем системные настройки
            const systemResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(defaultConfig.system)
            });
            
            const autoBotData = await autoBotResponse.json();
            const systemData = await systemResponse.json();
            
            if (autoBotData.success && systemData.success) {
                console.log('[BotsManager] ✅ Конфигурация по умолчанию сохранена');
                console.log('[BotsManager] 📊 Auto Bot сохранен:', autoBotData.saved_to_file);
                console.log('[BotsManager] 🔧 System config сохранен:', systemData.success);
                return true;
            } else {
                throw new Error(`API ошибка: ${autoBotData.message || systemData.message}`);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения конфигурации по умолчанию:', error);
            throw error;
        }
    }
    
    collectConfigurationData() {
        console.log('[BotsManager] 📋 Сбор данных конфигурации...');
        
        // 🔍 ОТЛАДКА: Проверяем наличие Enhanced RSI элементов
        console.log('[BotsManager] 🔍 Проверка Enhanced RSI элементов в DOM:');
        console.log('  enhancedRsiEnabled элемент:', !!document.getElementById('enhancedRsiEnabled'));
        console.log('  enhancedRsiVolumeConfirm элемент:', !!document.getElementById('enhancedRsiVolumeConfirm'));
        console.log('  enhancedRsiDivergenceConfirm элемент:', !!document.getElementById('enhancedRsiDivergenceConfirm'));
        console.log('  enhancedRsiUseStochRsi элемент:', !!document.getElementById('enhancedRsiUseStochRsi'));
        
        // Проверяем значения новых полей
        const positionSyncEl = document.getElementById('positionSyncInterval');
        const inactiveCleanupEl = document.getElementById('inactiveBotCleanupInterval');
        const inactiveTimeoutEl = document.getElementById('inactiveBotTimeout');
        const stopLossSetupEl = document.getElementById('stopLossSetupInterval');
        
        console.log('[BotsManager] 🔍 Значения полей:');
        console.log('  positionSyncInterval:', positionSyncEl?.value);
        console.log('  inactiveBotCleanupInterval:', inactiveCleanupEl?.value);
        console.log('  inactiveBotTimeout:', inactiveTimeoutEl?.value);
        console.log('  stopLossSetupInterval:', stopLossSetupEl?.value);
        
        // Собираем данные Auto Bot
        const autoBotConfig = {
            enabled: document.getElementById('globalAutoBotToggle')?.checked || false,
            max_concurrent: parseInt(document.getElementById('autoBotMaxConcurrent')?.value) || 5,
            risk_cap_percent: parseFloat(document.getElementById('autoBotRiskCap')?.value) || 10,
            scope: document.getElementById('autoBotScope')?.value || 'all',
            rsi_long_threshold: parseInt(document.getElementById('rsiLongThreshold')?.value) || 29,
            rsi_short_threshold: parseInt(document.getElementById('rsiShortThreshold')?.value) || 71,
            rsi_exit_long: parseInt(document.getElementById('rsiExitLong')?.value) || 65,
            rsi_exit_short: parseInt(document.getElementById('rsiExitShort')?.value) || 35,
            default_position_size: parseFloat(document.getElementById('defaultPositionSize')?.value) || 10,
            check_interval: parseInt(document.getElementById('checkInterval')?.value) || 180,
            max_loss_percent: parseFloat(document.getElementById('maxLossPercent')?.value) || 15.0,
            trailing_stop_activation: parseFloat(document.getElementById('trailingStopActivation')?.value) || 300.0,
            trailing_stop_distance: parseFloat(document.getElementById('trailingStopDistance')?.value) || 150.0,
            max_position_hours: parseInt(document.getElementById('maxPositionHours')?.value) || 0,
            break_even_protection: document.getElementById('breakEvenProtection')?.checked !== false,
            avoid_down_trend: document.getElementById('avoidDownTrend')?.checked !== false,
            avoid_up_trend: document.getElementById('avoidUpTrend')?.checked !== false,
            break_even_trigger: parseFloat(document.getElementById('breakEvenTrigger')?.value) || 100.0,
            enable_maturity_check: document.getElementById('enableMaturityCheck')?.checked !== false,
            min_candles_for_maturity: parseInt(document.getElementById('minCandlesForMaturity')?.value) || 200,
            min_rsi_low: parseInt(document.getElementById('minRsiLow')?.value) || 35,
            max_rsi_high: parseInt(document.getElementById('maxRsiHigh')?.value) || 65,
            min_volatility_threshold: parseFloat(document.getElementById('minVolatilityThreshold')?.value) || 0.05,
            // RSI временной фильтр
            rsi_time_filter_enabled: document.getElementById('rsiTimeFilterEnabled')?.checked !== false,
            rsi_time_filter_candles: parseInt(document.getElementById('rsiTimeFilterCandles')?.value) || 8,
            rsi_time_filter_upper: parseInt(document.getElementById('rsiTimeFilterUpper')?.value) || 65,
            rsi_time_filter_lower: parseInt(document.getElementById('rsiTimeFilterLower')?.value) || 35,
            // ExitScam фильтр
            exit_scam_enabled: document.getElementById('exitScamEnabled')?.checked !== false,
            exit_scam_candles: parseInt(document.getElementById('exitScamCandles')?.value) || 10,
            exit_scam_single_candle_percent: parseFloat(document.getElementById('exitScamSingleCandlePercent')?.value) || 15.0,
            exit_scam_multi_candle_count: parseInt(document.getElementById('exitScamMultiCandleCount')?.value) || 4,
            exit_scam_multi_candle_percent: parseFloat(document.getElementById('exitScamMultiCandlePercent')?.value) || 50.0,
            trading_enabled: document.getElementById('tradingEnabled')?.checked !== false,
            use_test_server: document.getElementById('useTestServer')?.checked || false,
            max_risk_per_trade: parseFloat(document.getElementById('maxRiskPerTrade')?.value) || 2.0,
            enhanced_rsi_enabled: (() => {
                const el = document.getElementById('enhancedRsiEnabled');
                const checked = el?.checked || false;
                console.log('[BotsManager] 🔍 Enhanced RSI Enabled - элемент:', !!el, 'значение:', checked);
                return checked;
            })(),
            enhanced_rsi_require_volume_confirmation: (() => {
                const el = document.getElementById('enhancedRsiVolumeConfirm');
                const checked = el?.checked || false;
                console.log('[BotsManager] 🔍 Enhanced RSI Volume - элемент:', !!el, 'значение:', checked);
                return checked;
            })(),
            enhanced_rsi_require_divergence_confirmation: (() => {
                const el = document.getElementById('enhancedRsiDivergenceConfirm');
                const checked = el?.checked || false;
                console.log('[BotsManager] 🔍 Enhanced RSI Divergence - элемент:', !!el, 'значение:', checked);
                return checked;
            })(),
            enhanced_rsi_use_stoch_rsi: (() => {
                const el = document.getElementById('enhancedRsiUseStochRsi');
                const checked = el?.checked || false;
                console.log('[BotsManager] 🔍 Enhanced RSI Stoch - элемент:', !!el, 'значение:', checked);
                return checked;
            })(),
            rsi_extreme_zone_timeout: parseInt(document.getElementById('rsiExtremeZoneTimeout')?.value) || 3,
            rsi_extreme_oversold: parseInt(document.getElementById('rsiExtremeOversold')?.value) || 20,
            rsi_extreme_overbought: parseInt(document.getElementById('rsiExtremeOverbought')?.value) || 80,
            rsi_volume_confirmation_multiplier: parseFloat(document.getElementById('rsiVolumeMultiplier')?.value) || 1.2,
            rsi_divergence_lookback: parseInt(document.getElementById('rsiDivergenceLookback')?.value) || 10
        };
        
        // Собираем системные настройки
        const systemConfig = {
            rsi_update_interval: parseInt(document.getElementById('rsiUpdateInterval')?.value) || 1800,
            auto_save_interval: parseInt(document.getElementById('autoSaveInterval')?.value) || 30,
            debug_mode: document.getElementById('debugMode')?.checked || false,
            auto_refresh_ui: document.getElementById('autoRefreshUI')?.checked !== false,
            refresh_interval: parseInt(document.getElementById('refreshInterval')?.value) || 3,
            position_sync_interval: parseInt(document.getElementById('positionSyncInterval')?.value) || 600,
            inactive_bot_cleanup_interval: parseInt(document.getElementById('inactiveBotCleanupInterval')?.value) || 600,
            inactive_bot_timeout: parseInt(document.getElementById('inactiveBotTimeout')?.value) || 600,
            stop_loss_setup_interval: parseInt(document.getElementById('stopLossSetupInterval')?.value) || 300,
            // EMA параметры тренда
            ema_fast: parseInt(document.getElementById('emaFast')?.value) || 50,
            ema_slow: parseInt(document.getElementById('emaSlow')?.value) || 200,
            // Параметры определения тренда
            trend_confirmation_bars: parseInt(document.getElementById('trendConfirmationBars')?.value) || 3,
            trend_min_confirmations: parseInt(document.getElementById('trendMinConfirmations')?.value) || 2,
            trend_require_slope: document.getElementById('trendRequireSlope')?.checked || false,
            trend_require_price: document.getElementById('trendRequirePrice')?.checked !== false,
            trend_require_candles: document.getElementById('trendRequireCandles')?.checked !== false,
            // Enhanced RSI настройки
            enhanced_rsi_enabled: autoBotConfig.enhanced_rsi_enabled,
            enhanced_rsi_require_volume_confirmation: autoBotConfig.enhanced_rsi_require_volume_confirmation,
            enhanced_rsi_require_divergence_confirmation: autoBotConfig.enhanced_rsi_require_divergence_confirmation,
            enhanced_rsi_use_stoch_rsi: autoBotConfig.enhanced_rsi_use_stoch_rsi,
            rsi_extreme_zone_timeout: autoBotConfig.rsi_extreme_zone_timeout,
            rsi_extreme_oversold: autoBotConfig.rsi_extreme_oversold,
            rsi_extreme_overbought: autoBotConfig.rsi_extreme_overbought,
            rsi_volume_confirmation_multiplier: autoBotConfig.rsi_volume_confirmation_multiplier,
            rsi_divergence_lookback: autoBotConfig.rsi_divergence_lookback
        };
        
        const result = {
            autoBot: autoBotConfig,
            system: systemConfig
        };
        
        // 🔍 ОТЛАДКА: Показываем итоговую конфигурацию Enhanced RSI
        console.log('[BotsManager] 🔍 ИТОГОВАЯ конфигурация Enhanced RSI:');
        console.log('  enhanced_rsi_enabled:', result.autoBot.enhanced_rsi_enabled);
        console.log('  enhanced_rsi_require_volume_confirmation:', result.autoBot.enhanced_rsi_require_volume_confirmation);
        console.log('  enhanced_rsi_require_divergence_confirmation:', result.autoBot.enhanced_rsi_require_divergence_confirmation);
        console.log('  enhanced_rsi_use_stoch_rsi:', result.autoBot.enhanced_rsi_use_stoch_rsi);
        
        return result;
    }

    // ✅ НОВЫЕ ФУНКЦИИ ДЛЯ СОХРАНЕНИЯ ОТДЕЛЬНЫХ БЛОКОВ
    
    async saveBasicSettings() {
        console.log('[BotsManager] 💾 Сохранение основных настроек...');
        try {
            const config = this.collectConfigurationData();
            const basicSettings = {
                enabled: config.autoBot.enabled,
                max_concurrent: config.autoBot.max_concurrent,
                risk_cap_percent: config.autoBot.risk_cap_percent,
                scope: config.autoBot.scope
            };
            
            await this.sendConfigUpdate('auto-bot', basicSettings, 'Основные настройки');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения основных настроек:', error);
            this.showNotification('❌ Ошибка сохранения основных настроек', 'error');
        }
    }
    
    async saveSystemSettings() {
        console.log('[BotsManager] 💾 Сохранение системных настроек...');
        try {
            const config = this.collectConfigurationData();
            const systemSettings = {
                rsi_update_interval: config.system.rsi_update_interval,
                auto_save_interval: config.system.auto_save_interval,
                debug_mode: config.system.debug_mode,
                auto_refresh_ui: config.system.auto_refresh_ui,
                refresh_interval: config.system.refresh_interval
            };
            
            await this.sendConfigUpdate('system-config', systemSettings, 'Системные настройки');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения системных настроек:', error);
            this.showNotification('❌ Ошибка сохранения системных настроек', 'error');
        }
    }
    
    async saveTradingParameters() {
        console.log('[BotsManager] 💾 Сохранение торговых параметров...');
        try {
            const config = this.collectConfigurationData();
            const tradingParams = {
                rsi_long_threshold: config.autoBot.rsi_long_threshold,
                rsi_short_threshold: config.autoBot.rsi_short_threshold,
                rsi_exit_long: config.autoBot.rsi_exit_long,
                rsi_exit_short: config.autoBot.rsi_exit_short,
                default_position_size: config.autoBot.default_position_size,
                check_interval: config.autoBot.check_interval
            };
            
            await this.sendConfigUpdate('auto-bot', tradingParams, 'Торговые параметры');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения торговых параметров:', error);
            this.showNotification('❌ Ошибка сохранения торговых параметров', 'error');
        }
    }
    
    async saveRsiExits() {
        console.log('[BotsManager] 💾 Сохранение RSI выходов...');
        try {
            const config = this.collectConfigurationData();
            const rsiExits = {
                rsi_exit_long: config.autoBot.rsi_exit_long,
                rsi_exit_short: config.autoBot.rsi_exit_short
            };
            
            await this.sendConfigUpdate('auto-bot', rsiExits, 'RSI выходы');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения RSI выходов:', error);
            this.showNotification('❌ Ошибка сохранения RSI выходов', 'error');
        }
    }
    
    async saveRsiTimeFilter() {
        console.log('[BotsManager] 💾 Сохранение RSI временного фильтра...');
        try {
            const config = this.collectConfigurationData();
            const rsiTimeFilter = {
                rsi_time_filter_enabled: config.autoBot.rsi_time_filter_enabled,
                rsi_time_filter_candles: config.autoBot.rsi_time_filter_candles,
                rsi_time_filter_upper: config.autoBot.rsi_time_filter_upper,
                rsi_time_filter_lower: config.autoBot.rsi_time_filter_lower
            };
            
            await this.sendConfigUpdate('auto-bot', rsiTimeFilter, 'RSI временной фильтр');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения RSI временного фильтра:', error);
            this.showNotification('❌ Ошибка сохранения RSI временного фильтра', 'error');
        }
    }
    
    async saveExitScamFilter() {
        console.log('[BotsManager] 💾 Сохранение ExitScam фильтра...');
        try {
            const config = this.collectConfigurationData();
            const exitScamFilter = {
                exit_scam_enabled: config.autoBot.exit_scam_enabled,
                exit_scam_candles: config.autoBot.exit_scam_candles,
                exit_scam_single_candle_percent: config.autoBot.exit_scam_single_candle_percent,
                exit_scam_multi_candle_count: config.autoBot.exit_scam_multi_candle_count,
                exit_scam_multi_candle_percent: config.autoBot.exit_scam_multi_candle_percent
            };
            
            await this.sendConfigUpdate('auto-bot', exitScamFilter, 'ExitScam фильтр');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения ExitScam фильтра:', error);
            this.showNotification('❌ Ошибка сохранения ExitScam фильтра', 'error');
        }
    }
    
    async saveEnhancedRsi() {
        console.log('[BotsManager] 💾 Сохранение Enhanced RSI...');
        try {
            const config = this.collectConfigurationData();
            const enhancedRsi = {
                enhanced_rsi_enabled: config.autoBot.enhanced_rsi_enabled,
                enhanced_rsi_require_volume_confirmation: config.autoBot.enhanced_rsi_require_volume_confirmation,
                enhanced_rsi_require_divergence_confirmation: config.autoBot.enhanced_rsi_require_divergence_confirmation,
                enhanced_rsi_use_stoch_rsi: config.autoBot.enhanced_rsi_use_stoch_rsi,
                rsi_extreme_zone_timeout: config.autoBot.rsi_extreme_zone_timeout,
                rsi_extreme_oversold: config.autoBot.rsi_extreme_oversold,
                rsi_extreme_overbought: config.autoBot.rsi_extreme_overbought,
                rsi_volume_confirmation_multiplier: config.autoBot.rsi_volume_confirmation_multiplier,
                rsi_divergence_lookback: config.autoBot.rsi_divergence_lookback
            };
            
            await this.sendConfigUpdate('auto-bot', enhancedRsi, 'Enhanced RSI');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения Enhanced RSI:', error);
            this.showNotification('❌ Ошибка сохранения Enhanced RSI', 'error');
        }
    }
    
    async saveTradingSettings() {
        console.log('[BotsManager] 💾 Сохранение торговых настроек...');
        try {
            const config = this.collectConfigurationData();
            const tradingSettings = {
                trading_enabled: config.autoBot.trading_enabled,
                use_test_server: config.autoBot.use_test_server,
                max_risk_per_trade: config.autoBot.max_risk_per_trade
            };
            
            await this.sendConfigUpdate('auto-bot', tradingSettings, 'Торговые настройки');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения торговых настроек:', error);
            this.showNotification('❌ Ошибка сохранения торговых настроек', 'error');
        }
    }
    
    async saveProtectiveMechanisms() {
        console.log('[BotsManager] 💾 Сохранение защитных механизмов...');
        try {
            const config = this.collectConfigurationData();
            const protectiveMechanisms = {
                max_loss_percent: config.autoBot.max_loss_percent,
                trailing_stop_activation: config.autoBot.trailing_stop_activation,
                trailing_stop_distance: config.autoBot.trailing_stop_distance,
                max_position_hours: config.autoBot.max_position_hours,
                break_even_protection: config.autoBot.break_even_protection,
                break_even_trigger: config.autoBot.break_even_trigger,
                avoid_down_trend: config.autoBot.avoid_down_trend,
                avoid_up_trend: config.autoBot.avoid_up_trend
            };
            
            await this.sendConfigUpdate('auto-bot', protectiveMechanisms, 'Защитные механизмы');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения защитных механизмов:', error);
            this.showNotification('❌ Ошибка сохранения защитных механизмов', 'error');
        }
    }
    
    async saveMaturitySettings() {
        console.log('[BotsManager] 💾 Сохранение настроек зрелости...');
        try {
            const config = this.collectConfigurationData();
            const maturitySettings = {
                enable_maturity_check: config.autoBot.enable_maturity_check,
                min_candles_for_maturity: config.autoBot.min_candles_for_maturity,
                min_rsi_low: config.autoBot.min_rsi_low,
                max_rsi_high: config.autoBot.max_rsi_high,
                min_volatility_threshold: config.autoBot.min_volatility_threshold
            };
            
            await this.sendConfigUpdate('auto-bot', maturitySettings, 'Настройки зрелости');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения настроек зрелости:', error);
            this.showNotification('❌ Ошибка сохранения настроек зрелости', 'error');
        }
    }
    
    async saveEmaParameters() {
        console.log('[BotsManager] 💾 Сохранение EMA параметров...');
        try {
            const config = this.collectConfigurationData();
            const emaParameters = {
                ema_fast: config.system.ema_fast,
                ema_slow: config.system.ema_slow,
                trend_confirmation_bars: config.system.trend_confirmation_bars
            };
            
            await this.sendConfigUpdate('system-config', emaParameters, 'EMA параметры');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения EMA параметров:', error);
            this.showNotification('❌ Ошибка сохранения EMA параметров', 'error');
        }
    }
    
    async saveTrendParameters() {
        console.log('[BotsManager] 💾 Сохранение параметров определения тренда...');
        try {
            const config = this.collectConfigurationData();
            const trendParameters = {
                trend_confirmation_bars: config.system.trend_confirmation_bars,
                trend_min_confirmations: config.system.trend_min_confirmations,
                trend_require_slope: config.system.trend_require_slope,
                trend_require_price: config.system.trend_require_price,
                trend_require_candles: config.system.trend_require_candles
            };
            
            console.log('[BotsManager] 📊 Параметры тренда для сохранения:', trendParameters);
            
            await this.sendConfigUpdate('system-config', trendParameters, 'Параметры определения тренда');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения параметров тренда:', error);
            this.showNotification('❌ Ошибка сохранения параметров тренда', 'error');
        }
    }
    
    // ✅ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ОТПРАВКИ КОНФИГУРАЦИИ
    async sendConfigUpdate(endpoint, data, sectionName) {
        this.showConfigurationLoading(true);
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            if (response.ok) {
                this.showNotification(`✅ ${sectionName} сохранены успешно`, 'success');
                console.log(`[BotsManager] ✅ ${sectionName} сохранены успешно`);
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } finally {
            this.showConfigurationLoading(false);
        }
    }

    async saveConfiguration() {
        console.log('[BotsManager] 💾 Сохранение конфигурации...');
        
        try {
            const config = this.collectConfigurationData();
            
            // Отладочные логи для Enhanced RSI
            console.log('[BotsManager] 🔍 Отправляемая конфигурация Enhanced RSI:');
            console.log('  enhanced_rsi_enabled:', config.autoBot.enhanced_rsi_enabled);
            console.log('  enhanced_rsi_require_volume_confirmation:', config.autoBot.enhanced_rsi_require_volume_confirmation);
            console.log('  enhanced_rsi_require_divergence_confirmation:', config.autoBot.enhanced_rsi_require_divergence_confirmation);
            console.log('  enhanced_rsi_use_stoch_rsi:', config.autoBot.enhanced_rsi_use_stoch_rsi);
            
            // Показываем индикатор загрузки
            this.showConfigurationLoading(true);
            
            // Сохраняем Auto Bot настройки
            const autoBotResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config.autoBot)
            });
            
            // Сохраняем системные настройки
            const systemResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config.system)
            });
            
            const autoBotData = await autoBotResponse.json();
            const systemData = await systemResponse.json();
            
            if (autoBotData.success && systemData.success) {
                this.showNotification('✅ Конфигурация сохранена в bot_config.py! Изменения применены автоматически.', 'success');
                console.log('[BotsManager] ✅ Конфигурация сохранена в bot_config.py и перезагружена');
                console.log('[BotsManager] 📊 Auto Bot сохранен:', autoBotData.saved_to_file);
                console.log('[BotsManager] 🔧 System config сохранен:', systemData.saved_to_file);
                
                // ✅ ОБНОВЛЯЕМ RSI ПОРОГИ (для фильтров и подписей)
                if (config.autoBot) {
                    this.updateRsiThresholds(config.autoBot);
                    console.log('[BotsManager] 🔄 RSI пороги обновлены после сохранения');
                }
                
                // ✅ ПЕРЕЗАГРУЖАЕМ КОНФИГУРАЦИЮ (чтобы UI отображал актуальные значения)
                setTimeout(() => {
                    console.log('[BotsManager] 🔄 Перезагрузка конфигурации для обновления UI...');
                    this.loadConfigurationData();
                }, 500);
                
                // ✅ ПЕРЕЗАГРУЖАЕМ ДАННЫЕ RSI (чтобы применить новые фильтры)
                setTimeout(() => {
                    console.log('[BotsManager] 🔄 Перезагрузка RSI данных для применения новых настроек...');
                    this.loadCoinsRsiData();
                }, 1000);
            } else {
                const errorMsg = !autoBotData.success ? autoBotData.message : systemData.message;
                throw new Error(`API ошибка: ${errorMsg}`);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения конфигурации:', error);
            this.showNotification('❌ Ошибка сохранения конфигурации: ' + error.message, 'error');
        } finally {
            this.showConfigurationLoading(false);
        }
    }
    
    async resetConfiguration() {
        console.log('[BotsManager] 🔄 Сброс конфигурации к умолчаниям...');
        
        if (!confirm('Вы уверены, что хотите сбросить конфигурацию к умолчаниям?')) {
            return;
        }
        
        try {
            // Загружаем конфигурацию по умолчанию
            const defaultConfig = {
                autoBot: {
                    enabled: false,
                    max_concurrent: 5,
                    risk_cap_percent: 10,
                    scope: 'all',
                    rsi_long_threshold: 29,
                    rsi_short_threshold: 71,
                    rsi_exit_long: 65,
                    rsi_exit_short: 35,
                    default_position_size: 10,
                    check_interval: 180,
                    max_loss_percent: 15.0,
                    trailing_stop_activation: 300.0,
                    trailing_stop_distance: 150.0,
                    max_position_hours: 0,
                    break_even_protection: true,
                    avoid_down_trend: true,
                    avoid_up_trend: true,
                    break_even_trigger: 100.0,
                    enable_maturity_check: true,
                    min_candles_for_maturity: 200,
                    min_rsi_low: 35,
                    max_rsi_high: 65,
                    trading_enabled: true,
                    use_test_server: false,
                    max_risk_per_trade: 2.0,
                    enhanced_rsi_enabled: true,
                    enhanced_rsi_require_volume_confirmation: true,
                    enhanced_rsi_require_divergence_confirmation: false,
                    enhanced_rsi_use_stoch_rsi: true,
                    rsi_extreme_zone_timeout: 3,
                    rsi_extreme_oversold: 20,
                    rsi_extreme_overbought: 80,
                    rsi_volume_confirmation_multiplier: 1.2,
                    rsi_divergence_lookback: 10
                },
                system: {
                    rsi_update_interval: 1800,
                    auto_save_interval: 30,
                    debug_mode: false,
                    auto_refresh_ui: true,
                    refresh_interval: 3,
                    position_sync_interval: 600,
                    inactive_bot_cleanup_interval: 600,
                    inactive_bot_timeout: 600,
                    stop_loss_setup_interval: 300
                }
            };
            
            await this.saveDefaultConfiguration(defaultConfig);
            this.showNotification('✅ Конфигурация сброшена к умолчаниям!', 'success');
            
            // Перезагружаем конфигурацию
            await this.loadConfigurationData();
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сброса конфигурации:', error);
            this.showNotification('❌ Ошибка сброса конфигурации: ' + error.message, 'error');
        }
    }

    testConfiguration() {
        console.log('[BotsManager] 🧪 Тестирование конфигурации...');
        const config = this.collectConfigurationData();
        
        // Простая валидация
        let errors = [];
        
        if (config.autoBot.rsi_long_threshold >= config.autoBot.rsi_short_threshold) {
            errors.push('RSI для LONG должен быть меньше RSI для SHORT');
        }
        
        if (config.autoBot.rsi_exit_long <= config.autoBot.rsi_long_threshold) {
            errors.push('RSI выход из LONG должен быть больше RSI входа в LONG');
        }
        
        if (config.autoBot.rsi_exit_short >= config.autoBot.rsi_short_threshold) {
            errors.push('RSI выход из SHORT должен быть меньше RSI входа в SHORT');
        }
        
        if (config.autoBot.max_loss_percent <= 0 || config.autoBot.max_loss_percent > 50) {
            errors.push('Стоп-лосс должен быть от 1% до 50%');
        }
        
        if (config.autoBot.trailing_stop_activation < config.autoBot.break_even_trigger) {
            errors.push('Активация Trailing Stop должна быть больше триггера безубыточности');
        }
        
        if (errors.length > 0) {
            this.showNotification('❌ Ошибки конфигурации:\n' + errors.join('\n'), 'error');
        } else {
            this.showNotification('✅ Конфигурация корректна!', 'success');
        }
    }
    
    syncDuplicateSettings(config) {
        console.log('[BotsManager] 🔄 Синхронизация дублированных настроек...');
        
        // КРИТИЧЕСКИ ВАЖНО: Синхронизируем переключатель Auto Bot на главной странице
        const globalAutoBotToggleEl = document.getElementById('globalAutoBotToggle');
        if (globalAutoBotToggleEl) {
            const enabled = config.enabled || false;
            globalAutoBotToggleEl.checked = enabled;
            console.log(`[BotsManager] 🤖 Auto Bot переключатель синхронизирован: ${enabled}`);
            
            // Обновляем визуальное состояние
            const toggleLabel = globalAutoBotToggleEl.closest('.auto-bot-toggle')?.querySelector('.toggle-label');
            if (toggleLabel) {
                toggleLabel.textContent = enabled ? '🤖 Auto Bot (ВКЛ)' : '🤖 Auto Bot (ВЫКЛ)';
            }
        }
        
        // Синхронизируем мобильный переключатель Auto Bot
        const mobileAutoBotToggleEl = document.getElementById('mobileAutobotToggle');
        if (mobileAutoBotToggleEl) {
            const enabled = config.enabled || false;
            mobileAutoBotToggleEl.checked = enabled;
            console.log(`[BotsManager] 🤖 Мобильный Auto Bot переключатель синхронизирован: ${enabled}`);
            
            // Обновляем визуальное состояние
            const statusText = document.getElementById('mobileAutobotStatusText');
            if (statusText) {
                statusText.textContent = enabled ? 'ВКЛ' : 'ВЫКЛ';
                statusText.className = enabled ? 'mobile-autobot-status enabled' : 'mobile-autobot-status';
            }
        }
        
        // Синхронизируем дублированные элементы на вкладке "Управление"
        const rsiLongDupEl = document.getElementById('rsiLongThresholdDup');
        if (rsiLongDupEl) rsiLongDupEl.value = config.rsi_long_threshold || 29;
        
        const rsiShortDupEl = document.getElementById('rsiShortThresholdDup');
        if (rsiShortDupEl) rsiShortDupEl.value = config.rsi_short_threshold || 71;
        
        const rsiExitLongDupEl = document.getElementById('rsiExitLongDup');
        if (rsiExitLongDupEl) rsiExitLongDupEl.value = config.rsi_exit_long || 65;
        
        const rsiExitShortDupEl = document.getElementById('rsiExitShortDup');
        if (rsiExitShortDupEl) rsiExitShortDupEl.value = config.rsi_exit_short || 35;
        
        const maxLossDupEl = document.getElementById('maxLossPercentDup');
        if (maxLossDupEl) maxLossDupEl.value = config.max_loss_percent || 15.0;
        
        const trailingActivationDupEl = document.getElementById('trailingStopActivationDup');
        if (trailingActivationDupEl) trailingActivationDupEl.value = config.trailing_stop_activation || 300.0;
        
        const trailingDistanceDupEl = document.getElementById('trailingStopDistanceDup');
        if (trailingDistanceDupEl) trailingDistanceDupEl.value = config.trailing_stop_distance || 150.0;
        
        const maxHoursDupEl = document.getElementById('maxPositionHoursDup');
        if (maxHoursDupEl) {
            const seconds = config.max_position_hours || 0;
            // Конвертируем секунды в минуты для отображения
            const minutes = Math.round(seconds / 60);
            maxHoursDupEl.value = minutes;
        }
        
        const breakEvenDupEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenDupEl) breakEvenDupEl.checked = config.break_even_protection !== false;
        
        const avoidDownTrendDupEl = document.getElementById('avoidDownTrendDup');
        if (avoidDownTrendDupEl) avoidDownTrendDupEl.checked = config.avoid_down_trend !== false;
        
        const avoidUpTrendDupEl = document.getElementById('avoidUpTrendDup');
        if (avoidUpTrendDupEl) avoidUpTrendDupEl.checked = config.avoid_up_trend !== false;
        
        const enableMaturityCheckDupEl = document.getElementById('enableMaturityCheckDup');
        if (enableMaturityCheckDupEl) enableMaturityCheckDupEl.checked = config.enable_maturity_check !== false;
        
        const breakEvenTriggerDupEl = document.getElementById('breakEvenTriggerDup');
        if (breakEvenTriggerDupEl) breakEvenTriggerDupEl.value = config.break_even_trigger || 100.0;
        
        console.log('[BotsManager] ✅ Дублированные настройки синхронизированы');
        
        // Обновляем подписи тренд-фильтров после синхронизации
        this.updateTrendFilterLabels();
    }
    
    async loadDuplicateSettings() {
        console.log('[BotsManager] 📋 Загрузка дублированных настроек...');
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`);
            const data = await response.json();
            
            if (data.success && data.config) {
                this.syncDuplicateSettings(data.config);
                this.initializeGlobalAutoBotToggle();
            this.initializeMobileAutoBotToggle();
                
                // Обновляем RSI пороги из конфигурации
                this.updateRsiThresholds(data.config);
                
                console.log('[BotsManager] ✅ Дублированные настройки загружены');
            } else {
                console.error('[BotsManager] ❌ Ошибка загрузки дублированных настроек:', data.message);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка запроса дублированных настроек:', error);
        }
    }
    
    async initializeGlobalAutoBotToggle() {
        const globalAutoBotToggleEl = document.getElementById('globalAutoBotToggle');
        console.log('[BotsManager] 🔍 initializeGlobalAutoBotToggle вызван');
        console.log('[BotsManager] 🔍 Элемент найден:', !!globalAutoBotToggleEl);
        console.log('[BotsManager] 🔍 data-initialized:', globalAutoBotToggleEl?.getAttribute('data-initialized'));
        
        if (globalAutoBotToggleEl && !globalAutoBotToggleEl.hasAttribute('data-initialized')) {
            console.log('[BotsManager] 🔧 Устанавливаем обработчик события...');
            globalAutoBotToggleEl.setAttribute('data-initialized', 'true');
            
            // КРИТИЧЕСКИ ВАЖНО: Загружаем текущее состояние Auto Bot с сервера
            try {
                console.log('[BotsManager] 🔄 Загрузка текущего состояния Auto Bot...');
                const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`);
                const data = await response.json();
                
                if (data.success && data.config) {
                    const autoBotEnabled = data.config.enabled;
                    console.log('[BotsManager] 🤖 Текущее состояние Auto Bot с сервера:', autoBotEnabled ? 'ВКЛ' : 'ВЫКЛ');
                    
                    // Устанавливаем состояние переключателя
                    globalAutoBotToggleEl.checked = autoBotEnabled;
                    
                    // Обновляем визуальное состояние
                    const toggleLabel = globalAutoBotToggleEl.closest('.auto-bot-toggle')?.querySelector('.toggle-label');
                    if (toggleLabel) {
                        toggleLabel.textContent = autoBotEnabled ? '🤖 Auto Bot (ВКЛ)' : '🤖 Auto Bot (ВЫКЛ)';
                    }
                    
                    console.log('[BotsManager] ✅ Переключатель Auto Bot инициализирован с состоянием:', autoBotEnabled);
                } else {
                    console.error('[BotsManager] ❌ Ошибка загрузки состояния Auto Bot:', data.message);
                }
            } catch (error) {
                console.error('[BotsManager] ❌ Ошибка запроса состояния Auto Bot:', error);
            }
            
            globalAutoBotToggleEl.addEventListener('change', async (e) => {
                const isEnabled = e.target.checked;
                console.log(`[BotsManager] 🤖 ИЗМЕНЕНИЕ ПЕРЕКЛЮЧАТЕЛЯ: ${isEnabled}`);
                
                // Помечаем, что пользователь изменил переключатель
                globalAutoBotToggleEl.setAttribute('data-user-changed', 'true');
                console.log('[BotsManager] 🔒 Флаг data-user-changed установлен');
                
                // Обновляем визуальное состояние сразу
                const toggleLabel = globalAutoBotToggleEl.closest('.auto-bot-toggle')?.querySelector('.toggle-label');
                if (toggleLabel) {
                    toggleLabel.textContent = isEnabled ? '🤖 Auto Bot (ВКЛ)' : '🤖 Auto Bot (ВЫКЛ)';
                }
                
                try {
                    const url = `${this.BOTS_SERVICE_URL}/api/bots/auto-bot`;
                    console.log(`[BotsManager] 📡 Отправка запроса на ${isEnabled ? 'включение' : 'выключение'} автобота...`);
                    console.log(`[BotsManager] 🌐 URL: ${url}`);
                    console.log(`[BotsManager] 📦 Данные: ${JSON.stringify({ enabled: isEnabled })}`);
                    // Сохраняем изменение через API
                    const response = await fetch(url, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ enabled: isEnabled })
                    });
                    console.log('[BotsManager] 📡 Ответ получен:', response.status);
                    
                    const result = await response.json();
                    console.log('[BotsManager] 📦 Результат от сервера:', result);
                    console.log('[BotsManager] 📊 Состояние enabled в ответе:', result.config?.enabled);
                    
                    if (result.success) {
                        this.showNotification(
                            isEnabled ? '✅ Auto Bot включен' : '⏸️ Auto Bot выключен', 
                            'success'
                        );
                        
                        // Синхронизируем с мобильным переключателем
                        const mobileToggle = document.getElementById('mobileAutobotToggle');
                        if (mobileToggle) {
                            mobileToggle.checked = isEnabled;
                            const mobileStatusText = document.getElementById('mobileAutobotStatusText');
                            if (mobileStatusText) {
                                mobileStatusText.textContent = isEnabled ? 'ВКЛ' : 'ВЫКЛ';
                                mobileStatusText.className = isEnabled ? 'mobile-autobot-status enabled' : 'mobile-autobot-status';
                            }
                            console.log(`[BotsManager] 🔄 Мобильный переключатель синхронизирован: ${isEnabled}`);
                        }
                        
                        // ✅ ИСПРАВЛЕНИЕ: Сбрасываем флаг с задержкой
                        // Даем время автообновлению получить новое состояние с сервера
                        setTimeout(() => {
                            globalAutoBotToggleEl.removeAttribute('data-user-changed');
                            console.log('[BotsManager] 🔓 Флаг data-user-changed снят после задержки');
                        }, 15000);  // 15 секунд - достаточно для автообновления
                        
                        console.log(`[BotsManager] ✅ Auto Bot ${isEnabled ? 'включен' : 'выключен'} и сохранен`);
                } else {
                        console.error('[BotsManager] ❌ Ошибка сохранения Auto Bot:', result.message);
                        // НЕ возвращаем переключатель в исходное состояние при ошибке API
                        // Пользователь может попробовать снова
                        this.showNotification('❌ Ошибка сохранения: ' + result.message, 'error');
                    }
                    
                } catch (error) {
                    console.error('[BotsManager] ❌ Ошибка изменения Auto Bot:', error);
                    // НЕ возвращаем переключатель в исходное состояние при ошибке соединения
                    // Пользователь может попробовать снова
                    this.showNotification('❌ Ошибка соединения с сервисом. Попробуйте еще раз.', 'error');
                }
            });
            
            console.log('[BotsManager] ✅ Обработчик главного переключателя Auto Bot инициализирован');
        }
    }

    initializeMobileAutoBotToggle() {
        const mobileAutoBotToggleEl = document.getElementById('mobileAutobotToggle');
        console.log('[BotsManager] 🔍 initializeMobileAutoBotToggle вызван');
        console.log('[BotsManager] 🔍 Мобильный элемент найден:', !!mobileAutoBotToggleEl);
        console.log('[BotsManager] 🔍 data-initialized:', mobileAutoBotToggleEl?.getAttribute('data-initialized'));
        
        if (mobileAutoBotToggleEl && !mobileAutoBotToggleEl.hasAttribute('data-initialized')) {
            console.log('[BotsManager] 🔧 Устанавливаем обработчик события для мобильного переключателя...');
            mobileAutoBotToggleEl.setAttribute('data-initialized', 'true');
            
            mobileAutoBotToggleEl.addEventListener('change', async (e) => {
                const isEnabled = e.target.checked;
                console.log(`[BotsManager] 🤖 ИЗМЕНЕНИЕ МОБИЛЬНОГО ПЕРЕКЛЮЧАТЕЛЯ: ${isEnabled}`);
                
                // Помечаем, что пользователь изменил переключатель
                mobileAutoBotToggleEl.setAttribute('data-user-changed', 'true');
                console.log('[BotsManager] 🔒 Флаг data-user-changed установлен для мобильного');
                
                // Обновляем визуальное состояние сразу
                const statusText = document.getElementById('mobileAutobotStatusText');
                if (statusText) {
                    statusText.textContent = isEnabled ? 'ВКЛ' : 'ВЫКЛ';
                    statusText.className = isEnabled ? 'mobile-autobot-status enabled' : 'mobile-autobot-status';
                }
                
                try {
                    const url = `${this.BOTS_SERVICE_URL}/api/bots/auto-bot`;
                    console.log(`[BotsManager] 📡 Отправка запроса на ${isEnabled ? 'включение' : 'выключение'} автобота...`);
                    console.log(`[BotsManager] 🌐 URL: ${url}`);
                    console.log(`[BotsManager] 📦 Данные: ${JSON.stringify({ enabled: isEnabled })}`);
                    
                    // Сохраняем изменение через API
                    const response = await fetch(url, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ enabled: isEnabled })
                    });
                    
                    const result = await response.json();
                    console.log('[BotsManager] 📨 Ответ сервера:', result);
                    
                    if (result.success) {
                        console.log(`[BotsManager] ✅ Auto Bot ${isEnabled ? 'включен' : 'выключен'} успешно`);
                        this.showNotification(`✅ Auto Bot ${isEnabled ? 'включен' : 'выключен'}`, 'success');
                        
                        // Синхронизируем с основным переключателем
                        const globalToggle = document.getElementById('globalAutoBotToggle');
                        if (globalToggle) {
                            globalToggle.checked = isEnabled;
                            const globalLabel = globalToggle.closest('.auto-bot-toggle')?.querySelector('.toggle-label');
                            if (globalLabel) {
                                globalLabel.textContent = isEnabled ? '🤖 Auto Bot (ВКЛ)' : '🤖 Auto Bot (ВЫКЛ)';
                            }
                            console.log(`[BotsManager] 🔄 Основной переключатель синхронизирован: ${isEnabled}`);
                        }
                        
                        // Убираем флаг изменения после успешного сохранения с задержкой
                        setTimeout(() => {
                            mobileAutoBotToggleEl.removeAttribute('data-user-changed');
                            console.log('[BotsManager] 🔓 Флаг data-user-changed снят для мобильного после задержки');
                        }, 15000);  // 15 секунд - достаточно для автообновления
                        
                    } else {
                        console.error('[BotsManager] ❌ Ошибка сервера:', result.message);
                        this.showNotification('❌ Ошибка сохранения: ' + result.message, 'error');
                    }
                    
                } catch (error) {
                    console.error('[BotsManager] ❌ Ошибка изменения Auto Bot:', error);
                    this.showNotification('❌ Ошибка соединения с сервисом. Попробуйте еще раз.', 'error');
                }
            });
            
            console.log('[BotsManager] ✅ Обработчик мобильного переключателя Auto Bot инициализирован');
        }
    }
    
    // ==========================================
    // МЕТОДЫ РАБОТЫ С ЕДИНЫМ ТОРГОВЫМ СЧЕТОМ
    // ==========================================
    
    async loadAccountInfo() {
        this.logDebug('[BotsManager] 💰 Загрузка информации о едином торговом счете...');
        
        try {
            // Используем тот же эндпоинт, что и страница Позиции
            const response = await fetch('/api/positions');
            const data = await response.json();
            
            if (data.wallet_data) {
                // Преобразуем данные в формат, ожидаемый updateAccountDisplay
                const accountData = {
                    success: true,
                    total_wallet_balance: data.wallet_data.total_balance,
                    total_available_balance: data.wallet_data.available_balance,
                    total_unrealized_pnl: data.wallet_data.realized_pnl, // Используем realized_pnl как unrealized
                    active_positions: data.stats?.total_trades || 0,
                    active_bots: this.activeBots?.length || 0
                };
                this.updateAccountDisplay(accountData);
                this.logDebug('[BotsManager] ✅ Информация о счете загружена:', accountData);
            } else {
                console.warn('[BotsManager] ⚠️ Нет данных аккаунта в ответе');
                this.updateAccountDisplay(null);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка запроса информации о счете:', error);
            this.updateAccountDisplay(null);
        }
    }
    
    updateAccountDisplay(accountData) {
        // Ищем заголовок h3 в правой панели
        const activeBotsHeader = document.querySelector('.active-bots-header h3');
        if (!activeBotsHeader) return;
        
        if (accountData && accountData.success) {
            const balance = parseFloat(accountData.total_wallet_balance || 0);
            const available = parseFloat(accountData.total_available_balance || 0);
            const pnl = parseFloat(accountData.total_unrealized_pnl || 0);
            const positions = parseInt(accountData.active_positions || 0);
            const activeBots = parseInt(accountData.active_bots || 0);
            
            const balanceText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['balance']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['balance'] : 'Баланс';
            const remainderText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['remainder']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['remainder'] : 'Остаток';
            const openPositionsText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['open_positions']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['open_positions'] : 'Открытых позиций';
            
            activeBotsHeader.innerHTML = `
                ${balanceText}  $${balance.toFixed(2)}<br>
                ${remainderText}  $${available.toFixed(2)}<br>
                PnL  ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}<br>
                ${openPositionsText}  ${positions}
            `;
        } else {
            const balanceText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['balance']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['balance'] : 'Баланс';
            const remainderText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['remainder']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['remainder'] : 'Остаток';
            const openPositionsText = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['open_positions']) ? TRANSLATIONS[document.documentElement.lang || 'ru']['open_positions'] : 'Открытых позиций';
            
            activeBotsHeader.innerHTML = `
                ${balanceText}  -<br>
                ${remainderText}  -<br>
                PnL  -<br>
                ${openPositionsText}  -
            `;
        }
        
        // Автообновление происходит через основной интервал в startPeriodicUpdate()
        // Не создаем отдельный интервал для accountInfo
    }
    
    // ==========================================
    // МАССОВЫЕ ОПЕРАЦИИ С БОТАМИ
    // ==========================================
    
    updateBulkControlsVisibility(bots) {
        const bulkControlsEl = document.getElementById('bulkBotControls');
        const countEl = document.getElementById('bulkControlsCount');
        
        if (bulkControlsEl && countEl) {
            if (bots && bots.length > 0) {
                bulkControlsEl.style.display = 'block';
                countEl.textContent = `${bots.length} ${bots.length === 1 ? 'бот' : 'ботов'}`;
                this.initializeBulkControls(bots);
            } else {
                bulkControlsEl.style.display = 'none';
            }
        }
    }
    
    initializeBulkControls(bots) {
        const startAllBtn = document.getElementById('startAllBotsBtn');
        const stopAllBtn = document.getElementById('stopAllBotsBtn');
        const deleteAllBtn = document.getElementById('deleteAllBotsBtn');
        
        if (startAllBtn && !startAllBtn.hasAttribute('data-initialized')) {
            startAllBtn.setAttribute('data-initialized', 'true');
            startAllBtn.addEventListener('click', () => this.startAllBots());
        }
        
        if (stopAllBtn && !stopAllBtn.hasAttribute('data-initialized')) {
            stopAllBtn.setAttribute('data-initialized', 'true');
            stopAllBtn.addEventListener('click', () => this.stopAllBots());
        }
        
        if (deleteAllBtn && !deleteAllBtn.hasAttribute('data-initialized')) {
            deleteAllBtn.setAttribute('data-initialized', 'true');
            deleteAllBtn.addEventListener('click', () => this.deleteAllBots());
        }
    }
    
    initializeConfigurationButtons() {
        console.log('[BotsManager] ⚙️ Инициализация кнопок конфигурации...');
        
        // Обработчик кнопки сохранения конфигурации
        const saveConfigBtn = document.getElementById('saveConfigBtn');
        if (saveConfigBtn && !saveConfigBtn.hasAttribute('data-initialized')) {
            saveConfigBtn.setAttribute('data-initialized', 'true');
            saveConfigBtn.addEventListener('click', () => this.saveConfiguration());
            console.log('[BotsManager] ✅ Кнопка "Сохранить конфигурацию" инициализирована');
        }
        
        // Обработчик кнопки сброса конфигурации
        const resetConfigBtn = document.getElementById('resetConfigBtn');
        if (resetConfigBtn && !resetConfigBtn.hasAttribute('data-initialized')) {
            resetConfigBtn.setAttribute('data-initialized', 'true');
            resetConfigBtn.addEventListener('click', () => this.resetConfiguration());
            console.log('[BotsManager] ✅ Кнопка "Сбросить к умолчаниям" инициализирована');
        }
        
        // Обработчик кнопки тестирования конфигурации
        const testConfigBtn = document.getElementById('testConfigBtn');
        if (testConfigBtn && !testConfigBtn.hasAttribute('data-initialized')) {
            testConfigBtn.setAttribute('data-initialized', 'true');
            testConfigBtn.addEventListener('click', () => this.testConfiguration());
            console.log('[BotsManager] ✅ Кнопка "Тестировать конфигурацию" инициализирована');
        }
        
        // ✅ ОБРАБОТЧИКИ ДЛЯ КНОПОК СОХРАНЕНИЯ ОТДЕЛЬНЫХ БЛОКОВ
        
        // Основные настройки
        const saveBasicBtn = document.querySelector('.config-section-save-btn[data-section="basic"]');
        if (saveBasicBtn && !saveBasicBtn.hasAttribute('data-initialized')) {
            saveBasicBtn.setAttribute('data-initialized', 'true');
            saveBasicBtn.addEventListener('click', () => this.saveBasicSettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить основные настройки" инициализирована');
        }
        
        // Системные настройки
        const saveSystemBtn = document.querySelector('.config-section-save-btn[data-section="system"]');
        if (saveSystemBtn && !saveSystemBtn.hasAttribute('data-initialized')) {
            saveSystemBtn.setAttribute('data-initialized', 'true');
            saveSystemBtn.addEventListener('click', () => this.saveSystemSettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить системные настройки" инициализирована');
        }
        
        // Торговые параметры
        const saveTradingBtn = document.querySelector('.config-section-save-btn[data-section="trading"]');
        if (saveTradingBtn && !saveTradingBtn.hasAttribute('data-initialized')) {
            saveTradingBtn.setAttribute('data-initialized', 'true');
            saveTradingBtn.addEventListener('click', () => this.saveTradingParameters());
            console.log('[BotsManager] ✅ Кнопка "Сохранить торговые параметры" инициализирована');
        }
        
        // RSI выходы
        const saveRsiExitsBtn = document.querySelector('.config-section-save-btn[data-section="rsi-exits"]');
        if (saveRsiExitsBtn && !saveRsiExitsBtn.hasAttribute('data-initialized')) {
            saveRsiExitsBtn.setAttribute('data-initialized', 'true');
            saveRsiExitsBtn.addEventListener('click', () => this.saveRsiExits());
            console.log('[BotsManager] ✅ Кнопка "Сохранить RSI выходы" инициализирована');
        }
        
        // RSI временной фильтр
        const saveRsiTimeBtn = document.querySelector('.config-section-save-btn[data-section="rsi-time-filter"]');
        if (saveRsiTimeBtn && !saveRsiTimeBtn.hasAttribute('data-initialized')) {
            saveRsiTimeBtn.setAttribute('data-initialized', 'true');
            saveRsiTimeBtn.addEventListener('click', () => this.saveRsiTimeFilter());
            console.log('[BotsManager] ✅ Кнопка "Сохранить RSI временной фильтр" инициализирована');
        }
        
        // ExitScam фильтр
        const saveExitScamBtn = document.querySelector('.config-section-save-btn[data-section="exit-scam"]');
        if (saveExitScamBtn && !saveExitScamBtn.hasAttribute('data-initialized')) {
            saveExitScamBtn.setAttribute('data-initialized', 'true');
            saveExitScamBtn.addEventListener('click', () => this.saveExitScamFilter());
            console.log('[BotsManager] ✅ Кнопка "Сохранить ExitScam фильтр" инициализирована');
        }
        
        // Enhanced RSI
        const saveEnhancedRsiBtn = document.querySelector('.config-section-save-btn[data-section="enhanced-rsi"]');
        if (saveEnhancedRsiBtn && !saveEnhancedRsiBtn.hasAttribute('data-initialized')) {
            saveEnhancedRsiBtn.setAttribute('data-initialized', 'true');
            saveEnhancedRsiBtn.addEventListener('click', () => this.saveEnhancedRsi());
            console.log('[BotsManager] ✅ Кнопка "Сохранить Enhanced RSI" инициализирована');
        }
        
        // Торговые настройки
        const saveTradingSettingsBtn = document.querySelector('.config-section-save-btn[data-section="trading-settings"]');
        if (saveTradingSettingsBtn && !saveTradingSettingsBtn.hasAttribute('data-initialized')) {
            saveTradingSettingsBtn.setAttribute('data-initialized', 'true');
            saveTradingSettingsBtn.addEventListener('click', () => this.saveTradingSettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить торговые настройки" инициализирована');
        }
        
        // Защитные механизмы
        const saveProtectiveBtn = document.querySelector('.config-section-save-btn[data-section="protective"]');
        if (saveProtectiveBtn && !saveProtectiveBtn.hasAttribute('data-initialized')) {
            saveProtectiveBtn.setAttribute('data-initialized', 'true');
            saveProtectiveBtn.addEventListener('click', () => this.saveProtectiveMechanisms());
            console.log('[BotsManager] ✅ Кнопка "Сохранить защитные механизмы" инициализирована');
        }
        
        // Настройки зрелости
        const saveMaturityBtn = document.querySelector('.config-section-save-btn[data-section="maturity"]');
        if (saveMaturityBtn && !saveMaturityBtn.hasAttribute('data-initialized')) {
            saveMaturityBtn.setAttribute('data-initialized', 'true');
            saveMaturityBtn.addEventListener('click', () => this.saveMaturitySettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить настройки зрелости" инициализирована');
        }
        
        // EMA параметры
        const saveEmaBtn = document.querySelector('.config-section-save-btn[data-section="ema"]');
        if (saveEmaBtn && !saveEmaBtn.hasAttribute('data-initialized')) {
            saveEmaBtn.setAttribute('data-initialized', 'true');
            saveEmaBtn.addEventListener('click', () => this.saveEmaParameters());
            console.log('[BotsManager] ✅ Кнопка "Сохранить EMA параметры" инициализирована');
        }
        
        // Параметры тренда
        const saveTrendBtn = document.querySelector('.config-section-save-btn[data-section="trend"]');
        if (saveTrendBtn && !saveTrendBtn.hasAttribute('data-initialized')) {
            saveTrendBtn.setAttribute('data-initialized', 'true');
            saveTrendBtn.addEventListener('click', () => this.saveTrendParameters());
            console.log('[BotsManager] ✅ Кнопка "Сохранить параметры тренда" инициализирована');
        }
        
        // Hot Reload кнопка
        const reloadModulesBtn = document.getElementById('reloadModulesBtn');
        if (reloadModulesBtn && !reloadModulesBtn.hasAttribute('data-initialized')) {
            reloadModulesBtn.setAttribute('data-initialized', 'true');
            reloadModulesBtn.addEventListener('click', () => this.reloadModules());
            console.log('[BotsManager] ✅ Кнопка "Hot Reload" инициализирована');
        }
        
        console.log('[BotsManager] ✅ Все кнопки конфигурации инициализированы');
    }
    
    async reloadModules() {
        console.log('[BotsManager] 🔄 Перезагрузка модулей...');
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/system/reload-modules`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(`✅ ${data.message}. Модули перезагружены без перезапуска сервера!`, 'success');
                console.log('[BotsManager] ✅ Перезагружено модулей:', data.reloaded);
                if (data.failed && data.failed.length > 0) {
                    console.error('[BotsManager] ❌ Ошибки при перезагрузке:', data.failed);
                }
                
                // Обновляем данные после перезагрузки
                await this.loadConfiguration();
                await this.loadCoinsRsiData();
            } else {
                this.showNotification(`❌ Ошибка перезагрузки: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка Hot Reload:', error);
            this.showNotification('❌ Ошибка перезагрузки модулей', 'error');
        }
    }
    
    async startAllBots() {
        if (!this.activeBots || this.activeBots.length === 0) {
            this.showNotification('⚠️ Нет ботов для запуска', 'warning');
            return;
        }

        const stoppedBots = this.activeBots.filter(bot => 
            bot.status === 'paused' || bot.status === 'idle' || bot.status === 'stopped'
        );
        
        if (stoppedBots.length === 0) {
            this.showNotification('ℹ️ Все боты уже запущены', 'info');
            return;
        }
        
        console.log(`[BotsManager] 🚀 Запуск ${stoppedBots.length} ботов...`);
        this.showConfigNotification('🚀 Массовый запуск ботов', `Запускаем ${stoppedBots.length} ботов...`);
        
        let successful = 0;
        let failed = 0;
        
        for (const bot of stoppedBots) {
            try {
                const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/start`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: bot.symbol })
                });
                
                const result = await response.json();
                if (result.success) {
                    successful++;
                } else {
                    failed++;
                }
            } catch (error) {
                failed++;
            }
        }
        
        await this.loadActiveBotsData();
        
        if (failed === 0) {
            this.showConfigNotification('✅ Все боты запущены', `Успешно запущено ${successful} ботов`);
        } else {
            this.showConfigNotification('⚠️ Запуск завершен с ошибками', 
                `Успешно: ${successful}, Ошибок: ${failed}`, 'error');
        }
    }
    
    async stopAllBots() {
        if (!this.activeBots || this.activeBots.length === 0) {
            this.showNotification('⚠️ Нет ботов для остановки', 'warning');
            return;
        }
        
        const runningBots = this.activeBots.filter(bot => 
            bot.status === 'running' || bot.status === 'idle' || 
            bot.status === 'in_position_long' || bot.status === 'in_position_short'
        );
        
        if (runningBots.length === 0) {
            this.showNotification('ℹ️ Все боты уже остановлены', 'info');
            return;
        }
        
        console.log(`[BotsManager] ⏹️ Остановка ${runningBots.length} ботов...`);
        this.showConfigNotification('⏹️ Массовая остановка ботов', `Останавливаем ${runningBots.length} ботов...`);
        
        // Немедленно обновляем UI для всех ботов
        runningBots.forEach(bot => {
            this.updateBotStatusInUI(bot.symbol, 'stopping');
        });
        
        let successful = 0;
        let failed = 0;
        
        for (const bot of runningBots) {
            try {
                const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/stop`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: bot.symbol })
                });
                
                const result = await response.json();
                if (result.success) {
                    successful++;
                } else {
                    failed++;
                }
            } catch (error) {
                console.error(`[BotsManager] ❌ Ошибка остановки бота ${bot.symbol}:`, error);
                failed++;
            }
        }
        
        await this.loadActiveBotsData();
        
        if (failed === 0) {
            this.showConfigNotification('✅ Все боты остановлены', `Успешно остановлено ${successful} ботов`);
                } else {
            this.showConfigNotification('⚠️ Остановка завершена с ошибками', 
                `Успешно: ${successful}, Ошибок: ${failed}`, 'error');
        }
    }
    
    async deleteAllBots() {
        if (!this.activeBots || this.activeBots.length === 0) {
            this.showNotification('⚠️ Нет ботов для удаления', 'warning');
            return;
        }
        
        const confirmMessage = `🗑️ Удалить всех ${this.activeBots.length} ботов?\n\nЭто действие нельзя отменить!`;
        
        if (!confirm(confirmMessage)) {
            return;
        }
        
        console.log(`[BotsManager] 🗑️ Удаление ${this.activeBots.length} ботов...`);
        this.showConfigNotification('🗑️ Массовое удаление ботов', `Удаляем ${this.activeBots.length} ботов...`);
        
        // Немедленно обновляем UI для всех ботов
        this.activeBots.forEach(bot => {
            this.updateBotStatusInUI(bot.symbol, 'deleting');
        });
        
        let successful = 0;
        let failed = 0;
        
        for (const bot of this.activeBots) {
            try {
                const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/delete`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbol: bot.symbol })
                });
                
                const result = await response.json();
                if (result.success) {
                    successful++;
                } else {
                    failed++;
                }
            } catch (error) {
                failed++;
            }
        }
        
        await this.loadActiveBotsData();
        this.updateCoinsListWithBotStatus();
        
        if (failed === 0) {
            this.showConfigNotification('✅ Все боты удалены', `Успешно удалено ${successful} ботов`);
        } else {
            this.showConfigNotification('⚠️ Удаление завершено с ошибками', 
                `Успешно: ${successful}, Ошибок: ${failed}`, 'error');
        }
    }
    
    // ==========================================
    // УЛУЧШЕННЫЕ УВЕДОМЛЕНИЯ О СОХРАНЕНИИ
    // ==========================================
    
    showConfigNotification(title, message, type = 'success', changes = null) {
        // Удаляем предыдущее уведомление если есть
        const existingNotification = document.querySelector('.config-save-notification');
        if (existingNotification) {
            existingNotification.remove();
        }
        
        // Создаем новое уведомление
        const notification = document.createElement('div');
        notification.className = `config-save-notification ${type === 'error' ? 'error' : ''}`;
        
        let changesHtml = '';
        if (changes && changes.length > 0) {
            changesHtml = `
                <div class="config-changes-list">
                    <strong>${this.translate('changes_label')}</strong>
                    <ul>
                        ${changes.map(change => `<li>${change}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        notification.innerHTML = `
            <div class="config-notification-header">
                <span class="config-notification-title">${title}</span>
                <button class="config-notification-close" type="button">&times;</button>
            </div>
            <div class="config-notification-body">
                ${message}
                ${changesHtml}
            </div>
        `;
        
        // Добавляем в DOM
        document.body.appendChild(notification);
        
        // Показываем с анимацией
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        // Обработчик закрытия
        const closeBtn = notification.querySelector('.config-notification-close');
        const closeNotification = () => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 400);
        };
        
        closeBtn.addEventListener('click', closeNotification);
        
        // Автоматическое закрытие через 5 секунд
        setTimeout(closeNotification, 5000);
        
        console.log(`[BotsManager] 📢 Уведомление: ${title} - ${message}`);
    }
    
    // ==========================================
    // ДЕТЕКЦИЯ ИЗМЕНЕНИЙ КОНФИГУРАЦИИ
    // ==========================================
    
    detectConfigChanges(oldAutoBot, oldSystem, newAutoBot, newSystem) {
        const changes = [];
        
        // Словарь с человеко-читаемыми названиями настроек
        const configLabels = {
            // Auto Bot настройки
            'enabled': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Auto Bot enabled' : 'Auto Bot включен',
            'max_concurrent': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Max concurrent bots' : 'Макс. одновременных ботов',
            'risk_cap_percent': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Risk cap (% of deposit)' : 'Рискованность (% от депозита)',
            'scope': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Action scope' : 'Область действия',
            'rsi_long_threshold': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI for LONG positions' : 'RSI для LONG позиций',
            'rsi_short_threshold': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI for SHORT positions' : 'RSI для SHORT позиций',
            'rsi_exit_long': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI exit from LONG' : 'RSI выход из LONG',
            'rsi_exit_short': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI exit from SHORT' : 'RSI выход из SHORT',
            'default_position_size': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Default position size' : 'Размер позиции по умолчанию',
            'check_interval': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Check interval (min)' : 'Интервал проверки (мин)',
            'max_risk_per_trade': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Max risk per trade (%)' : 'Макс. риск на сделку (%)',
            'max_loss_percent': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Max loss (%)' : 'Макс. убыток (%)',
            'trailing_stop_activation': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Trailing stop activation (%)' : 'Активация трейлинг-стопа (%)',
            'trailing_stop_distance': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Trailing stop distance (%)' : 'Расстояние трейлинг-стопа (%)',
            'max_position_hours': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Max time in position (minutes)' : 'Макс. время в позиции (минуты)',
            'break_even_protection': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Break-even protection' : 'Защита безубыточности',
            'break_even_trigger': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Break-even trigger (%)' : 'Триггер безубыточности (%)',
            'avoid_down_trend': window.languageUtils?.getCurrentLanguage() === 'en' ? '🔻 Avoid downtrend (LONG)' : '🔻 Избегать нисходящий тренд (LONG)',
            'avoid_up_trend': window.languageUtils?.getCurrentLanguage() === 'en' ? '📈 Avoid uptrend (SHORT)' : '📈 Избегать восходящий тренд (SHORT)',
            
            // Системные настройки
            'rsi_update_interval': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI update interval' : 'Интервал обновления RSI',
            'auto_save_interval': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Auto-save interval' : 'Интервал автосохранения',
            'debug_mode': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Debug mode' : 'Режим отладки',
            'auto_refresh_ui': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Auto-refresh UI' : 'Автообновление UI'
        };
        
        // Функция для форматирования значений
        const formatValue = (key, value) => {
            const isEnglish = window.languageUtils?.getCurrentLanguage() === 'en';
            
            if (typeof value === 'boolean') {
                return isEnglish ? 
                    (value ? 'enabled' : 'disabled') : 
                    (value ? 'включено' : 'выключено');
            }
            if (key === 'scope') {
                if (isEnglish) {
                    return value === 'all' ? 'All coins' : 
                           value === 'whitelist' ? 'Whitelist' : 
                           value === 'blacklist' ? 'Blacklist' : value;
                } else {
                    return value === 'all' ? 'Все монеты' : 
                           value === 'whitelist' ? 'Белый список' : 
                           value === 'blacklist' ? 'Черный список' : value;
                }
            }
            if (key === 'rsi_update_interval') {
                const minutes = Math.round(value / 60);
                return isEnglish ? 
                    `${minutes} min (${value} sec)` : 
                    `${minutes} мин (${value} сек)`;
            }
            if (key === 'auto_save_interval') {
                return isEnglish ? `${value} sec` : `${value} сек`;
            }
            return value;
        };
        
        // Сравниваем Auto Bot настройки
        if (oldAutoBot && newAutoBot) {
            Object.keys(newAutoBot).forEach(key => {
                const oldValue = oldAutoBot[key];
                const newValue = newAutoBot[key];
                
                if (oldValue !== newValue && configLabels[key]) {
                    changes.push(
                        `${configLabels[key]}: ${formatValue(key, oldValue)} → ${formatValue(key, newValue)}`
                    );
                }
            });
        }
        
        // Сравниваем системные настройки
        if (oldSystem && newSystem) {
            Object.keys(newSystem).forEach(key => {
                const oldValue = oldSystem[key];
                const newValue = newSystem[key];
                
                if (oldValue !== newValue && configLabels[key]) {
                    changes.push(
                        `${configLabels[key]}: ${formatValue(key, oldValue)} → ${formatValue(key, newValue)}`
                    );
                }
            });
        }
        
        console.log('[BotsManager] 🔍 Обнаружено изменений:', changes.length);
        changes.forEach(change => console.log('[BotsManager] 📝', change));
        
        return changes;
    }
    
    getBotPositionInfo(bot) {
        // ОТЛАДКА: Проверяем данные бота
        console.log(`[DEBUG] getBotPositionInfo для ${bot.symbol}:`, {
            position_side: bot.position_side,
            entry_price: bot.entry_price,
            status: bot.status,
            current_price: bot.current_price,
            stop_loss_price: bot.stop_loss_price,
            exchange_position: bot.exchange_position
        });
        
        // Проверяем, есть ли активная позиция
        if (!bot.position_side || !bot.entry_price) {
            // Если нет активной позиции, показываем информацию о статусе бота
            let statusText = '';
            let statusColor = '#888';
            let statusIcon = '📍';
            
            if (bot.status === 'in_position_long') {
                statusText = 'LONG (закрыта)';
                statusColor = '#4caf50';
                statusIcon = '📈';
            } else if (bot.status === 'in_position_short') {
                statusText = 'SHORT (закрыта)';
                statusColor = '#f44336';
                statusIcon = '📉';
            } else if (bot.status === 'running') {
                statusText = 'Ожидание сигнала';
                statusColor = '#2196f3';
                statusIcon = '🔄';
            } else {
                statusText = 'Нет позиции';
                statusColor = '#888';
                statusIcon = '📍';
            }
            
            return `<div style="display: flex; justify-content: space-between; margin-bottom: 4px;"><span>${statusIcon} Позиция:</span><span style="color: ${statusColor};">${statusText}</span></div>`;
        }
        
        const sideColor = bot.position_side === 'LONG' ? '#4caf50' : '#f44336';
        const sideIcon = bot.position_side === 'LONG' ? '📈' : '📉';
        
        let positionHtml = `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                <span style="color: #888;">${sideIcon} Позиция</span>
                <span style="color: ${sideColor}; font-weight: 600;">${bot.position_side}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                <span style="color: #888;">💵 Вход</span>
                <span style="color: #fff; font-weight: 600;">$${bot.entry_price.toFixed(6)}</span>
            </div>
        `;
        
        // ✅ ИСПРАВЛЕНО: Используем current_price напрямую из bot (обновляется каждую секунду)
        if (bot.current_price || bot.mark_price) {
            const currentPrice = bot.current_price || bot.mark_price;
            const entryPrice = bot.entry_price || 0;
            const priceChange = entryPrice > 0 ? ((currentPrice - entryPrice) / entryPrice) * 100 : 0;
            const priceChangeColor = priceChange >= 0 ? '#4caf50' : '#f44336';
            const priceChangeIcon = priceChange >= 0 ? '↗️' : '↘️';
            
            positionHtml += `
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                    <span style="color: #888;">📊 Текущая</span>
                    <span style="color: ${priceChangeColor}; font-weight: 600;">$${currentPrice.toFixed(6)} ${priceChangeIcon}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                    <span style="color: #888;">📈 Изменение</span>
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
            const currentRsi = bot.rsi_data?.rsi6h || 50;
            
            if (bot.position_side === 'LONG' && currentRsi < rsiExitLong) {
                const takeProfitPercent = (rsiExitLong - currentRsi) * 0.5;
                takeProfit = bot.entry_price * (1 + takeProfitPercent / 100);
            } else if (bot.position_side === 'SHORT' && currentRsi > rsiExitShort) {
                const takeProfitPercent = (currentRsi - rsiExitShort) * 0.5;
                takeProfit = bot.entry_price * (1 - takeProfitPercent / 100);
            }
        }
        
        positionHtml += `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                <span style="color: #888;">🛡️ Стоп-лосс</span>
                <span style="color: ${stopLoss ? '#ff9800' : '#666'}; font-weight: 600;">${stopLoss ? `$${parseFloat(stopLoss).toFixed(6)}` : 'Не установлен'}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                <span style="color: #888;">🎯 Тейк-профит</span>
                <span style="color: ${takeProfit ? '#4caf50' : '#666'}; font-weight: 600;">${takeProfit ? `$${parseFloat(takeProfit).toFixed(6)}` : 'Не установлен'}</span>
            </div>
        `;
        
        // Добавляем RSI данные если есть
        if (bot.rsi_data) {
            const rsi = bot.rsi_data.rsi6h;
            const trend = bot.rsi_data.trend6h;
            
            if (rsi) {
                let rsiColor = '#888';
                if (rsi > 70) rsiColor = '#f44336'; // Перекупленность
                else if (rsi < 30) rsiColor = '#4caf50'; // Перепроданность
                
                positionHtml += `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                        <span style="color: #888;">📊 RSI</span>
                        <span style="color: ${rsiColor}; font-weight: 600;">${rsi.toFixed(1)}</span>
                    </div>
                `;
            }
            
            if (trend) {
                let trendColor = '#888';
                let trendIcon = '➡️';
                if (trend === 'UP') { trendColor = '#4caf50'; trendIcon = '📈'; }
                else if (trend === 'DOWN') { trendColor = '#f44336'; trendIcon = '📉'; }
                else if (trend === 'NEUTRAL') { trendColor = '#ff9800'; trendIcon = '➡️'; }
                
                positionHtml += `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                        <span style="color: #888;">${trendIcon} Тренд</span>
                        <span style="color: ${trendColor}; font-weight: 600;">${trend}</span>
                    </div>
                `;
            }
        }
        
        return positionHtml;
    }
    
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
                <span>⏱️ Время:</span>
                <span style="color: #888; font-weight: 500;">${timeText}</span>
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
            let updateColor = '#4caf50'; // зеленый - свежие данные
            if (updateMinutes > 1) {
                updateColor = '#ff9800'; // оранжевый - данные старше минуты
            }
            if (updateMinutes > 5) {
                updateColor = '#f44336'; // красный - данные старше 5 минут
            }
            
            timeInfoHtml += `
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>🔄 Обновлено:</span>
                    <span style="color: ${updateColor}; font-weight: 500;">${updateTimeText}</span>
                </div>
            `;
        }
        
        return timeInfoHtml;
    }
    
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
    }
    
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
                trend: bot.trend6h || 'NEUTRAL',
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
                trend: bot.trend6h || 'NEUTRAL',
                workTime: bot.work_time || '0м',
                lastUpdate: bot.last_update || 'Неизвестно'
            });
        }
        
        return trades;
    }
    
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
            <div class="trade-item" style="border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 8px 0; background: #2a2a2a; transition: all 0.3s ease;" onmouseover="this.style.backgroundColor='#333'" onmouseout="this.style.backgroundColor='#2a2a2a'">
                <div class="trade-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #444;">
                    <div class="trade-side ${sideClass}" style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 16px;">${sideIcon}</span>
                        <span style="color: ${trade.side === 'LONG' ? '#4caf50' : '#f44336'}; font-weight: bold;">${trade.side}</span>
                    </div>
                    <div class="trade-status ${trade.status}" style="background: ${trade.status === 'active' ? '#4caf50' : '#ff5722'}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px;">
                        ${trade.status === 'active' ? 'Активна' : 'Закрыта'}
                    </div>
                </div>
                
                <div class="trade-details" style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; color: #ccc;">
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: #888;">💵 Вход:</span>
                        <span class="trade-detail-value" style="color: #fff; font-weight: 600;">$${trade.entryPrice.toFixed(6)}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: #888;">📊 Текущая:</span>
                        <span class="trade-detail-value" style="color: #fff; font-weight: 600;">$${trade.currentPrice.toFixed(6)}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: #888;">📈 Изменение:</span>
                        <span class="trade-detail-value ${priceChangeClass}" style="color: ${priceChange >= 0 ? '#4caf50' : '#f44336'}; font-weight: 600;">${priceChange.toFixed(2)}%</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: #888;">💰 Объем:</span>
                        <span class="trade-detail-value" style="color: #fff; font-weight: 600;">${trade.volume.toFixed(2)} ${trade.volumeMode.toUpperCase()}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: #888;">🛡️ Стоп-лосс:</span>
                        <span class="trade-detail-value" style="color: #ff9800; font-weight: 600;">$${parseFloat(trade.stopLossPrice).toFixed(6)} (${trade.stopLossPercent}%)</span>
                    </div>
                    
                    ${trade.takeProfitPrice ? `
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: #888;">🎯 Тейк-профит:</span>
                        <span class="trade-detail-value" style="color: #4caf50; font-weight: 600;">$${parseFloat(trade.takeProfitPrice).toFixed(6)}</span>
                    </div>
                    ` : ''}
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: #888;">📊 RSI:</span>
                        <span class="trade-detail-value" style="color: #fff; font-weight: 600;">${trade.rsi ? trade.rsi.toFixed(1) : 'N/A'}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: #888;">➡️ Тренд:</span>
                        <span class="trade-detail-value" style="color: ${trade.trend === 'UP' ? '#4caf50' : trade.trend === 'DOWN' ? '#f44336' : '#ff9800'}; font-weight: 600;">${trade.trend || 'NEUTRAL'}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: #888;">⏱️ Время:</span>
                        <span class="trade-detail-value" style="color: #fff; font-weight: 600;">${trade.workTime || '0м'}</span>
                    </div>
                    
                    <div class="trade-detail-item" style="display: flex; justify-content: space-between; align-items: center; padding: 4px 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                        <span class="trade-detail-label" style="color: #888;">🔄 Обновлено:</span>
                        <span class="trade-detail-value" style="color: #fff; font-weight: 600;">${trade.lastUpdate || 'Неизвестно'}</span>
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
     */
    initializeManualPositionsControls() {
        console.log('[BotsManager] 🔄 Инициализация кнопки обновления ручных позиций...');
        
        // Кнопка обновления ручных позиций
        const refreshBtn = document.getElementById('refreshManualPositionsBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', async () => {
                console.log('[BotsManager] 🔄 Обновление ручных позиций...');
                
                try {
                    const response = await fetch(`http://localhost:5001/api/bots/manual-positions/refresh`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        console.log('[BotsManager] ✅ Ручные позиции обновлены:', result);
                        
                        // Обновляем данные и интерфейс
                        await this.loadCoinsRsiData();
                        
                        // Показываем уведомление
                        if (window.showToast) {
                            window.showToast(`${window.languageUtils.translate('updated')} ${result.count} ${window.languageUtils.translate('manual_positions')}`, 'success');
                        }
                    } else {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                } catch (error) {
                    console.error('[BotsManager] ❌ Ошибка обновления ручных позиций:', error);
                    if (window.showToast) {
                        window.showToast(`Ошибка обновления: ${error.message}`, 'error');
                    }
                }
            });
        }
    }
    
    /**
     * Инициализирует кнопки загрузки RSI данных
     */
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
     */
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
     */
    showNotification(message, type = 'info') {
        // Простое уведомление в консоли, можно заменить на toast
        console.log(`[${type.toUpperCase()}] ${message}`);
    }

    // ==================== ИСТОРИЯ БОТОВ ====================

    /**
     * Инициализирует вкладку истории ботов
     */
    initializeHistoryTab() {
        console.log('[BotsManager] 📊 Инициализация вкладки истории ботов...');
        
        // Инициализируем фильтры
        this.initializeHistoryFilters();
        
        // Инициализируем подвкладки истории
        this.initializeHistorySubTabs();
        
        // Загружаем данные
        this.loadHistoryData();
        
        // Инициализируем кнопки действий
        this.initializeHistoryActionButtons();
    }

    /**
     * Инициализирует фильтры истории
     */
    initializeHistoryFilters() {
        // Фильтр по боту
        const botFilter = document.getElementById('historyBotFilter');
        if (botFilter) {
            botFilter.addEventListener('change', () => this.loadHistoryData());
        }

        // Фильтр по типу действия
        const actionFilter = document.getElementById('historyActionFilter');
        if (actionFilter) {
            actionFilter.addEventListener('change', () => this.loadHistoryData());
        }

        // Фильтр по периоду
        const dateFilter = document.getElementById('historyDateFilter');
        if (dateFilter) {
            dateFilter.addEventListener('change', () => this.loadHistoryData());
        }

        // Кнопки фильтров
        const applyBtn = document.querySelector('.history-filters .btn-primary');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this.loadHistoryData());
        }

        const clearBtn = document.querySelector('.history-filters .btn-secondary');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearHistoryFilters());
        }

        const exportBtn = document.querySelector('.history-filters .btn-info');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportHistoryData());
        }
    }

    /**
     * Инициализирует подвкладки истории
     */
    initializeHistorySubTabs() {
        const tabButtons = document.querySelectorAll('.history-tab-btn');
        const tabContents = document.querySelectorAll('.history-tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.dataset.historyTab;
                
                // Убираем активный класс со всех кнопок и контента
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Добавляем активный класс к выбранной кнопке и контенту
                button.classList.add('active');
                const targetContent = document.getElementById(`${tabName}History`);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
                
                // Загружаем данные для выбранной вкладки
                this.loadHistoryData(tabName);
            });
        });
    }

    /**
     * Инициализирует кнопки действий истории
     */
    initializeHistoryActionButtons() {
        // Кнопка обновления
        const refreshBtn = document.querySelector('.history-actions .btn-primary');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadHistoryData());
        }

        // Кнопка создания демо-данных
        const demoBtn = document.querySelector('.history-actions .btn-success');
        if (demoBtn) {
            demoBtn.addEventListener('click', () => this.createDemoHistoryData());
        }

        // Кнопка очистки истории
        const clearBtn = document.querySelector('.history-actions .btn-warning');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearAllHistory());
        }
    }

    /**
     * Загружает данные истории
     */
    async loadHistoryData(tabName = 'actions') {
        try {
            console.log(`[BotsManager] 📊 Загрузка данных истории: ${tabName}`);
            
            // Получаем параметры фильтров
            const filters = this.getHistoryFilters();
            
            // Загружаем данные в зависимости от вкладки
            switch (tabName) {
                case 'actions':
                    await this.loadBotActions(filters);
                    break;
                case 'trades':
                    await this.loadBotTrades(filters);
                    break;
                case 'signals':
                    await this.loadBotSignals(filters);
                    break;
            }
            
            // Загружаем статистику
            await this.loadHistoryStatistics(filters.symbol);
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки данных истории:', error);
            this.showNotification(`Ошибка загрузки истории: ${error.message}`, 'error');
        }
    }

    /**
     * Получает параметры фильтров
     */
    getHistoryFilters() {
        const botFilter = document.getElementById('historyBotFilter');
        const actionFilter = document.getElementById('historyActionFilter');
        const dateFilter = document.getElementById('historyDateFilter');
        
        return {
            symbol: botFilter ? botFilter.value : null,
            action_type: actionFilter ? actionFilter.value : null,
            trade_type: actionFilter ? actionFilter.value : null,
            period: dateFilter ? dateFilter.value : null,
            limit: 100
        };
    }

    /**
     * Загружает действия ботов
     */
    async loadBotActions(filters) {
        try {
            const params = new URLSearchParams();
            if (filters.symbol && filters.symbol !== 'all') params.append('symbol', filters.symbol);
            if (filters.action_type && filters.action_type !== 'all') params.append('action_type', filters.action_type);
            params.append('limit', filters.limit);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/history?${params}`);
            const data = await response.json();
            
            if (data.success) {
                this.displayBotActions(data.history);
            } else {
                throw new Error(data.error || 'Ошибка загрузки действий');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки действий ботов:', error);
            this.displayBotActions([]);
        }
    }

    /**
     * Загружает сделки ботов
     */
    async loadBotTrades(filters) {
        try {
            const params = new URLSearchParams();
            if (filters.symbol && filters.symbol !== 'all') params.append('symbol', filters.symbol);
            if (filters.trade_type && filters.trade_type !== 'all') params.append('trade_type', filters.trade_type);
            params.append('limit', filters.limit);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/trades?${params}`);
            const data = await response.json();
            
            if (data.success) {
                this.displayBotTrades(data.trades);
            } else {
                throw new Error(data.error || 'Ошибка загрузки сделок');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки сделок ботов:', error);
            this.displayBotTrades([]);
        }
    }
    
    /**
     * Загружает сигналы ботов
     */
    async loadBotSignals(filters) {
        try {
            const params = new URLSearchParams();
            if (filters.symbol && filters.symbol !== 'all') params.append('symbol', filters.symbol);
            params.append('action_type', 'SIGNAL');
            params.append('limit', filters.limit);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/history?${params}`);
            const data = await response.json();
            
            if (data.success) {
                this.displayBotSignals(data.history);
            } else {
                throw new Error(data.error || 'Ошибка загрузки сигналов');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки сигналов ботов:', error);
            this.displayBotSignals([]);
        }
    }

    /**
     * Загружает статистику истории
     */
    async loadHistoryStatistics(symbol = null) {
        try {
            const params = new URLSearchParams();
            if (symbol && symbol !== 'all') params.append('symbol', symbol);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/statistics?${params}`);
            const data = await response.json();
            
            if (data.success) {
                this.displayHistoryStatistics(data.statistics);
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки статистики:', error);
        }
    }

    /**
     * Отображает действия ботов
     */
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
     */
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
        
        const html = trades.map(trade => `
            <div class="history-item trade-item ${trade.status === 'CLOSED' ? 'closed' : 'open'}">
                <div class="history-item-header">
                    <span class="history-trade-direction ${trade.direction.toLowerCase()}">${trade.direction}</span>
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
                    </div>
                    <div class="trade-status">Статус: ${trade.status === 'OPEN' ? 'Открыта' : 'Закрыта'}</div>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = html;
    }

    /**
     * Отображает сигналы ботов
     */
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
     */
    displayHistoryStatistics(stats) {
        // Обновляем карточки статистики
        const totalActionsEl = document.querySelector('.history-stats .stat-card:nth-child(1) .stat-value');
        const totalTradesEl = document.querySelector('.history-stats .stat-card:nth-child(2) .stat-value');
        const totalPnlEl = document.querySelector('.history-stats .stat-card:nth-child(3) .stat-value');
        const successRateEl = document.querySelector('.history-stats .stat-card:nth-child(4) .stat-value');
        
        if (totalActionsEl) totalActionsEl.textContent = stats.total_trades || 0;
        if (totalTradesEl) totalTradesEl.textContent = stats.total_trades || 0;
        if (totalPnlEl) totalPnlEl.textContent = `$${stats.total_pnl?.toFixed(2) || '0.00'}`;
        if (successRateEl) successRateEl.textContent = `${stats.win_rate?.toFixed(1) || '0'}%`;
    }

    /**
     * Очищает фильтры истории
     */
    clearHistoryFilters() {
        const botFilter = document.getElementById('historyBotFilter');
        const actionFilter = document.getElementById('historyActionFilter');
        const dateFilter = document.getElementById('historyDateFilter');
        
        if (botFilter) botFilter.value = 'all';
        if (actionFilter) actionFilter.value = 'all';
        if (dateFilter) dateFilter.value = 'today';
        
        this.loadHistoryData();
    }

    /**
     * Экспортирует данные истории
     */
    exportHistoryData() {
        console.log('[BotsManager] 📤 Экспорт данных истории (функция в разработке)');
        this.showNotification('Функция экспорта в разработке', 'info');
    }

    /**
     * Создает демо-данные истории
     */
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
     */
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
     */
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
     */
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
    }
    
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
    }
    
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
}

// Экспортируем класс глобально сразу после определения
window.BotsManager = BotsManager;

// BotsManager инициализируется в app.js, не здесь
// Version: 2025-10-21 03:47:29
