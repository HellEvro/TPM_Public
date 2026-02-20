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
        this.activeBotsFilter = 'all'; // Фильтр вкладки "Боты в работе": all, long, short, profitable, loss
        this._lastActiveBotsFilter = 'all'; // Для определения необходимости перерисовки при смене фильтра
        
        // RSI пороговые значения из конфигурации
        this.rsiLongThreshold = 29;
        this.rsiShortThreshold = 71;
        
        // Флаг для предотвращения множественных обновлений подписей
        this.trendLabelsUpdated = false;
        
        // Версия данных для отслеживания изменений
        this.lastDataVersion = 0;
        
        // Кэш последнего отображённого состояния (чтобы не перерисовывать DOM без изменений — убирает «дискотеку»)
        this._lastAccountDisplay = null;
        this._lastServiceStatus = { status: null, message: null };
        
        // Единый интервал обновления UI и мониторинга ботов (мин. 5 сек — иначе интерфейс мигает)
        this.refreshInterval = 5000;
        this.monitoringTimer = null;
        
        // Debounce для поиска
        this.searchDebounceTimer = null;
        
        // Список делистинговых монет
        this.delistedCoins = [];
        
        // Кэш конфигурации Auto Bot для быстрого доступа
        this.cachedAutoBotConfig = null;
        // Исходные значения всех параметров при загрузке страницы (для отслеживания изменений)
        this.originalConfig = null;
        
        // Автосохранение конфигурации - таймер для debounce
        this.autoSaveTimer = null;
        this.autoSaveDelay = 2000; // 2 секунды
        this.toggleAutoSaveTimer = null;
        this.toggleAutoSaveDelay = 400;
        // Флаг для предотвращения автосохранения при программном изменении полей
        this.isProgrammaticChange = false;
        this.aiConfigDirty = false;
        
        // URL сервиса ботов — всегда порт 5001 (сервис bots.py)
        const hostname = window.location.hostname || '127.0.0.1';
        const protocol = window.location.protocol || 'http:';
        this.BOTS_SERVICE_URL = `${protocol}//${hostname}:5001`;
        this.apiUrl = this.BOTS_SERVICE_URL + '/api/bots';
        console.log('[BotsManager] 🔗 BOTS_SERVICE_URL:', this.BOTS_SERVICE_URL);
        
        // Уровень логирования: 'error' - только ошибки, 'info' - важные события, 'debug' - все
        this.logLevel = 'error'; // ✅ ОТКЛЮЧЕНЫ СПАМ-ЛОГИ - только ошибки

        // Состояние вкладки истории
        this.historyInitialized = false;
        this.currentHistoryTab = 'actions';
        this.historyBotSymbols = [];
        
        // Текущий таймфрейм системы (загружается из API)
        this.currentTimeframe = '6h'; // Дефолтное значение, будет обновлено при загрузке
        
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

    // Метод для получения перевода
    getTranslation(key) {
        const currentLang = document.documentElement.lang || 'ru';
        return TRANSLATIONS && TRANSLATIONS[currentLang] && TRANSLATIONS[currentLang][key] || key;
    }

    async init() {
        console.log('[BotsManager] 🚀 Инициализация менеджера ботов...');
        console.log('[BotsManager] 💡 Для включения debug логов: window.botsManager.logLevel = "debug"');
        
        try {
            // Инициализируем интерфейс
            this.initializeInterface();
            // Инициализируем селектор периода для AI
            this.initAIPeriodSelector();
            
            // КРИТИЧЕСКИ ВАЖНО: Инициализируем обработчик Auto Bot переключателя
            console.log('[BotsManager] 🤖 Инициализация обработчика Auto Bot переключателя...');
            this.initializeGlobalAutoBotToggle();
            this.initializeMobileAutoBotToggle();
            
            // Инициализируем управление таймфреймом
            this.initTimeframeControls();
            
            // Проверяем статус сервиса ботов
            await this.checkBotsService();
            
            // Синхронизируем позиции при инициализации
            if (this.serviceOnline) {
                console.log('[BotsManager] 🔄 Синхронизация позиций при инициализации...');
                
                // Загружаем делистинговые монеты при инициализации
                await this.loadDelistedCoins();
                
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
        
        // Инициализируем автосохранение конфигурации
        this.initializeAutoSave();
        
        // Загружаем счётчик зрелых монет
        this.loadMatureCoinsCount();
        
        // Принудительно применяем стили для читаемости
        this.applyReadabilityStyles();
        
        // Инициализируем кнопку обновления ручных позиций
        this.initializeManualPositionsControls();
        
        // Инициализируем кнопки загрузки RSI
        this.initializeRSILoadingButtons();
        
        // Инициализируем фильтры вкладки "Боты в работе"
        this.initActiveBotsFilters();
        
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
                '#defaultPositionMode',
                '#leverage',
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
                'active-bots': 'activeBotsTab',
                'analytics': 'analyticsTab',
                'history': 'historyTab'
            };
            
            const targetId = tabIdMap[tabName] || `${tabName}Tab`;
            const isActive = content.id === targetId;
            content.classList.toggle('active', isActive);
        });

        if (tabName !== 'config') this.hideFloatingSaveButton();
        // Загружаем данные для соответствующего таба
        switch(tabName) {
                    case 'management':
            this.loadCoinsRsiData();
            this.loadFiltersData(); // Загружаем фильтры для кнопок управления
            this.loadDuplicateSettings(); // Загружаем дублированные настройки
            break;
            case 'filters':
                this.loadFiltersData();
                // Загружаем актуальные данные монет для вкладки «Фильтры монет»; после загрузки обновляем поиск при активном запросе
                this.loadCoinsRsiData().then(() => {
                    const searchInput = document.getElementById('coinSearchInput');
                    const term = searchInput ? searchInput.value.trim() : '';
                    if (term) {
                        this.filterCoins(term);
                        this.updateSmartFilterControls(term);
                    }
                });
                break;
            case 'config':
                console.log('[BotsManager] 🎛️ Переключение на вкладку КОНФИГУРАЦИЯ');
                if (typeof this.applyConfigViewMode === 'function') this.applyConfigViewMode();
                setTimeout(() => this.applyReadabilityStyles(), 100);
                this.loadConfigurationData();
                this.showConfigurationLoading(false);
                this.createFloatingSaveButton();
                setTimeout(() => this.updateFloatingSaveButtonVisibility(), 300);
                break;
            case 'active-bots':
            case 'activeBotsTab':
                this.loadActiveBotsData();
                break;
            case 'history':
                this.initializeHistoryTab();
                break;
            case 'analytics':
                this.initializeAnalyticsTab();
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
            // Сохраняем счетчик при обновлении текста
            const countEl = buyFilterBtn.querySelector('#filterBuyZoneCount');
            // Извлекаем число из счетчика (может быть в формате " (6)" или "6")
            let count = '0';
            if (countEl) {
                const countText = countEl.textContent.trim();
                // Извлекаем число из строки вида " (6)" или "6"
                const match = countText.match(/\d+/);
                count = match ? match[0] : '0';
            }
            buyFilterBtn.innerHTML = `🟢 ≤${this.rsiLongThreshold} (<span id="filterBuyZoneCount">${count}</span>)`;
        }
        
        if (sellFilterBtn) {
            // Сохраняем счетчик при обновлении текста
            const countEl = sellFilterBtn.querySelector('#filterSellZoneCount');
            // Извлекаем число из счетчика (может быть в формате " (6)" или "6")
            let count = '0';
            if (countEl) {
                const countText = countEl.textContent.trim();
                // Извлекаем число из строки вида " (6)" или "6"
                const match = countText.match(/\d+/);
                count = match ? match[0] : '0';
            }
            sellFilterBtn.innerHTML = `🔴 ≥${this.rsiShortThreshold} (<span id="filterSellZoneCount">${count}</span>)`;
        }
        
        // Обновляем подписи тренд-фильтров с RSI значениями
        this.updateTrendFilterLabels();
        
        console.log(`[BotsManager] 🔄 Обновлены кнопки фильтров RSI: ≤${this.rsiLongThreshold}, ≥${this.rsiShortThreshold}`);
    }

    initActiveBotsFilters() {
        document.querySelectorAll('.active-bots-filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const clickedBtn = e.currentTarget;
                const filter = clickedBtn.dataset.filter;
                this.activeBotsFilter = filter;
                document.querySelectorAll('.active-bots-filter-btn').forEach(b => b.classList.remove('active'));
                clickedBtn.classList.add('active');
                this.renderActiveBotsDetails();
            });
        });
    }

    getFilteredActiveBotsForDetails() {
        const bots = Array.isArray(this.activeBots) ? this.activeBots : [];
        if (this.activeBotsFilter === 'all') return bots;
        return bots.filter(bot => {
            const pnl = bot.unrealized_pnl_usdt ?? bot.unrealized_pnl ?? 0;
            const pnlVal = Number.parseFloat(pnl) || 0;
            switch (this.activeBotsFilter) {
                case 'long': return bot.status === 'in_position_long';
                case 'short': return bot.status === 'in_position_short';
                case 'profitable': return pnlVal >= 0;
                case 'loss': return pnlVal < 0;
                default: return true;
            }
        });
    }

    /** Виртуальные позиции ПРИИ в виде объектов как у ботов — для отображения в списке «Боты в работе» с бейджем «Виртуальная». */
    getVirtualPositionsAsBots() {
        const list = Array.isArray(this.activeVirtualPositions) ? this.activeVirtualPositions : [];
        const filter = this.activeBotsFilter;
        return list
            .filter(v => {
                if (filter === 'long') return (v.direction || '').toUpperCase() === 'LONG';
                if (filter === 'short') return (v.direction || '').toUpperCase() === 'SHORT';
                return true;
            })
            .map((v, i) => {
                const entry = parseFloat(v.entry_price) || 0;
                const current = parseFloat(v.current_price) || 0;
                const isLong = (v.direction || '').toUpperCase() === 'LONG';
                const pnlPct = entry ? (isLong ? (current - entry) / entry : (entry - current) / entry) * 100 : 0;
                const pnlUsdt = 0; // виртуальная позиция без объёма в USDT
                return {
                    symbol: v.symbol,
                    is_virtual: true,
                    _virtualIndex: i,
                    position_side: isLong ? 'Long' : 'Short',
                    status: isLong ? 'virtual_long' : 'virtual_short',
                    entry_price: v.entry_price,
                    current_price: v.current_price,
                    unrealized_pnl_usdt: pnlUsdt,
                    unrealized_pnl: pnlPct,
                    config: {},
                    volume_value: 0,
                    position_size: 0
                };
            });
    }

    updateActiveBotsFilterCounts() {
        const bots = Array.isArray(this.activeBots) ? this.activeBots : [];
        const counts = {
            all: bots.length,
            long: bots.filter(b => b.status === 'in_position_long').length,
            short: bots.filter(b => b.status === 'in_position_short').length,
            profitable: bots.filter(b => ((b.unrealized_pnl_usdt ?? b.unrealized_pnl ?? 0) || 0) >= 0).length,
            loss: bots.filter(b => ((b.unrealized_pnl_usdt ?? b.unrealized_pnl ?? 0) || 0) < 0).length
        };
        const idMap = { all: 'All', long: 'Long', short: 'Short', profitable: 'Profitable', loss: 'Loss' };
        Object.keys(counts).forEach(key => {
            const el = document.getElementById(`activeBotsFilter${idMap[key]}Count`);
            if (el) el.textContent = counts[key];
        });
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
                // Получаем RSI с учетом текущего таймфрейма
                const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
                const rsiKey = `rsi${currentTimeframe}`;
                const rsiValue = coinData[rsiKey] || coinData.rsi6h || coinData.rsi || 50;
                const rsiClass = this.getRsiZoneClass(rsiValue);
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
    }

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
    }

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
    }
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
    }

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
    }

    renderCoinsList() {
        const coinsListElement = document.getElementById('coinsRsiList');
        if (!coinsListElement) {
            console.warn('[BotsManager] ⚠️ Элемент coinsRsiList не найден');
            return;
        }

        this.logDebug(`[BotsManager] 🎨 Отрисовка списка монет: ${this.coinsRsiData.length} монет`);
        
        if (this.coinsRsiData.length === 0) {
            const inProgress = this.lastUpdateInProgress === true;
            const stats = this.lastRsiStats || {};
            const processed = (stats.successful_coins || 0) + (stats.failed_coins || 0);
            const total = stats.total_coins || 0;
            console.warn('[BotsManager] ⚠️ Нет данных RSI для отображения', inProgress ? '(идёт загрузка на сервере)' : '');
            coinsListElement.innerHTML = `
                <div class="loading-state">
                    <p>⏳ ${inProgress ? (window.languageUtils.translate('loading_rsi_data') || 'Загрузка данных RSI...') : (window.languageUtils.translate('no_rsi_data') || 'Нет данных RSI')}</p>
                    <small>${inProgress
                        ? (window.languageUtils.translate('first_load_warning') || 'Первая загрузка может занять несколько минут. Не закрывайте вкладку.')
                        : (total ? `Расчёт завершён: ${processed}/${total} монет. Если список пуст — проверьте логи bots.py.` : 'Запустите bots.py и дождитесь завершения расчёта RSI.')}</small>
                </div>
            `;
            return;
        }
        
        // Получаем текущий таймфрейм для отображения данных
        const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
        const rsiKey = `rsi${currentTimeframe}`;
        const trendKey = `trend${currentTimeframe}`;
        
        const coinsHtml = this.coinsRsiData.map(coin => {
            const rsiValue = coin[rsiKey] || coin.rsi6h || coin.rsi || 50;
            const trendValue = coin[trendKey] || coin.trend6h || coin.trend || 'NEUTRAL';
            const rsiClass = this.getRsiZoneClass(rsiValue);
            const trendClass = trendValue ? `trend-${trendValue.toLowerCase()}` : 'trend-none';
            
            // Используем универсальную функцию для определения сигнала
            const effectiveSignal = this.getEffectiveSignal(coin);
            const signalClass = effectiveSignal === 'ENTER_LONG' ? 'enter-long' : 
                               effectiveSignal === 'ENTER_SHORT' ? 'enter-short' : '';
            
            // ✅ Проверяем недоступность для торговли
            const isUnavailable = effectiveSignal === 'UNAVAILABLE';
            const isDelisting = isUnavailable && (coin.trading_status === 'Closed' || coin.is_delisting || (this.delistedCoins && this.delistedCoins.includes(coin.symbol)));
            const isNewCoin = isUnavailable && coin.trading_status === 'Delivering';
            
            // Формируем классы
            const unavailableClass = isUnavailable ? 'unavailable-coin' : '';
            const delistingClass = isDelisting ? 'delisting-coin' : '';
            const newCoinClass = isNewCoin ? 'new-coin' : '';
            
            // Проверяем, есть ли ручная позиция
            const isManualPosition = coin.manual_position || false;
            const manualClass = isManualPosition ? 'manual-position' : '';
            
            // Проверяем, зрелая ли монета
            const isMature = coin.is_mature || false;
            const matureClass = isMature ? 'mature-coin' : '';
            
            // Убраны спам логи для лучшей отладки
            
            return `
                <li class="coin-item ${rsiClass} ${trendClass} ${signalClass} ${manualClass} ${matureClass} ${unavailableClass} ${delistingClass} ${newCoinClass}" data-symbol="${coin.symbol}">
                    <div class="coin-item-content">
                        <div class="coin-header">
                            <span class="coin-symbol">${coin.symbol}</span>
                            <div class="coin-header-right">
                                ${isManualPosition ? '<span class="manual-position-indicator" title="Ручная позиция">✋</span>' : ''}
                                ${isMature ? '<span class="mature-coin-indicator" title="Зрелая монета">💎</span>' : ''}
                                ${isDelisting ? '<span class="delisting-indicator" title="Монета на делистинге">⚠️</span>' : ''}
                                ${isNewCoin ? '<span class="new-coin-indicator" title="Новая монета (включение в листинг)">🆕</span>' : ''}
                                ${this.generateWarningIndicator(coin)}
                                ${(() => {
                                    const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
                                    const rsiKey = `rsi${currentTimeframe}`;
                                    const rsiValue = coin[rsiKey] || coin.rsi6h || coin.rsi || 50;
                                    return `<span class="coin-rsi ${this.getRsiZoneClass(rsiValue)}">${rsiValue}</span>`;
                                })()}
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
                            ${(() => {
                                const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
                                const trendKey = `trend${currentTimeframe}`;
                                const trendValue = coin[trendKey] || coin.trend6h || coin.trend || 'NEUTRAL';
                                return `<span class="coin-trend ${trendValue}">${trendValue}</span>`;
                            })()}
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
                stochDescription = window.languageUtils.translate('stochastic_oversold').replace('{k}', stochK.toFixed(1));
            } else if (stochK > 80) {
                stochIcon = '⬆️';
                stochStatus = 'OVERBOUGHT';
                stochDescription = window.languageUtils.translate('stochastic_overbought').replace('{k}', stochK.toFixed(1));
            } else {
                stochIcon = '➡️';
                stochStatus = 'NEUTRAL';
                stochDescription = window.languageUtils.translate('stochastic_neutral').replace('{k}', stochK.toFixed(1));
            }
            
            // Добавляем информацию о пересечении %K и %D
            let crossoverInfo = '';
            if (stochK > stochD) {
                crossoverInfo = ' ' + window.languageUtils.translate('stochastic_bullish_signal').replace('{d}', stochD.toFixed(1));
            } else if (stochK < stochD) {
                crossoverInfo = ' ' + window.languageUtils.translate('stochastic_bearish_signal').replace('{d}', stochD.toFixed(1));
            } else {
                crossoverInfo = ' (%K = %D - ' + window.languageUtils.translate('neutral') + ')';
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
            return '';
        }
        
        const isBlocked = timeFilterInfo.blocked;
        const reason = timeFilterInfo.reason || '';
        const lastExtremeCandlesAgo = timeFilterInfo.last_extreme_candles_ago;
        const calmCandles = timeFilterInfo.calm_candles;
        
        let icon = '';
        let className = '';
        let title = '';
        
        // Определяем тип статуса по причине
        if (reason.includes('Ожидание') || reason.includes('ожидание') || reason.includes('прошло только')) {
            // Ожидание - показываем с иконкой ожидания
            icon = '⏳';
            className = 'time-filter-waiting';
            title = `Временной фильтр: ${reason}`;
        } else if (isBlocked) {
            // Фильтр блокирует вход
            icon = '⏰';
            className = 'time-filter-blocked';
            title = `Временной фильтр блокирует: ${reason}`;
        } else {
            // Фильтр пройден, показываем информацию
            icon = '✅';
            className = 'time-filter-allowed';
            title = `Временной фильтр: ${reason}`;
            if (lastExtremeCandlesAgo !== null && lastExtremeCandlesAgo !== undefined) {
                title += ` (${lastExtremeCandlesAgo} свечей назад)`;
            }
            if (calmCandles !== null && calmCandles !== undefined) {
                title += ` (${calmCandles} спокойных свечей)`;
            }
        }
        
        // ВСЕГДА показываем иконку, если есть reason
        if (reason && icon) {
            return `<div class="time-filter-info ${className}" title="${title}" style="margin-left: 4px; font-size: 14px; cursor: help;">${icon}</div>`;
        }
        
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
     * @returns {string} - Эффективный сигнал (ENTER_LONG, ENTER_SHORT, WAIT, UNAVAILABLE)
     */
    getEffectiveSignal(coin) {
        // ✅ ПРОВЕРКА СТАТУСА ТОРГОВЛИ: Исключаем монеты недоступные для торговли
        if (coin.is_delisting || coin.trading_status === 'Closed' || coin.trading_status === 'Delivering') {
            return 'UNAVAILABLE'; // Статус для недоступных для торговли монет (делистинг + новые монеты)
        }
        
        // ✅ ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Исключаем известные делистинговые монеты
        // Получаем список делистинговых монет с сервера
        if (this.delistedCoins && this.delistedCoins.includes(coin.symbol)) {
            return 'UNAVAILABLE';
        }
        
        // ✅ КРИТИЧНО: Получаем базовый сигнал для проверки блокировок
        let signal = coin.signal || 'WAIT';
        
        // ✅ ПРОВЕРКА БЛОКИРОВОК ФИЛЬТРОВ: Если монета заблокирована - возвращаем WAIT
        // Это ВАЖНО: монеты с заблокированными фильтрами НЕ должны отображаться в списке LONG/SHORT!
        
        // 1. Проверяем ExitScam фильтр
        if (coin.blocked_by_exit_scam === true) {
            return 'WAIT';
        }
        
        // 2. Проверяем RSI Time фильтр
        if (coin.blocked_by_rsi_time === true) {
            return 'WAIT';
        }
        
        // 3. Проверяем защиту от повторных входов после убытка
        if (coin.blocked_by_loss_reentry === true) {
            return 'WAIT';
        }
        
        // 4. Проверяем зрелость монеты
        if (coin.is_mature === false) {
            return 'WAIT';
        }
        
        // 4. Проверяем Whitelist/Blacklist (Scope)
        if (coin.blocked_by_scope === true) {
            return 'WAIT';
        }
        
        // ✅ КРИТИЧНО: Если API уже установил effective_signal (в т.ч. WAIT после проверки AI) — используем его.
        // Иначе список LONG/SHORT слева будет показывать монеты, которые API исключил (расхождение с карточкой).
        if (coin.effective_signal !== undefined && coin.effective_signal !== null && coin.effective_signal !== '') {
            return coin.effective_signal;
        }
        
        // Если базовый сигнал WAIT - возвращаем сразу
        if (signal === 'WAIT') {
            return 'WAIT';
        }
        
        // ✅ ПРОВЕРКА Enhanced RSI: Если включен и дает другой сигнал - используем его
        if (coin.enhanced_rsi && coin.enhanced_rsi.enabled && coin.enhanced_rsi.enhanced_signal) {
            const enhancedSignal = coin.enhanced_rsi.enhanced_signal;
            // Если Enhanced RSI говорит WAIT - блокируем
            if (enhancedSignal === 'WAIT') {
                return 'WAIT';
            }
            signal = enhancedSignal;
        }
        
        // ✅ ПРОВЕРКА ФИЛЬТРОВ ТРЕНДОВ (если Enhanced RSI не заблокировал)
        const autoConfig = this.cachedAutoBotConfig || {};
        const avoidDownTrend = autoConfig.avoid_down_trend === true;
        const avoidUpTrend = autoConfig.avoid_up_trend === true;
        // Получаем RSI и тренд с учетом текущего таймфрейма
        const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
        const rsiKey = `rsi${currentTimeframe}`;
        const trendKey = `trend${currentTimeframe}`;
        const rsi = coin[rsiKey] || coin.rsi6h || coin.rsi || 50;
        const trend = coin[trendKey] || coin.trend6h || coin.trend || 'NEUTRAL';
        const rsiLongThreshold = autoConfig.rsi_long_threshold || 29;
        const rsiShortThreshold = autoConfig.rsi_short_threshold || 71;
        
        if (signal === 'ENTER_LONG' && avoidDownTrend && rsi <= rsiLongThreshold && trend === 'DOWN') {
            return 'WAIT';
        }
        
        if (signal === 'ENTER_SHORT' && avoidUpTrend && rsi >= rsiShortThreshold && trend === 'UP') {
            return 'WAIT';
        }
        
        // Возвращаем проверенный сигнал (effective_signal из API уже обработан в начале функции)
        return signal;
    }

    updateSignalCounters() {
        // Подсчитываем все категории
        const allCount = this.coinsRsiData.length;
        const longCount = this.coinsRsiData.filter(coin => this.getEffectiveSignal(coin) === 'ENTER_LONG').length;
        const shortCount = this.coinsRsiData.filter(coin => this.getEffectiveSignal(coin) === 'ENTER_SHORT').length;
        // Получаем текущий таймфрейм для подсчета
        const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
        const rsiKey = `rsi${currentTimeframe}`;
        const buyZoneCount = this.coinsRsiData.filter(coin => {
            const rsi = coin[rsiKey] || coin.rsi6h || coin.rsi;
            return rsi && rsi <= this.rsiLongThreshold;
        }).length;
        const sellZoneCount = this.coinsRsiData.filter(coin => {
            const rsi = coin[rsiKey] || coin.rsi6h || coin.rsi;
            return rsi && rsi >= this.rsiShortThreshold;
        }).length;
        // Используем тот же currentTimeframe для подсчета трендов
        const trendKey = `trend${currentTimeframe}`;
        const trendUpCount = this.coinsRsiData.filter(coin => {
            const trend = coin[trendKey] || coin.trend6h || coin.trend;
            return trend === 'UP';
        }).length;
        const trendDownCount = this.coinsRsiData.filter(coin => {
            const trend = coin[trendKey] || coin.trend6h || coin.trend;
            return trend === 'DOWN';
        }).length;
        const manualPositionCount = this.coinsRsiData.filter(coin => coin.manual_position === true).length;
        const unavailableCount = this.coinsRsiData.filter(coin => this.getEffectiveSignal(coin) === 'UNAVAILABLE').length;
        const delistedCount = this.coinsRsiData.filter(coin =>
            coin.trading_status === 'Closed' || coin.is_delisting || (this.delistedCoins && this.delistedCoins.includes(coin.symbol))
        ).length;
        
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
        const delistedCountEl = document.getElementById('delistedCoinsCount');
        
        
        // Обновляем счетчики фильтров
        if (allCountEl) allCountEl.textContent = allCount;
        
        if (buyZoneCountEl) buyZoneCountEl.textContent = ` (${buyZoneCount})`;
        if (sellZoneCountEl) sellZoneCountEl.textContent = ` (${sellZoneCount})`;
        if (trendUpCountEl) trendUpCountEl.textContent = trendUpCount;
        if (trendDownCountEl) trendDownCountEl.textContent = trendDownCount;
        if (longCountEl) longCountEl.textContent = longCount;
        if (shortCountEl) shortCountEl.textContent = shortCount;
        if (manualCountEl) manualCountEl.textContent = `(${manualPositionCount})`;
        if (delistedCountEl) delistedCountEl.textContent = `(${delistedCount})`;
        
        // ✅ Логируем недоступные для торговли монеты
        if (unavailableCount > 0) {
            const unavailableCoins = this.coinsRsiData.filter(coin => this.getEffectiveSignal(coin) === 'UNAVAILABLE');
            const delistingCoins = unavailableCoins.filter(coin => coin.trading_status === 'Closed' || coin.is_delisting);
            const newCoins = unavailableCoins.filter(coin => coin.trading_status === 'Delivering');
            
            if (delistingCoins.length > 0) {
                console.warn(`[BotsManager] ⚠️ Найдено ${delistingCoins.length} монет на делистинге:`, delistingCoins.map(coin => coin.symbol));
            }
            if (newCoins.length > 0) {
                console.info(`[BotsManager] ℹ️ Найдено ${newCoins.length} новых монет (Delivering):`, newCoins.map(coin => coin.symbol));
            }
        }
        
        this.logDebug(`[BotsManager] 📊 Счетчики фильтров: ALL=${allCount}, BUY=${buyZoneCount}, SELL=${sellZoneCount}, UP=${trendUpCount}, DOWN=${trendDownCount}, LONG=${longCount}, SHORT=${shortCount}, MANUAL=${manualPositionCount}, DELISTED=${delistedCount}, UNAVAILABLE=${unavailableCount}`);
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
        // Получаем текущий таймфрейм для отображения
        const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
        const rsiKey = `rsi${currentTimeframe}`;
        const trendKey = `trend${currentTimeframe}`;
        
        const rsiElement = document.getElementById('selectedCoinRSI');
        const trendElement = document.getElementById('selectedCoinTrend');
        const zoneElement = document.getElementById('selectedCoinZone');
        const signalElement = document.getElementById('selectedCoinSignal');
        const changeElement = document.getElementById('selectedCoinChange');

        console.log('[BotsManager] 🔍 Найденные элементы:', {
            symbolElement: !!symbolElement,
            priceElement: !!priceElement,
            rsiElement: !!rsiElement,
            trendElement: !!trendElement,
            zoneElement: !!zoneElement,
            signalElement: !!signalElement,
            changeElement: !!changeElement
        });

        if (symbolElement) {
            const exchangeUrl = this.getExchangeLink(coin.symbol, 'bybit');
            
            // Проверяем статус делистинга
            const isDelisting = coin.is_delisting || coin.trading_status === 'Closed' || coin.trading_status === 'Delivering';
            const delistedTag = isDelisting ? '<span class="delisted-status">DELISTED</span>' : '';
            
            symbolElement.innerHTML = `
                🪙 ${coin.symbol} 
                ${delistedTag}
                <a href="${exchangeUrl}" target="_blank" class="exchange-link" title="Открыть на Bybit">
                    🔗
                </a>
            `;
            console.log('[BotsManager] ✅ Символ обновлен:', coin.symbol, isDelisting ? '(DELISTED)' : '');
        }
        
        // Используем правильные поля из RSI данных
        if (priceElement) {
            const price = coin.current_price || coin.mark_price || coin.last_price || coin.price || 0;
            priceElement.textContent = `$${price.toFixed(6)}`;
            console.log('[BotsManager] ✅ Цена обновлена:', price);
        }
        
        if (rsiElement) {
            const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
            const rsiKey = `rsi${currentTimeframe}`;
            const enhancedRsiKey = `rsi_${currentTimeframe.replace('h', 'H')}`;
            const rsi = coin.enhanced_rsi?.[enhancedRsiKey] || coin[rsiKey] || coin.rsi6h || coin.rsi || '-';
            rsiElement.textContent = rsi;
            rsiElement.className = `value rsi-indicator ${this.getRsiZoneClass(rsi)}`;
            console.log('[BotsManager] ✅ RSI обновлен:', rsi);
        }
        
        if (trendElement) {
            const trend = coin[trendKey] || coin.trend6h || coin.trend || 'NEUTRAL';
            trendElement.textContent = trend;
            trendElement.className = `value trend-indicator ${trend}`;
            console.log('[BotsManager] ✅ Тренд обновлен:', trend);
            
            // ✅ Обновляем подсказку в зависимости от настроек избегания трендов
            const trendHintElement = document.getElementById('trendHint');
            if (trendHintElement) {
                // Получаем текущие настройки из кэша конфигурации
                const avoidDownTrend = this.cachedAutoBotConfig?.avoid_down_trend !== false;
                const avoidUpTrend = this.cachedAutoBotConfig?.avoid_up_trend !== false;
                
                // Если оба фильтра отключены - тренд не используется
                if (!avoidDownTrend && !avoidUpTrend) {
                    trendHintElement.textContent = '(фильтры трендов отключены)';
                    trendHintElement.style.color = 'var(--warning-color)';
                } else if (!avoidDownTrend && avoidUpTrend) {
                    trendHintElement.textContent = '(DOWN тренд не блокирует LONG)';
                    trendHintElement.style.color = 'var(--text-muted)';
                } else if (avoidDownTrend && !avoidUpTrend) {
                    trendHintElement.textContent = '(UP тренд не блокирует SHORT)';
                    trendHintElement.style.color = 'var(--text-muted)';
                } else {
                    // Оба фильтра включены - показываем период анализа с учетом текущего таймфрейма
                    const period = this.cachedAutoBotConfig?.trend_analysis_period || 30;
                    // Пересчитываем дни для текущего таймфрейма
                    const timeframeHours = {
                        '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60, '30m': 30/60,
                        '1h': 1, '2h': 2, '4h': 4, '6h': 6, '8h': 8, '12h': 12, '1d': 24
                    };
                    const hoursPerCandle = timeframeHours[currentTimeframe] || 6;
                    const days = (period * hoursPerCandle / 24).toFixed(1);
                    trendHintElement.textContent = `(анализ за ${days} дней на ${currentTimeframe.toUpperCase()})`;
                    trendHintElement.style.color = 'var(--text-muted)';
                }
            }
        }
        
        // ❌ EMA данные больше не используются и не отображаются
        
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
                } else if (bot.status === 'running' || bot.status === 'waiting') {
                    // Бот запущен — вход по рынку при появлении сигнала
                    botStatus = window.languageUtils.translate('entry_by_market');
                } else if (bot.status === 'in_position_long') {
                    botStatus = window.languageUtils.translate('active_status');
                } else if (bot.status === 'in_position_short') {
                    botStatus = window.languageUtils.translate('active_status');
                } else {
                    botStatus = bot.status || window.languageUtils.translate('bot_not_created');
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
                const signalText = stochK > stochD 
                    ? window.languageUtils.getTranslation('stochastic_bullish_signal', {d: stochD.toFixed(1)})
                    : window.languageUtils.getTranslation('stochastic_bearish_signal', {d: stochD.toFixed(1)});
                const zoneText = window.languageUtils.getTranslation('stochastic_oversold', {k: stochK.toFixed(1)});
                stochValue = `<span style="color: var(--green-text);">${zoneText}</span><br><span style="color: ${stochK > stochD ? 'var(--green-text)' : 'var(--red-text)'};">${signalText}</span>`;
            } else if (stochK > 80) {
                stochStatus = 'OVERBOUGHT';
                const signalText = stochK > stochD 
                    ? window.languageUtils.getTranslation('stochastic_bullish_signal', {d: stochD.toFixed(1)})
                    : window.languageUtils.getTranslation('stochastic_bearish_signal', {d: stochD.toFixed(1)});
                const zoneText = window.languageUtils.getTranslation('stochastic_overbought', {k: stochK.toFixed(1)});
                stochValue = `<span style="color: var(--red-text);">${zoneText}</span><br><span style="color: ${stochK > stochD ? 'var(--green-text)' : 'var(--red-text)'};">${signalText}</span>`;
            } else {
                stochStatus = 'NEUTRAL';
                const signalText = stochK > stochD 
                    ? window.languageUtils.getTranslation('stochastic_bullish_signal', {d: stochD.toFixed(1)})
                    : window.languageUtils.getTranslation('stochastic_bearish_signal', {d: stochD.toFixed(1)});
                const zoneText = window.languageUtils.getTranslation('stochastic_neutral', {k: stochK.toFixed(1)});
                stochValue = `<span style="color: var(--warning-color);">${zoneText}</span><br><span style="color: ${stochK > stochD ? 'var(--green-text)' : 'var(--red-text)'};">${signalText}</span>`;
            }
        } else if (coin.enhanced_rsi && coin.enhanced_rsi.confirmations) {
            const stochK = coin.enhanced_rsi.confirmations.stoch_rsi_k;
            const stochD = coin.enhanced_rsi.confirmations.stoch_rsi_d || 0;
            if (stochK !== undefined && stochK !== null) {
                let stochStatus = '';
                let crossoverInfo = '';
                
                if (stochK < 20) {
                    stochStatus = 'OVERSOLD';
                    const signalText = stochK > stochD 
                        ? window.languageUtils.getTranslation('stochastic_bullish_signal', {d: stochD.toFixed(1)})
                        : window.languageUtils.getTranslation('stochastic_bearish_signal', {d: stochD.toFixed(1)});
                    const zoneText = window.languageUtils.getTranslation('stochastic_oversold', {k: stochK.toFixed(1)});
                    stochValue = `<span style="color: var(--green-text);">${zoneText}</span><br><span style="color: ${stochK > stochD ? 'var(--green-text)' : 'var(--red-text)'};">${signalText}</span>`;
                } else if (stochK > 80) {
                    stochStatus = 'OVERBOUGHT';
                    const signalText = stochK > stochD 
                        ? window.languageUtils.getTranslation('stochastic_bullish_signal', {d: stochD.toFixed(1)})
                        : window.languageUtils.getTranslation('stochastic_bearish_signal', {d: stochD.toFixed(1)});
                    const zoneText = window.languageUtils.getTranslation('stochastic_overbought', {k: stochK.toFixed(1)});
                    stochValue = `<span style="color: var(--red-text);">${zoneText}</span><br><span style="color: ${stochK > stochD ? 'var(--green-text)' : 'var(--red-text)'};">${signalText}</span>`;
                } else {
                    stochStatus = 'NEUTRAL';
                    const signalText = stochK > stochD 
                        ? window.languageUtils.getTranslation('stochastic_bullish_signal', {d: stochD.toFixed(1)})
                        : window.languageUtils.getTranslation('stochastic_bearish_signal', {d: stochD.toFixed(1)});
                    const zoneText = window.languageUtils.getTranslation('stochastic_neutral', {k: stochK.toFixed(1)});
                    stochValue = `<span style="color: var(--warning-color);">${zoneText}</span><br><span style="color: ${stochK > stochD ? 'var(--green-text)' : 'var(--red-text)'};">${signalText}</span>`;
                }
            }
        }
        
        if (stochValue) {
            activeStatusData.stochastic_rsi = stochValue;
        }
        
        // ExitScam защита (ExitScam Protection) - проверяем разные поля
        // ✅ ИСПРАВЛЕНИЕ: Используем exit_scam_info если доступно
        if (coin.exit_scam_info) {
            const exitScamInfo = coin.exit_scam_info;
            const isBlocked = exitScamInfo.blocked;
            const reason = exitScamInfo.reason || '';
            
            if (isBlocked) {
                activeStatusData.exit_scam = `Блокирует: ${reason}`;
            } else {
                activeStatusData.exit_scam = `Пройден: ${reason}`;
            }
        } else if (coin.exit_scam_status && coin.exit_scam_status !== 'NONE' && coin.exit_scam_status !== null) {
            activeStatusData.exit_scam = coin.exit_scam_status;
        } else if (coin.exit_scam && coin.exit_scam !== 'NONE') {
            activeStatusData.exit_scam = coin.exit_scam;
        } else if (coin.scam_status && coin.scam_status !== 'NONE') {
            activeStatusData.exit_scam = coin.scam_status;
        } else if (coin.blocked_by_exit_scam === true) {
            activeStatusData.exit_scam = 'Блокирует: обнаружены резкие движения цены';
        }
        
        // RSI Time Filter - преобразуем time_filter_info в строковый статус
        if (coin.time_filter_info) {
            const timeFilter = coin.time_filter_info;
            const isBlocked = timeFilter.blocked;
            const reason = timeFilter.reason || '';
            const calmCandles = timeFilter.calm_candles || 0;
            
            console.log(`[RSI_TIME_FILTER] ${coin.symbol}: time_filter_info =`, timeFilter);
            
            if (isBlocked) {
                if (reason.includes('Ожидание') || reason.includes('ожидание') || reason.includes('прошло только')) {
                    activeStatusData.rsi_time_filter = `WAITING: ${reason}`;
                } else {
                    activeStatusData.rsi_time_filter = `BLOCKED: ${reason}`;
                }
            } else {
                activeStatusData.rsi_time_filter = `ALLOWED: ${reason}`;
            }
            
            console.log(`[RSI_TIME_FILTER] ${coin.symbol}: activeStatusData.rsi_time_filter =`, activeStatusData.rsi_time_filter);
        } else if (coin.rsi_time_filter && coin.rsi_time_filter !== 'NONE' && coin.rsi_time_filter !== null) {
            activeStatusData.rsi_time_filter = coin.rsi_time_filter;
        } else if (coin.time_filter && coin.time_filter !== 'NONE') {
            activeStatusData.rsi_time_filter = coin.time_filter;
        } else if (coin.rsi_time_status && coin.rsi_time_status !== 'NONE') {
            activeStatusData.rsi_time_filter = coin.rsi_time_status;
        } else {
            console.log(`[RSI_TIME_FILTER] ${coin.symbol}: НЕТ time_filter_info и других полей`);
        }
        
        // Защита от повторных входов после убыточных закрытий - преобразуем loss_reentry_info в строковый статус
        if (coin.loss_reentry_info) {
            const lossReentry = coin.loss_reentry_info;
            const isBlocked = lossReentry.blocked;
            const reason = lossReentry.reason || '';
            
            if (isBlocked) {
                activeStatusData.loss_reentry_protection = `BLOCKED: ${reason}`;
            } else {
                activeStatusData.loss_reentry_protection = `ALLOWED: ${reason}`;
            }
            
            console.log(`[LOSS_REENTRY] ${coin.symbol}: activeStatusData.loss_reentry_protection =`, activeStatusData.loss_reentry_protection);
        }
        
        // Enhanced RSI информация (если включена)
        if (coin.enhanced_rsi && coin.enhanced_rsi.enabled) {
            const enhancedSignal = coin.enhanced_rsi.enhanced_signal;
            const baseSignal = coin.signal || 'WAIT';
            const enhancedReason = coin.enhanced_rsi.enhanced_reason || '';
            const warningMessage = coin.enhanced_rsi.warning_message || '';
            const confirmations = coin.enhanced_rsi.confirmations || {};
            
            let enhancedRsiText = '';
            
            // Функция для преобразования технической причины в понятный текст
            const parseEnhancedReason = (reason) => {
                if (!reason) return '';
                
                // Парсим причину для понятного отображения
                if (reason.includes('fresh_oversold')) {
                    const rsiMatch = reason.match(/fresh_oversold_(\d+\.?\d*)/);
                    const rsi = rsiMatch ? rsiMatch[1] : '';
                    const factors = [];
                    
                    if (reason.includes('base_oversold')) factors.push('RSI в зоне перепроданности');
                    if (reason.includes('bullish_divergence')) factors.push('бычья дивергенция');
                    if (reason.includes('stoch_oversold')) factors.push('Stochastic RSI перепродан');
                    if (reason.includes('volume_confirm')) factors.push('подтверждение объемом');
                    
                    if (factors.length > 0) {
                        return `RSI ${rsi} недавно вошел в зону перепроданности. Подтверждения: ${factors.join(', ')}`;
                    }
                    return `RSI ${rsi} недавно вошел в зону перепроданности`;
                } else if (reason.includes('enhanced_oversold')) {
                    const rsiMatch = reason.match(/enhanced_oversold_(\d+\.?\d*)/);
                    const rsi = rsiMatch ? rsiMatch[1] : '';
                    const factors = [];
                    
                    if (reason.includes('base_oversold')) factors.push('RSI в зоне перепроданности');
                    if (reason.includes('bullish_divergence')) factors.push('бычья дивергенция');
                    if (reason.includes('stoch_oversold')) factors.push('Stochastic RSI перепродан');
                    if (reason.includes('volume_confirm')) factors.push('подтверждение объемом');
                    
                    if (factors.length > 0) {
                        return `RSI ${rsi} в зоне перепроданности. Подтверждения: ${factors.join(', ')}`;
                    }
                    return `RSI ${rsi} в зоне перепроданности`;
                } else if (reason.includes('fresh_overbought')) {
                    const rsiMatch = reason.match(/fresh_overbought_(\d+\.?\d*)/);
                    const rsi = rsiMatch ? rsiMatch[1] : '';
                    const factors = [];
                    
                    if (reason.includes('base_overbought')) factors.push('RSI в зоне перекупленности');
                    if (reason.includes('bearish_divergence')) factors.push('медвежья дивергенция');
                    if (reason.includes('stoch_overbought')) factors.push('Stochastic RSI перекуплен');
                    if (reason.includes('volume_confirm')) factors.push('подтверждение объемом');
                    
                    if (factors.length > 0) {
                        return `RSI ${rsi} недавно вошел в зону перекупленности. Подтверждения: ${factors.join(', ')}`;
                    }
                    return `RSI ${rsi} недавно вошел в зону перекупленности`;
                } else if (reason.includes('enhanced_overbought')) {
                    const rsiMatch = reason.match(/enhanced_overbought_(\d+\.?\d*)/);
                    const rsi = rsiMatch ? rsiMatch[1] : '';
                    const factors = [];
                    
                    if (reason.includes('base_overbought')) factors.push('RSI в зоне перекупленности');
                    if (reason.includes('bearish_divergence')) factors.push('медвежья дивергенция');
                    if (reason.includes('stoch_overbought')) factors.push('Stochastic RSI перекуплен');
                    if (reason.includes('volume_confirm')) factors.push('подтверждение объемом');
                    
                    if (factors.length > 0) {
                        return `RSI ${rsi} в зоне перекупленности. Подтверждения: ${factors.join(', ')}`;
                    }
                    return `RSI ${rsi} в зоне перекупленности`;
                } else if (reason.includes('strict_mode_bullish_divergence')) {
                    const rsiMatch = reason.match(/strict_mode_bullish_divergence_(\d+\.?\d*)/);
                    const rsi = rsiMatch ? rsiMatch[1] : '';
                    return `Строгий режим: RSI ${rsi} + бычья дивергенция`;
                } else if (reason.includes('strict_mode_bearish_divergence')) {
                    const rsiMatch = reason.match(/strict_mode_bearish_divergence_(\d+\.?\d*)/);
                    const rsi = rsiMatch ? rsiMatch[1] : '';
                    return `Строгий режим: RSI ${rsi} + медвежья дивергенция`;
                } else if (reason.includes('strict_mode_no_divergence')) {
                    const rsiMatch = reason.match(/strict_mode_no_divergence_(\d+\.?\d*)/);
                    const rsi = rsiMatch ? rsiMatch[1] : '';
                    return `Строгий режим: требуется дивергенция (RSI ${rsi})`;
                } else if (reason.includes('insufficient_confirmation')) {
                    const rsiMatch = reason.match(/oversold_but_insufficient_confirmation_(\d+\.?\d*)/);
                    const rsi = rsiMatch ? rsiMatch[1] : '';
                    const durationMatch = reason.match(/duration_(\d+)/);
                    const duration = durationMatch ? durationMatch[1] : '';
                    return `RSI ${rsi} в зоне ${duration} свечей, но недостаточно подтверждений`;
                } else if (reason.includes('enhanced_neutral')) {
                    const rsiMatch = reason.match(/enhanced_neutral_(\d+\.?\d*)/);
                    const rsi = rsiMatch ? rsiMatch[1] : '';
                    return `RSI ${rsi} в нейтральной зоне`;
                }
                
                // Если не распознано - возвращаем как есть, но убираем подчеркивания
                return reason.replace(/_/g, ' ');
            };
            
            if (enhancedSignal) {
                // Если Enhanced RSI изменил сигнал
                if (enhancedSignal !== baseSignal && baseSignal !== 'WAIT') {
                    const reasonText = parseEnhancedReason(enhancedReason);
                    enhancedRsiText = `Сигнал изменен: ${baseSignal} → ${enhancedSignal}`;
                    if (reasonText) {
                        enhancedRsiText += `. ${reasonText}`;
                    }
                } else if (enhancedSignal === 'WAIT' && baseSignal !== 'WAIT') {
                    const reasonText = parseEnhancedReason(enhancedReason);
                    enhancedRsiText = `Блокировка: базовый сигнал ${baseSignal} заблокирован Enhanced RSI`;
                    if (reasonText) {
                        enhancedRsiText += `. ${reasonText}`;
                    }
                } else if (enhancedSignal === baseSignal || enhancedSignal === 'ENTER_LONG' || enhancedSignal === 'ENTER_SHORT') {
                    // Enhanced RSI подтвердил или разрешил сигнал
                    const reasonText = parseEnhancedReason(enhancedReason);
                    if (reasonText) {
                        enhancedRsiText = `${enhancedSignal === 'ENTER_LONG' ? '✅ LONG разрешен' : enhancedSignal === 'ENTER_SHORT' ? '✅ SHORT разрешен' : `Сигнал: ${enhancedSignal}`}. ${reasonText}`;
                    } else {
                        enhancedRsiText = `${enhancedSignal === 'ENTER_LONG' ? '✅ LONG разрешен' : enhancedSignal === 'ENTER_SHORT' ? '✅ SHORT разрешен' : `Сигнал: ${enhancedSignal}`}`;
                    }
                } else {
                    const reasonText = parseEnhancedReason(enhancedReason);
                    enhancedRsiText = `Сигнал: ${enhancedSignal}`;
                    if (reasonText) {
                        enhancedRsiText += `. ${reasonText}`;
                    }
                }
                
                if (warningMessage) {
                    enhancedRsiText += ` | ${warningMessage}`;
                }
            } else {
                enhancedRsiText = 'Включена, но сигнал не определен';
            }
            
            if (enhancedRsiText) {
                activeStatusData.enhanced_rsi = enhancedRsiText;
            }
        }
        // Функция для полной проверки всех фильтров и сбора причин блокировки
        const checkAllBlockingFilters = (coin) => {
            const blockReasons = [];
            const autoConfig = this.cachedAutoBotConfig || {};
            const baseSignal = coin.signal || 'WAIT';
            // Получаем RSI с учетом текущего таймфрейма
            const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
            const rsiKey = `rsi${currentTimeframe}`;
            const trendKey = `trend${currentTimeframe}`;
            const rsi = coin[rsiKey] || coin.rsi6h || coin.rsi || 50;
            const trend = coin[trendKey] || coin.trend6h || coin.trend || 'NEUTRAL';
            const rsiLongThreshold = autoConfig.rsi_long_threshold || 29;
            const rsiShortThreshold = autoConfig.rsi_short_threshold || 71;
            
            // 1. ExitScam — показываем только если фильтр включён
            if (autoConfig.exit_scam_enabled !== false && coin.blocked_by_exit_scam === true) {
                const exitScamInfo = coin.exit_scam_info;
                if (exitScamInfo && exitScamInfo.reason) {
                    blockReasons.push(`ExitScam фильтр: ${exitScamInfo.reason}`);
                } else {
                    blockReasons.push('ExitScam фильтр');
                }
            }
            
            // 2. RSI Time — показываем только если фильтр включён
            if (autoConfig.rsi_time_filter_enabled !== false && coin.blocked_by_rsi_time === true) {
                const timeFilterInfo = coin.time_filter_info;
                if (timeFilterInfo && timeFilterInfo.reason) {
                    blockReasons.push(`RSI Time фильтр: ${timeFilterInfo.reason}`);
                } else {
                    blockReasons.push('RSI Time фильтр');
                }
            }
            
            // 3. Защита от повторных входов — показываем только если фильтр включён
            if (autoConfig.loss_reentry_protection !== false && coin.blocked_by_loss_reentry === true) {
                const lossReentryInfo = coin.loss_reentry_info;
                if (lossReentryInfo && lossReentryInfo.reason) {
                    blockReasons.push(`Защита от повторных входов: ${lossReentryInfo.reason}`);
                } else {
                    blockReasons.push('Защита от повторных входов после убытка');
                }
            }
            
            // 4. Зрелость монеты — показываем только если проверка зрелости включена
            if (autoConfig.enable_maturity_check !== false && coin.is_mature === false) {
                blockReasons.push('Незрелая монета');
            }
            
            // 5. Whitelist/Blacklist (scope)
            if (coin.blocked_by_scope === true) {
                blockReasons.push('Whitelist/Blacklist');
            }
            
            // 5. Проверяем Enhanced RSI
            const enhancedRsiEnabled = coin.enhanced_rsi && coin.enhanced_rsi.enabled;
            const enhancedSignal = enhancedRsiEnabled ? coin.enhanced_rsi.enhanced_signal : null;
            const enhancedReason = enhancedRsiEnabled ? (coin.enhanced_rsi.enhanced_reason || '') : '';
            
            if (enhancedRsiEnabled && enhancedSignal === 'WAIT' && baseSignal !== 'WAIT') {
                // Enhanced RSI заблокировал сигнал
                let enhancedReasonText = 'Enhanced RSI';
                if (enhancedReason) {
                    if (enhancedReason.includes('insufficient_confirmation')) {
                        enhancedReasonText = 'Enhanced RSI: недостаточно подтверждений (нужно 2, если долго в зоне)';
                    } else if (enhancedReason.includes('strict_mode_no_divergence')) {
                        enhancedReasonText = 'Enhanced RSI: строгий режим - требуется дивергенция';
                    } else if (enhancedReason.includes('strict_mode')) {
                        enhancedReasonText = 'Enhanced RSI: строгий режим (требуется дивергенция)';
                    } else if (enhancedReason.includes('duration')) {
                        enhancedReasonText = 'Enhanced RSI: слишком долго в экстремальной зоне (нужно больше подтверждений)';
                    } else if (enhancedReason.includes('neutral') || enhancedReason.includes('enhanced_neutral')) {
                        enhancedReasonText = `Enhanced RSI: RSI ${rsi.toFixed(1)} не попадает в adaptive уровень`;
                    } else {
                        enhancedReasonText = `Enhanced RSI (${enhancedReason})`;
                    }
                } else {
                    enhancedReasonText = `Enhanced RSI: RSI ${rsi.toFixed(1)} заблокирован`;
                }
                blockReasons.push(enhancedReasonText);
            }
            
            // 6. Проверяем фильтры трендов (только если Enhanced RSI НЕ заблокировал)
            const enhancedRsiBlocked = enhancedRsiEnabled && enhancedSignal === 'WAIT' && baseSignal !== 'WAIT';
            if (!enhancedRsiBlocked) {
                const avoidDownTrend = autoConfig.avoid_down_trend === true;
                const avoidUpTrend = autoConfig.avoid_up_trend === true;
                
                if (baseSignal === 'ENTER_LONG' && avoidDownTrend && rsi <= rsiLongThreshold && trend === 'DOWN') {
                    blockReasons.push('Фильтр DOWN тренда');
                }
                if (baseSignal === 'ENTER_SHORT' && avoidUpTrend && rsi >= rsiShortThreshold && trend === 'UP') {
                    blockReasons.push('Фильтр UP тренда');
                }
            }
            
            return {
                reasons: blockReasons,
                enhancedRsiEnabled: enhancedRsiEnabled,
                enhancedSignal: enhancedSignal
            };
        };
        // Сводка причин блокировки сигнала
        const effectiveSignal = coin.effective_signal || this.getEffectiveSignal(coin);
        const baseSignal = coin.signal || 'WAIT';
        
        if (effectiveSignal === 'WAIT' && baseSignal !== 'WAIT') {
            // Сигнал был заблокирован - проверяем ВСЕ фильтры
            const filterCheck = checkAllBlockingFilters(coin);
            
            if (filterCheck.reasons.length > 0) {
                activeStatusData.signal_block_reason = `Базовый сигнал ${baseSignal} заблокирован: ${filterCheck.reasons.join(', ')}`;
            } else if (coin.signal_block_reason) {
                activeStatusData.signal_block_reason = coin.signal_block_reason;
            } else {
                activeStatusData.signal_block_reason = `Базовый сигнал ${baseSignal} изменен на WAIT (причина не определена)`;
            }
        } else if (effectiveSignal === 'WAIT' && baseSignal === 'WAIT') {
            // Базовый сигнал уже WAIT - проверяем ВСЕ фильтры
            const filterCheck = checkAllBlockingFilters(coin);
            const autoConfig = this.cachedAutoBotConfig || {};
            // Получаем RSI с учетом текущего таймфрейма
            const currentTimeframe = this.currentTimeframe || document.getElementById('systemTimeframe')?.value || '6h';
            const rsiKey = `rsi${currentTimeframe}`;
            const rsi = coin[rsiKey] || coin.rsi6h || coin.rsi || 50;
            const rsiLongThreshold = autoConfig.rsi_long_threshold || 29;
            const rsiShortThreshold = autoConfig.rsi_short_threshold || 71;
            
            // Формируем сообщение на основе результатов проверки фильтров
            let reasonText = '';
            
            if (rsi <= rsiLongThreshold) {
                // RSI низкий, но сигнал WAIT
                if (filterCheck.enhancedRsiEnabled && filterCheck.enhancedSignal === 'WAIT') {
                    reasonText = `RSI ${rsi.toFixed(1)} ≤ ${rsiLongThreshold}, но Enhanced RSI вернул WAIT`;
                } else if (filterCheck.enhancedRsiEnabled && filterCheck.enhancedSignal === 'ENTER_LONG') {
                    // Enhanced RSI разрешил LONG, но другие фильтры блокируют
                    if (filterCheck.reasons.length > 0) {
                        reasonText = `RSI ${rsi.toFixed(1)} ≤ ${rsiLongThreshold}, Enhanced RSI разрешил LONG, но заблокировано: ${filterCheck.reasons.join(', ')}`;
                    } else {
                        reasonText = `RSI ${rsi.toFixed(1)} ≤ ${rsiLongThreshold}, Enhanced RSI разрешил LONG, но сигнал WAIT`;
                    }
                } else {
                    // Другие причины блокировки
                    if (filterCheck.reasons.length > 0) {
                        reasonText = `RSI ${rsi.toFixed(1)} ≤ ${rsiLongThreshold}, но заблокировано: ${filterCheck.reasons.join(', ')}`;
                    } else {
                        reasonText = `RSI ${rsi.toFixed(1)} ≤ ${rsiLongThreshold}, но сигнал WAIT`;
                    }
                }
            } else if (rsi >= rsiShortThreshold) {
                // RSI высокий, но сигнал WAIT
                if (filterCheck.enhancedRsiEnabled && filterCheck.enhancedSignal === 'WAIT') {
                    reasonText = `RSI ${rsi.toFixed(1)} ≥ ${rsiShortThreshold}, но Enhanced RSI вернул WAIT`;
                } else if (filterCheck.enhancedRsiEnabled && filterCheck.enhancedSignal === 'ENTER_SHORT') {
                    // Enhanced RSI разрешил SHORT, но другие фильтры блокируют
                    if (filterCheck.reasons.length > 0) {
                        reasonText = `RSI ${rsi.toFixed(1)} ≥ ${rsiShortThreshold}, Enhanced RSI разрешил SHORT, но заблокировано: ${filterCheck.reasons.join(', ')}`;
                    } else {
                        reasonText = `RSI ${rsi.toFixed(1)} ≥ ${rsiShortThreshold}, Enhanced RSI разрешил SHORT, но сигнал WAIT`;
                    }
                } else {
                    // Другие причины блокировки
                    if (filterCheck.reasons.length > 0) {
                        reasonText = `RSI ${rsi.toFixed(1)} ≥ ${rsiShortThreshold}, но заблокировано: ${filterCheck.reasons.join(', ')}`;
                    } else {
                        reasonText = `RSI ${rsi.toFixed(1)} ≥ ${rsiShortThreshold}, но сигнал WAIT`;
                    }
                }
            } else {
                // RSI в нейтральной зоне
                if (filterCheck.reasons.length > 0) {
                    reasonText = `RSI ${rsi.toFixed(1)} в нейтральной зоне, заблокировано: ${filterCheck.reasons.join(', ')}`;
                }
            }
            
            if (reasonText) {
                activeStatusData.signal_block_reason = reasonText;
            }
        }
        
        // Enhanced RSI Warning (если есть, но не включена система)
        if (coin.enhanced_rsi?.warning_type && coin.enhanced_rsi.warning_type !== 'ERROR' && !coin.enhanced_rsi.enabled) {
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
            activeStatusData.maturity = window.languageUtils.getTranslation('mature_coin_description', {candles: actualCandles, min: minCandles});
        } else if (coin.is_mature === false) {
            const minCandles = this.autoBotConfig?.min_candles_for_maturity || 400;
            activeStatusData.maturity = window.languageUtils.getTranslation('immature_coin_description', {min: minCandles});
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
        
        this.updateFilterItem('enhancedRsiItem', 'selectedCoinEnhancedRsi', 'enhancedRsiIcon', 
                             activeStatusData.enhanced_rsi, 'Enhanced RSI');
        
        this.updateFilterItem('signalBlockReasonItem', 'selectedCoinSignalBlockReason', 'signalBlockReasonIcon', 
                             activeStatusData.signal_block_reason, 'Причина блокировки');
        
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
                    const blocksLabel = window.languageUtils.translate('blocks_label');
                    const safeLabel = window.languageUtils.translate('safe_label');
                    if (statusValue.includes(blocksLabel) || statusValue.toLowerCase().includes('block')) {
                        valueElement.innerHTML = `<span style="color: var(--red-text);">${statusValue}</span>`;
                        return; // Выходим рано для цветного контента
                    } else if (statusValue.includes(safeLabel) || statusValue.toLowerCase().includes('safe')) {
                        valueElement.innerHTML = `<span style="color: var(--green-text);">${statusValue}</span>`;
                        return; // Выходим рано для цветного контента
                    }
                    
                    if (statusValue.includes('SAFE')) { icon = '🛡️'; description = 'ExitScam: Безопасно'; }
                    else if (statusValue.includes('RISK')) { icon = '⚠️'; description = 'ExitScam: Риск обнаружен'; }
                    else if (statusValue.includes('SCAM')) { icon = '🚨'; description = 'ExitScam: Возможный скам'; }
                    else if (statusValue.includes('CHECKING')) { icon = '🔍'; description = 'ExitScam: Проверка'; }
                }
                else if (label === 'RSI Time Filter') {
                    // Убираем префикс статуса из текста для отображения
                    let displayText = statusValue;
                    if (statusValue.includes('ALLOWED:')) {
                        icon = '✅';
                        displayText = statusValue.replace('ALLOWED:', '').trim();
                        description = 'RSI Time Filter разрешен';
                    } else if (statusValue.includes('WAITING:')) {
                        icon = '⏳';
                        displayText = statusValue.replace('WAITING:', '').trim();
                        description = 'RSI Time Filter ожидание';
                    } else if (statusValue.includes('BLOCKED:')) {
                        icon = '❌';
                        displayText = statusValue.replace('BLOCKED:', '').trim();
                        description = 'RSI Time Filter заблокирован';
                    } else if (statusValue.includes('TIMEOUT')) {
                        icon = '⏰';
                        description = 'RSI Time Filter таймаут';
                    } else {
                        icon = '⏰';
                        description = statusValue || 'RSI Time Filter';
                    }
                    // Обновляем текст значения без префикса
                }
                else if (label === 'Enhanced RSI') {
                    // Специальная обработка для Enhanced RSI
                    let displayText = statusValue;
                    if (statusValue.includes('Блокировка:') || statusValue.includes('заблокирован')) {
                        icon = '🚫';
                        description = 'Enhanced RSI заблокировал сигнал';
                        valueElement.innerHTML = `<span style="color: var(--red-text);">${displayText}</span>`;
                        iconElement.textContent = icon;
                        iconElement.title = description;
                        return; // Выходим рано для цветного контента
                    } else if (statusValue.includes('Сигнал изменен:')) {
                        icon = '🔄';
                        description = 'Enhanced RSI изменил сигнал';
                        valueElement.innerHTML = `<span style="color: var(--warning-color);">${displayText}</span>`;
                        iconElement.textContent = icon;
                        iconElement.title = description;
                        return; // Выходим рано для цветного контента
                    } else if (statusValue.includes('Сигнал:')) {
                        icon = '🧠';
                        description = 'Enhanced RSI сигнал';
                        valueElement.textContent = displayText;
                    } else {
                        icon = '🧠';
                        description = 'Enhanced RSI';
                        valueElement.textContent = displayText;
                    }
                }
                else if (label === 'Причина блокировки') {
                    // Специальная обработка для причины блокировки сигнала
                    let displayText = statusValue;
                    icon = '🚫';
                    description = 'Причина блокировки сигнала';
                    valueElement.innerHTML = `<span style="color: var(--red-text); font-weight: bold;">${displayText}</span>`;
                    iconElement.textContent = icon;
                    iconElement.title = description;
                    return; // Выходим рано для цветного контента
                }
                else if (label === 'Статус бота') {
                    // Устанавливаем цвет для статуса бота в зависимости от значения
                    if (statusValue === window.languageUtils.translate('active_status') || 
                        statusValue.includes('running') || 
                        statusValue.includes('active') ||
                        statusValue === 'Активен') {
                        valueElement.style.color = 'var(--green-color)';
                        valueElement.classList.add('active-status');
                    } else if (statusValue.includes('waiting') || statusValue.includes('idle')) {
                        valueElement.style.color = 'var(--blue-color)';
                    } else if (statusValue.includes('error') || statusValue.includes('stopped')) {
                        valueElement.style.color = 'var(--red-color)';
                    } else if (statusValue.includes('paused')) {
                        valueElement.style.color = 'var(--warning-color)';
                    } else {
                        valueElement.style.color = 'var(--text-color)';
                    }
                    
                    if (statusValue === 'Нет бота' || statusValue === window.languageUtils.translate('bot_not_created')) { 
                        icon = '❓'; 
                        description = 'Бот не создан';
                        valueElement.style.color = 'var(--text-muted, var(--text-color))';
                        
                        const manualButtons = document.getElementById('manualBotButtons');
                        const longBtn = document.getElementById('enableBotLongBtn');
                        const shortBtn = document.getElementById('enableBotShortBtn');
                        if (manualButtons && longBtn && shortBtn) {
                            manualButtons.style.display = 'inline-flex';
                            longBtn.style.display = 'inline-block';
                            shortBtn.style.display = 'inline-block';
                        }
                    }
                    else if (statusValue.includes('running') || statusValue === window.languageUtils.translate('active_status') || statusValue === 'Активен') { 
                        icon = '🟢'; 
                        description = window.languageUtils.translate('bot_active_and_working');
                        valueElement.style.color = 'var(--green-color)';
                        // Скрываем кнопку для активных ботов
                        const manualButtons = document.getElementById('manualBotButtons');
                        if (manualButtons) manualButtons.style.display = 'none';
                    }
                    else if (statusValue.includes('waiting') || statusValue.includes('running') || statusValue.includes('idle')) { 
                        icon = '🔵'; 
                        description = window.languageUtils.translate('entry_by_market');
                        valueElement.style.color = 'var(--blue-color)';
                    }
                    else if (statusValue.includes('error')) { 
                        icon = '🔴'; 
                        description = window.languageUtils.translate('error_in_work');
                        valueElement.style.color = 'var(--red-color)';
                    }
                    else if (statusValue.includes('stopped')) { 
                        icon = '🔴'; 
                        description = window.languageUtils.translate('bot_stopped_desc');
                        valueElement.style.color = 'var(--red-color)';
                    }
                    else if (statusValue.includes('in_position')) { 
                        icon = '🟣'; 
                        description = window.languageUtils.translate('in_position_desc');
                        valueElement.style.color = 'var(--green-color)';
                    }
                    else if (statusValue.includes('paused')) { 
                        icon = '⚪'; 
                        description = window.languageUtils.translate('paused_status');
                        valueElement.style.color = 'var(--warning-color)';
                    }
                }
                
                iconElement.textContent = icon;
                iconElement.title = `${label}: ${description || statusValue}`;
                valueElement.title = `${label}: ${description || statusValue}`;
            } else {
                // Если нет статуса - скрываем элемент
                itemElement.style.display = 'none';
            }
        } else {
            // Если элементы не найдены - логируем для отладки
            if (label === 'RSI Time Filter') {
                console.warn(`[RSI_TIME_FILTER] Элементы не найдены для ${label}:`, {itemId, valueId, iconId, statusValue});
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
                value: window.languageUtils.getTranslation('mature_coin_description', {candles: actualCandles, min: minCandles}),
                icon: '',
                description: 'Монета имеет достаточно истории для надежного анализа'
            });
        } else if (coin.is_mature === false) {
            const minCandles = this.autoBotConfig?.min_candles_for_maturity || 400;
            realFilters.push({
                itemId: 'maturityDiamondItem',
                valueId: 'selectedCoinMaturityDiamond',
                iconId: 'maturityDiamondIcon',
                value: window.languageUtils.getTranslation('immature_coin_description', {min: minCandles}),
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
                        stochDescription = window.languageUtils.translate('stochastic_oversold').replace('{k}', stochK.toFixed(1));
                    } else if (stochK > 80) {
                        stochIcon = '⬆️';
                        stochStatus = 'OVERBOUGHT';
                        stochDescription = window.languageUtils.translate('stochastic_overbought').replace('{k}', stochK.toFixed(1));
                    } else {
                        stochIcon = '➡️';
                        stochStatus = 'NEUTRAL';
                        stochDescription = window.languageUtils.translate('stochastic_neutral').replace('{k}', stochK.toFixed(1));
                    }
                    
                    // Добавляем информацию о пересечении
                    let crossoverInfo = '';
                    if (stochK > stochD) {
                        crossoverInfo = ' ' + window.languageUtils.translate('stochastic_bullish_signal').replace('{d}', stochD.toFixed(1));
                    } else if (stochK < stochD) {
                        crossoverInfo = ' ' + window.languageUtils.translate('stochastic_bearish_signal').replace('{d}', stochD.toFixed(1));
                    } else {
                        crossoverInfo = ' (%K = %D - ' + (window.languageUtils.translate('neutral') || 'нейтрально') + ')';
                    }
                    
                    const fullDescription = `Stochastic RSI: ${stochDescription}${crossoverInfo}`;
                    
                    // Создаем подробное описание для отображения на странице
                    let detailedValue = '';
                    
                    // Определяем сигнал пересечения с цветами
                    let signalInfo = '';
                    if (stochK > stochD) {
                        signalInfo = `<span style="color: var(--green-text);">${window.languageUtils.getTranslation('stochastic_bullish_signal', {d: stochD.toFixed(1)})}</span>`;
                    } else if (stochK < stochD) {
                        signalInfo = `<span style="color: var(--red-text);">${window.languageUtils.getTranslation('stochastic_bearish_signal', {d: stochD.toFixed(1)})}</span>`;
                    } else {
                        signalInfo = `<span style="color: var(--warning-color);">Нейтральный сигнал: %D=${stochD.toFixed(1)} (%K = %D)</span>`;
                    }
                    
                    if (stochStatus === 'OVERSOLD') {
                        detailedValue = `<span style="color: var(--green-text);">${window.languageUtils.getTranslation('stochastic_oversold', {k: stochK.toFixed(1)})}</span><br>${signalInfo}`;
                    } else if (stochStatus === 'OVERBOUGHT') {
                        detailedValue = `<span style="color: var(--red-text);">${window.languageUtils.getTranslation('stochastic_overbought', {k: stochK.toFixed(1)})}</span><br>${signalInfo}`;
                    } else {
                        detailedValue = `<span style="color: var(--warning-color);">${window.languageUtils.getTranslation('stochastic_neutral', {k: stochK.toFixed(1)})}</span><br>${signalInfo}`;
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
                value: isBlocked ? window.languageUtils.translate('rsi_time_filter_blocked').replace('{reason}', reason) : window.languageUtils.translate('rsi_time_filter_allowed').replace('{reason}', reason),
                icon: isBlocked ? '⏰' : '⏱️',
                        description: `RSI Time Filter: ${reason}${calmCandles > 0 ? ` (${calmCandles} ${window.languageUtils.translate('calm_candles') || 'calm candles'})` : ''}`
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
                coloredValue = `<span style="color: var(--red-text);">${window.languageUtils.translate('blocks_label')} ${reason}</span>`;
            } else {
                coloredValue = `<span style="color: var(--green-text);">${window.languageUtils.translate('safe_label')} ${reason}</span>`;
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
        
        // 6. Защита от повторных входов после убыточных закрытий
        if (coin.loss_reentry_info) {
            const lossReentry = coin.loss_reentry_info;
            const isBlocked = lossReentry.blocked;
            const reason = lossReentry.reason || '';
            const candlesPassed = lossReentry.candles_passed;
            const requiredCandles = lossReentry.required_candles;
            const lossCount = lossReentry.loss_count;
            
            // Добавляем цветовое кодирование
            let coloredValue = '';
            let icon = '';
            if (isBlocked) {
                coloredValue = `<span style="color: var(--red-text);">${window.languageUtils.translate('loss_reentry_blocked') || 'Блокирует'}: ${reason}</span>`;
                icon = '🚫';
            } else {
                coloredValue = `<span style="color: var(--green-text);">${window.languageUtils.translate('loss_reentry_allowed') || 'Разрешено'}: ${reason}</span>`;
                icon = '✅';
            }
            
            // Формируем описание с деталями
            let description = `${window.languageUtils.translate('loss_reentry_protection_label') || 'Защита от повторных входов'}: ${reason}`;
            if (candlesPassed !== undefined && requiredCandles !== undefined) {
                description += ` (прошло ${candlesPassed}/${requiredCandles} свечей)`;
            }
            if (lossCount !== undefined) {
                description += ` [N=${lossCount}]`;
            }
            
            realFilters.push({
                itemId: 'lossReentryItem',
                valueId: 'selectedCoinLossReentry',
                iconId: 'lossReentryIcon',
                value: coloredValue,
                icon: icon,
                description: description
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
                case 'delisted':
                    visible = item.classList.contains('delisting-coin');
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
    }
    
    collectDuplicateSettings() {
        console.log('[BotsManager] 📋 Сбор дублированных настроек...');
        
        const settings = {};
        
        // RSI настройки
        const rsiLongEl = document.getElementById('rsiLongThresholdDup');
        if (rsiLongEl && rsiLongEl.value) settings.rsi_long_threshold = parseInt(rsiLongEl.value);
        
        const rsiShortEl = document.getElementById('rsiShortThresholdDup');
        if (rsiShortEl && rsiShortEl.value) settings.rsi_short_threshold = parseInt(rsiShortEl.value);
        
        // ✅ Новые параметры RSI выхода с учетом тренда
        const rsiExitLongWithTrendEl = document.getElementById('rsiExitLongWithTrendDup');
        if (rsiExitLongWithTrendEl && rsiExitLongWithTrendEl.value) {
            settings.rsi_exit_long_with_trend = parseInt(rsiExitLongWithTrendEl.value);
        }
        
        const rsiExitLongAgainstTrendEl = document.getElementById('rsiExitLongAgainstTrendDup');
        if (rsiExitLongAgainstTrendEl && rsiExitLongAgainstTrendEl.value) {
            settings.rsi_exit_long_against_trend = parseInt(rsiExitLongAgainstTrendEl.value);
        }
        
        const rsiExitShortWithTrendEl = document.getElementById('rsiExitShortWithTrendDup');
        if (rsiExitShortWithTrendEl && rsiExitShortWithTrendEl.value) {
            settings.rsi_exit_short_with_trend = parseInt(rsiExitShortWithTrendEl.value);
        }
        
        const rsiExitShortAgainstTrendEl = document.getElementById('rsiExitShortAgainstTrendDup');
        if (rsiExitShortAgainstTrendEl && rsiExitShortAgainstTrendEl.value) {
            settings.rsi_exit_short_against_trend = parseInt(rsiExitShortAgainstTrendEl.value);
        }
        
        // Защитные механизмы
        const maxLossEl = document.getElementById('maxLossPercentDup');
        if (maxLossEl && maxLossEl.value) settings.max_loss_percent = parseFloat(maxLossEl.value);
        
        const takeProfitEl = document.getElementById('takeProfitPercentDup');
        if (takeProfitEl && takeProfitEl.value !== '') settings.take_profit_percent = parseFloat(takeProfitEl.value);
        const closeAtProfitEl = document.getElementById('closeAtProfitEnabledDup');
        if (closeAtProfitEl) settings.close_at_profit_enabled = closeAtProfitEl.checked;
        
        const trailingActivationEl = document.getElementById('trailingStopActivationDup');
        if (trailingActivationEl && trailingActivationEl.value) settings.trailing_stop_activation = parseFloat(trailingActivationEl.value);
        
        const trailingDistanceEl = document.getElementById('trailingStopDistanceDup');
        if (trailingDistanceEl && trailingDistanceEl.value) settings.trailing_stop_distance = parseFloat(trailingDistanceEl.value);

        const trailingTakeEl = document.getElementById('trailingTakeDistanceDup');
        if (trailingTakeEl && trailingTakeEl.value) settings.trailing_take_distance = parseFloat(trailingTakeEl.value);

        const trailingIntervalEl = document.getElementById('trailingUpdateIntervalDup');
        if (trailingIntervalEl && trailingIntervalEl.value) settings.trailing_update_interval = parseFloat(trailingIntervalEl.value);
        
        const maxHoursEl = document.getElementById('maxPositionHoursDup');
        if (maxHoursEl) {
            const seconds = parseInt(maxHoursEl.value) || 0;
            // В конфиге хранятся часы; передаём часы (секунды / 3600)
            settings.max_position_hours = seconds / 3600;
        }
        
        const breakEvenEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenEl) settings.break_even_protection = breakEvenEl.checked;
        
        const breakEvenTriggerEl = document.getElementById('breakEvenTriggerDup');
        if (breakEvenTriggerEl && breakEvenTriggerEl.value) {
            const triggerValue = parseFloat(breakEvenTriggerEl.value);
            settings.break_even_trigger = triggerValue;
            settings.break_even_trigger_percent = triggerValue;
        }

        const avoidDownTrendEl = document.getElementById('avoidDownTrendDup');
        if (avoidDownTrendEl) settings.avoid_down_trend = avoidDownTrendEl.checked;

        const avoidUpTrendEl = document.getElementById('avoidUpTrendDup');
        if (avoidUpTrendEl) settings.avoid_up_trend = avoidUpTrendEl.checked;

        const lossReentryProtectionEl = document.getElementById('lossReentryProtection');
        if (lossReentryProtectionEl) settings.loss_reentry_protection = lossReentryProtectionEl.checked;

        const lossReentryCountEl = document.getElementById('lossReentryCount');
        if (lossReentryCountEl && lossReentryCountEl.value) {
            settings.loss_reentry_count = parseInt(lossReentryCountEl.value);
        }

        const lossReentryCandlesEl = document.getElementById('lossReentryCandles');
        if (lossReentryCandlesEl && lossReentryCandlesEl.value) {
            settings.loss_reentry_candles = parseInt(lossReentryCandlesEl.value);
        }

        const maturityCheckEl = document.getElementById('enableMaturityCheckDup');
        if (maturityCheckEl) settings.enable_maturity_check = maturityCheckEl.checked;

        const minCandlesMaturityEl = document.getElementById('minCandlesForMaturityDup');
        if (minCandlesMaturityEl && minCandlesMaturityEl.value) {
            settings.min_candles_for_maturity = parseInt(minCandlesMaturityEl.value);
        }

        const minRsiLowEl = document.getElementById('minRsiLowDup');
        if (minRsiLowEl && minRsiLowEl.value) {
            settings.min_rsi_low = parseFloat(minRsiLowEl.value);
        }

        const maxRsiHighEl = document.getElementById('maxRsiHighDup');
        if (maxRsiHighEl && maxRsiHighEl.value) {
            settings.max_rsi_high = parseFloat(maxRsiHighEl.value);
        }

        const rsiTimeFilterEnabledEl = document.getElementById('rsiTimeFilterEnabledDup');
        if (rsiTimeFilterEnabledEl) settings.rsi_time_filter_enabled = rsiTimeFilterEnabledEl.checked;

        const rsiTimeFilterCandlesEl = document.getElementById('rsiTimeFilterCandlesDup');
        if (rsiTimeFilterCandlesEl && rsiTimeFilterCandlesEl.value) {
            const candles = parseInt(rsiTimeFilterCandlesEl.value);
            settings.rsi_time_filter_candles = candles;
        }

        const rsiTimeFilterUpperEl = document.getElementById('rsiTimeFilterUpperDup');
        if (rsiTimeFilterUpperEl && rsiTimeFilterUpperEl.value) {
            settings.rsi_time_filter_upper = parseFloat(rsiTimeFilterUpperEl.value);
        }

        const rsiTimeFilterLowerEl = document.getElementById('rsiTimeFilterLowerDup');
        if (rsiTimeFilterLowerEl && rsiTimeFilterLowerEl.value) {
            settings.rsi_time_filter_lower = parseFloat(rsiTimeFilterLowerEl.value);
        }

        const exitScamEnabledEl = document.getElementById('exitScamEnabledDup');
        if (exitScamEnabledEl) settings.exit_scam_enabled = exitScamEnabledEl.checked;

        const exitScamCandlesEl = document.getElementById('exitScamCandlesDup');
        if (exitScamCandlesEl && exitScamCandlesEl.value) {
            settings.exit_scam_candles = parseInt(exitScamCandlesEl.value);
        }

        const exitScamSingleEl = document.getElementById('exitScamSingleCandleDup');
        if (exitScamSingleEl && exitScamSingleEl.value) {
            settings.exit_scam_single_candle_percent = parseFloat(exitScamSingleEl.value);
        }

        const exitScamMultiCountEl = document.getElementById('exitScamMultiCountDup');
        if (exitScamMultiCountEl && exitScamMultiCountEl.value) {
            settings.exit_scam_multi_candle_count = parseInt(exitScamMultiCountEl.value);
        }

        const exitScamMultiPercentEl = document.getElementById('exitScamMultiPercentDup');
        if (exitScamMultiPercentEl && exitScamMultiPercentEl.value) {
            settings.exit_scam_multi_candle_percent = parseFloat(exitScamMultiPercentEl.value);
        }

        const trendDetectionEnabledEl = document.getElementById('trendDetectionEnabledDup');
        if (trendDetectionEnabledEl) settings.trend_detection_enabled = trendDetectionEnabledEl.checked;

        const trendAnalysisPeriodEl = document.getElementById('trendAnalysisPeriodDup');
        if (trendAnalysisPeriodEl && trendAnalysisPeriodEl.value) {
            settings.trend_analysis_period = parseInt(trendAnalysisPeriodEl.value);
        }

        const trendPriceChangeEl = document.getElementById('trendPriceChangeThresholdDup');
        if (trendPriceChangeEl && trendPriceChangeEl.value) {
            settings.trend_price_change_threshold = parseFloat(trendPriceChangeEl.value);
        }

        const trendCandlesThresholdEl = document.getElementById('trendCandlesThresholdDup');
        if (trendCandlesThresholdEl && trendCandlesThresholdEl.value) {
            settings.trend_candles_threshold = parseInt(trendCandlesThresholdEl.value);
        }
        
        // ✅ Enhanced RSI настройки для индивидуальных настроек монеты
        const enhancedRsiEnabledDupEl = document.getElementById('enhancedRsiEnabledDup');
        if (enhancedRsiEnabledDupEl) {
            settings.enhanced_rsi_enabled = enhancedRsiEnabledDupEl.checked;
        }
        
        const enhancedRsiVolumeConfirmDupEl = document.getElementById('enhancedRsiVolumeConfirmDup');
        if (enhancedRsiVolumeConfirmDupEl) {
            settings.enhanced_rsi_require_volume_confirmation = enhancedRsiVolumeConfirmDupEl.checked;
        }
        
        const enhancedRsiDivergenceConfirmDupEl = document.getElementById('enhancedRsiDivergenceConfirmDup');
        if (enhancedRsiDivergenceConfirmDupEl) {
            settings.enhanced_rsi_require_divergence_confirmation = enhancedRsiDivergenceConfirmDupEl.checked;
        }
        
        const enhancedRsiUseStochRsiDupEl = document.getElementById('enhancedRsiUseStochRsiDup');
        if (enhancedRsiUseStochRsiDupEl) {
            settings.enhanced_rsi_use_stoch_rsi = enhancedRsiUseStochRsiDupEl.checked;
        }
        
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

    /**
     * Подбор параметров ExitScam по истории свечей для выбранной монеты.
     * Результат сохраняется в индивидуальные настройки монеты и подставляется в форму.
     */
    async learnExitScamForCoin() {
        if (!this.selectedCoin || !this.selectedCoin.symbol) {
            this.showNotification('⚠️ Выберите монету для подбора ExitScam', 'warning');
            return;
        }
        const symbol = this.selectedCoin.symbol;
        const btn = document.getElementById('learnExitScamForCoinBtn');
        const originalText = btn ? btn.innerHTML : '';
        try {
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span>⏳ Анализ свечей...</span>';
            }
            this.showNotification(`🧠 Анализ свечей ${symbol}...`, 'info');
            const exitScamTfEl = document.getElementById('exitScamTimeframe');
            const currentTf = exitScamTfEl?.value || this.cachedAutoBotConfig?.exit_scam_timeframe || '6h';
            const response = await fetch(
                `${this.BOTS_SERVICE_URL}/api/bots/individual-settings/${encodeURIComponent(symbol)}/learn-exit-scam`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ aggressiveness: 'normal', timeframe: currentTf })
                }
            );
            const data = await response.json();
            if (data.success && data.params) {
                await this.loadAndApplyIndividualSettings(symbol);
                this.updateIndividualSettingsStatus(true);
                const p = data.params;
                this.showNotification(
                    `✅ ExitScam для ${symbol}: 1 св ${p.exit_scam_single_candle_percent}%, ${p.exit_scam_multi_candle_count} св ${p.exit_scam_multi_candle_percent}%`,
                    'success'
                );
            } else {
                const err = data.error || 'Не удалось подобрать параметры';
                this.showNotification(`❌ ExitScam: ${err}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка learn-exit-scam:', error);
            this.showNotification('❌ Ошибка соединения при подборе ExitScam', 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = originalText;
            }
        }
    }

    /**
     * Расчёт ExitScam по истории для всех монет (ручной запуск). Использует текущий ТФ из UI.
     */
    async learnExitScamForAllCoins() {
        const btn = document.getElementById('learnExitScamForAllCoinsBtn');
        const originalText = btn ? btn.innerHTML : '';
        try {
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span>⏳ Расчёт для всех монет...</span>';
            }
            this.showNotification('🧠 Расчёт индивидуального ExitScam для всех монет...', 'info');
            const exitScamTfEl = document.getElementById('exitScamTimeframe');
            const currentTf = exitScamTfEl?.value || this.cachedAutoBotConfig?.exit_scam_timeframe || '6h';
            const response = await fetch(
                `${this.BOTS_SERVICE_URL}/api/bots/individual-settings/learn-exit-scam-all`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ aggressiveness: 'normal', timeframe: currentTf })
                }
            );
            const data = await response.json();
            if (data.success) {
                const u = data.updated_count || 0;
                const f = data.failed_count || 0;
                const sample = (data.sample_params || []).slice(0, 5);
                const sampleStr = sample.length
                    ? sample.map(s => `${s.symbol} ${s.exit_scam_single_candle_percent}%/${s.exit_scam_multi_candle_count}св ${s.exit_scam_multi_candle_percent}%`).join(', ')
                    : '';
                const msg = sampleStr
                    ? `✅ Обновлено ${u} монет (ошибок: ${f}). Примеры: ${sampleStr}. Выберите монету и загрузите настройки — значения индивидуальны.`
                    : `✅ Индивидуальный ExitScam для всех: обновлено ${u} монет, без данных/ошибок: ${f}`;
                this.showNotification(msg, 'success');
            } else {
                this.showNotification(`❌ ${data.error || 'Ошибка расчёта'}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка learn-exit-scam-all:', error);
            this.showNotification('❌ Ошибка соединения при расчёте для всех', 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = originalText;
            }
        }
    }

    /**
     * Сброс индивидуальных настроек ExitScam для всех монет — будут использоваться значения из конфига.
     */
    async resetExitScamToConfigForAll() {
        const btn = document.getElementById('resetExitScamToConfigForAllBtn');
        const originalText = btn ? btn.innerHTML : '';
        try {
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<span>⏳ Сброс...</span>';
            }
            this.showNotification('🔄 Сброс ExitScam к общим настройкам для всех монет...', 'info');
            const response = await fetch(
                `${this.BOTS_SERVICE_URL}/api/bots/individual-settings/reset-exit-scam-all`,
                { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }
            );
            const data = await response.json();
            if (data.success) {
                const n = data.reset_count || 0;
                this.showNotification(
                    n > 0 ? `✅ ExitScam сброшен к общим настройкам для ${n} монет` : '✅ Нет индивидуальных ExitScam — все уже используют конфиг',
                    'success'
                );
            } else {
                this.showNotification(`❌ ${data.error || 'Ошибка сброса'}`, 'error');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка reset-exit-scam-all:', error);
            this.showNotification('❌ Ошибка соединения при сбросе ExitScam', 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = originalText;
            }
        }
    }

    async resetAllCoinsToGlobalSettings() {
        try {
            const confirmed = confirm('⚠️ Вы уверены, что хотите сбросить индивидуальные настройки ВСЕХ монет к глобальным настройкам?\n\nЭто действие нельзя отменить!');
            if (!confirmed) {
                return false;
            }
            
            console.log('[BotsManager] 🔄 Сброс всех индивидуальных настроек к глобальным');
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/individual-settings/reset-all`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            if (data.success) {
                console.log(`[BotsManager] ✅ Сброшены индивидуальные настройки для ${data.removed_count} монет`);
                
                // Формируем красивое сообщение
                const coinWord = data.removed_count === 1 ? 'монеты' : 
                                data.removed_count >= 2 && data.removed_count <= 4 ? 'монет' : 'монет';
                const message = data.removed_count > 0 
                    ? `✅ Сброшены индивидуальные настройки для ${data.removed_count} ${coinWord}. Все монеты теперь используют глобальные настройки.`
                    : '✅ Индивидуальные настройки отсутствуют. Все монеты используют глобальные настройки.';
                
                this.showNotification(message, 'success');
                
                // Обновляем статус индивидуальных настроек, если выбрана монета
                if (this.selectedCoin) {
                    this.updateIndividualSettingsStatus(false);
                }
                
                return true;
            } else {
                console.error(`[BotsManager] ❌ Ошибка сброса настроек: ${data.error}`);
                this.showNotification(`❌ Ошибка сброса: ${data.error}`, 'error');
                return false;
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сброса всех индивидуальных настроек:', error);
            this.showNotification('❌ Ошибка соединения при сбросе настроек', 'error');
            return false;
        }
    }

    /**
     * Маппинг ключей конфига на ID элементов для подсветки отличий от основного конфига.
     * Только ключи, присутствующие в индивидуальных настройках и отличающиеся от main config, подсвечиваются.
     */
    getIndividualSettingsElementMap() {
        return {
            rsi_long_threshold: 'rsiLongThresholdDup',
            rsi_short_threshold: 'rsiShortThresholdDup',
            rsi_exit_long_with_trend: 'rsiExitLongWithTrendDup',
            rsi_exit_long_against_trend: 'rsiExitLongAgainstTrendDup',
            rsi_exit_short_with_trend: 'rsiExitShortWithTrendDup',
            rsi_exit_short_against_trend: 'rsiExitShortAgainstTrendDup',
            max_loss_percent: 'maxLossPercentDup',
            take_profit_percent: 'takeProfitPercentDup',
            close_at_profit_enabled: 'closeAtProfitEnabledDup',
            trailing_stop_activation: 'trailingStopActivationDup',
            trailing_stop_distance: 'trailingStopDistanceDup',
            trailing_take_distance: 'trailingTakeDistanceDup',
            trailing_update_interval: 'trailingUpdateIntervalDup',
            max_position_hours: 'maxPositionHoursDup',
            break_even_protection: 'breakEvenProtectionDup',
            break_even_trigger: 'breakEvenTriggerDup',
            break_even_trigger_percent: 'breakEvenTriggerDup',
            avoid_down_trend: 'avoidDownTrendDup',
            avoid_up_trend: 'avoidUpTrendDup',
            enable_maturity_check: 'enableMaturityCheckDup',
            min_candles_for_maturity: 'minCandlesForMaturityDup',
            min_rsi_low: 'minRsiLowDup',
            max_rsi_high: 'maxRsiHighDup',
            rsi_time_filter_enabled: 'rsiTimeFilterEnabledDup',
            rsi_time_filter_candles: 'rsiTimeFilterCandlesDup',
            rsi_time_filter_upper: 'rsiTimeFilterUpperDup',
            rsi_time_filter_lower: 'rsiTimeFilterLowerDup',
            exit_scam_enabled: 'exitScamEnabledDup',
            exit_scam_candles: 'exitScamCandlesDup',
            exit_scam_single_candle_percent: 'exitScamSingleCandleDup',
            exit_scam_multi_candle_count: 'exitScamMultiCountDup',
            exit_scam_multi_candle_percent: 'exitScamMultiPercentDup',
            trend_detection_enabled: 'trendDetectionEnabledDup',
            trend_analysis_period: 'trendAnalysisPeriodDup',
            trend_price_change_threshold: 'trendPriceChangeThresholdDup',
            trend_candles_threshold: 'trendCandlesThresholdDup',
            volume_mode: 'volumeModeSelect',
            volume_value: 'volumeValueInput',
            leverage: 'leverageCoinInput',
            enhanced_rsi_enabled: 'enhancedRsiEnabledDup',
            enhanced_rsi_require_volume_confirmation: 'enhancedRsiVolumeConfirmDup',
            enhanced_rsi_require_divergence_confirmation: 'enhancedRsiDivergenceConfirmDup',
            enhanced_rsi_use_stoch_rsi: 'enhancedRsiUseStochRsiDup'
        };
    }

    clearIndividualSettingDiffHighlights() {
        document.querySelectorAll('.setting-item.individual-setting-diff').forEach(el => {
            el.classList.remove('individual-setting-diff');
        });
    }

    /**
     * Подсвечивает настройки, которые отличаются от основного конфига.
     * @param {Object} individualSettings - индивидуальные настройки монеты
     */
    highlightIndividualSettingDiffs(individualSettings) {
        this.clearIndividualSettingDiffHighlights();
        if (!individualSettings || typeof individualSettings !== 'object') return;

        const config = this.cachedAutoBotConfig || {};
        const fallback = {
            rsi_long_threshold: 29, rsi_short_threshold: 71,
            rsi_exit_long_with_trend: 65, rsi_exit_long_against_trend: 60,
            rsi_exit_short_with_trend: 35, rsi_exit_short_against_trend: 40,
            max_loss_percent: 15.0, take_profit_percent: 5.0, close_at_profit_enabled: true,
            trailing_stop_activation: 20.0, trailing_stop_distance: 5.0,
            trailing_take_distance: 0.5, trailing_update_interval: 3.0,
            max_position_hours: 0, break_even_protection: true,
            break_even_trigger: 20.0, break_even_trigger_percent: 20.0,
            avoid_down_trend: true, avoid_up_trend: true,
            enable_maturity_check: true, min_candles_for_maturity: 400,
            min_rsi_low: 35, max_rsi_high: 65,
            rsi_time_filter_enabled: true, rsi_time_filter_candles: 6,
            rsi_time_filter_upper: 65, rsi_time_filter_lower: 35,
            exit_scam_enabled: true, exit_scam_candles: 8,
            exit_scam_single_candle_percent: 15, exit_scam_multi_candle_count: 4,
            exit_scam_multi_candle_percent: 50, trend_detection_enabled: false,
            trend_analysis_period: 30, trend_price_change_threshold: 7,
            trend_candles_threshold: 70, volume_mode: 'usdt', volume_value: 10,
            leverage: 10, enhanced_rsi_enabled: false,
            enhanced_rsi_require_volume_confirmation: false,
            enhanced_rsi_require_divergence_confirmation: false,
            enhanced_rsi_use_stoch_rsi: false
        };

        const getMainValue = (key) => {
            const v = config[key];
            return v !== undefined ? v : fallback[key];
        };

        const valuesEqual = (a, b) => {
            if (a === b) return true;
            if (typeof a === 'boolean' || typeof b === 'boolean') return Boolean(a) === Boolean(b);
            const na = Number(a);
            const nb = Number(b);
            if (!Number.isNaN(na) && !Number.isNaN(nb)) return na === nb;
            return String(a) === String(b);
        };

        const elementMap = this.getIndividualSettingsElementMap();

        for (const [configKey, elementId] of Object.entries(elementMap)) {
            if (!(configKey in individualSettings)) continue;
            if (configKey === 'break_even_trigger' && 'break_even_trigger_percent' in individualSettings) continue;

            const indVal = individualSettings[configKey];
            let mainVal = getMainValue(configKey);
            if (configKey === 'break_even_trigger_percent') {
                mainVal = getMainValue('break_even_trigger') ?? getMainValue('break_even_trigger_percent');
            }

            if (!valuesEqual(indVal, mainVal)) {
                const el = document.getElementById(elementId);
                if (el) {
                    const parent = el.closest('.setting-item');
                    if (parent) parent.classList.add('individual-setting-diff');
                }
            }
        }
    }

    applyIndividualSettingsToUI(settings) {
        if (!settings) return;
        
        console.log('[BotsManager] 🎨 Применение индивидуальных настроек к UI:', settings);
        const fallbackConfig = this.cachedAutoBotConfig || {};
        const getSettingValue = (key) => {
            if (settings[key] !== undefined) return settings[key];
            return fallbackConfig[key];
        };
        
        // RSI настройки
        const rsiLongEl = document.getElementById('rsiLongThresholdDup');
        if (rsiLongEl && settings.rsi_long_threshold !== undefined) {
            rsiLongEl.value = settings.rsi_long_threshold;
        }
        
        const rsiShortEl = document.getElementById('rsiShortThresholdDup');
        if (rsiShortEl && settings.rsi_short_threshold !== undefined) {
            rsiShortEl.value = settings.rsi_short_threshold;
        }
        
        // ✅ Новые параметры RSI выхода с учетом тренда
        const rsiExitLongWithTrendEl = document.getElementById('rsiExitLongWithTrendDup');
        if (rsiExitLongWithTrendEl && settings.rsi_exit_long_with_trend !== undefined) {
            rsiExitLongWithTrendEl.value = settings.rsi_exit_long_with_trend;
        }
        
        const rsiExitLongAgainstTrendEl = document.getElementById('rsiExitLongAgainstTrendDup');
        if (rsiExitLongAgainstTrendEl && settings.rsi_exit_long_against_trend !== undefined) {
            rsiExitLongAgainstTrendEl.value = settings.rsi_exit_long_against_trend;
        }
        
        const rsiExitShortWithTrendEl = document.getElementById('rsiExitShortWithTrendDup');
        if (rsiExitShortWithTrendEl && settings.rsi_exit_short_with_trend !== undefined) {
            rsiExitShortWithTrendEl.value = settings.rsi_exit_short_with_trend;
        }
        
        const rsiExitShortAgainstTrendEl = document.getElementById('rsiExitShortAgainstTrendDup');
        if (rsiExitShortAgainstTrendEl && settings.rsi_exit_short_against_trend !== undefined) {
            rsiExitShortAgainstTrendEl.value = settings.rsi_exit_short_against_trend;
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
            // В конфиге часы; показываем в секундах
            maxHoursEl.value = Math.round((settings.max_position_hours || 0) * 3600);
        }
        
        const breakEvenEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenEl && settings.break_even_protection !== undefined) {
            breakEvenEl.checked = settings.break_even_protection;
        }
        
        const breakEvenTriggerEl = document.getElementById('breakEvenTriggerDup');
        const breakEvenTriggerValue = settings.break_even_trigger_percent ?? settings.break_even_trigger;
        if (breakEvenTriggerEl && breakEvenTriggerValue !== undefined) {
            breakEvenTriggerEl.value = breakEvenTriggerValue;
        }
        
        // Трендовые настройки
        const avoidDownTrendEl = document.getElementById('avoidDownTrendDup');
        if (avoidDownTrendEl) {
            const value = getSettingValue('avoid_down_trend');
            if (value !== undefined) {
                avoidDownTrendEl.checked = Boolean(value);
            }
        }
        
        const avoidUpTrendEl = document.getElementById('avoidUpTrendDup');
        if (avoidUpTrendEl) {
            const value = getSettingValue('avoid_up_trend');
            if (value !== undefined) {
                avoidUpTrendEl.checked = Boolean(value);
            }
        }
        
        const enableMaturityEl = document.getElementById('enableMaturityCheckDup');
        if (enableMaturityEl) {
            const value = getSettingValue('enable_maturity_check');
            if (value !== undefined) {
                enableMaturityEl.checked = Boolean(value);
            }
        }

        const minCandlesMaturityEl = document.getElementById('minCandlesForMaturityDup');
        if (minCandlesMaturityEl) {
            const value = getSettingValue('min_candles_for_maturity');
            if (value !== undefined) {
                minCandlesMaturityEl.value = value;
            }
        }

        const minRsiLowEl = document.getElementById('minRsiLowDup');
        if (minRsiLowEl) {
            const value = getSettingValue('min_rsi_low');
            if (value !== undefined) {
                minRsiLowEl.value = value;
            }
        }

        const maxRsiHighEl = document.getElementById('maxRsiHighDup');
        if (maxRsiHighEl) {
            const value = getSettingValue('max_rsi_high');
            if (value !== undefined) {
                maxRsiHighEl.value = value;
            }
        }

        const rsiTimeFilterEnabledEl = document.getElementById('rsiTimeFilterEnabledDup');
        if (rsiTimeFilterEnabledEl) {
            const value = getSettingValue('rsi_time_filter_enabled');
            if (value !== undefined) {
                rsiTimeFilterEnabledEl.checked = Boolean(value);
            }
        }

        const rsiTimeFilterCandlesEl = document.getElementById('rsiTimeFilterCandlesDup');
        if (rsiTimeFilterCandlesEl) {
            const value = getSettingValue('rsi_time_filter_candles');
            if (value !== undefined) {
                rsiTimeFilterCandlesEl.value = value;
            }
        }

        const rsiTimeFilterUpperEl = document.getElementById('rsiTimeFilterUpperDup');
        if (rsiTimeFilterUpperEl) {
            const value = getSettingValue('rsi_time_filter_upper');
            if (value !== undefined) {
                rsiTimeFilterUpperEl.value = value;
            }
        }

        const rsiTimeFilterLowerEl = document.getElementById('rsiTimeFilterLowerDup');
        if (rsiTimeFilterLowerEl) {
            const value = getSettingValue('rsi_time_filter_lower');
            if (value !== undefined) {
                rsiTimeFilterLowerEl.value = value;
            }
        }

        const exitScamEnabledEl = document.getElementById('exitScamEnabledDup');
        if (exitScamEnabledEl) {
            const value = getSettingValue('exit_scam_enabled');
            if (value !== undefined) {
                exitScamEnabledEl.checked = Boolean(value);
            }
        }

        const exitScamCandlesEl = document.getElementById('exitScamCandlesDup');
        if (exitScamCandlesEl) {
            const value = getSettingValue('exit_scam_candles');
            if (value !== undefined) {
                exitScamCandlesEl.value = value;
            }
        }

        const exitScamSingleEl = document.getElementById('exitScamSingleCandleDup');
        if (exitScamSingleEl) {
            const value = getSettingValue('exit_scam_single_candle_percent');
            if (value !== undefined) {
                exitScamSingleEl.value = value;
            }
        }

        const exitScamMultiCountEl = document.getElementById('exitScamMultiCountDup');
        if (exitScamMultiCountEl) {
            const value = getSettingValue('exit_scam_multi_candle_count');
            if (value !== undefined) {
                exitScamMultiCountEl.value = value;
            }
        }

        const exitScamMultiPercentEl = document.getElementById('exitScamMultiPercentDup');
        if (exitScamMultiPercentEl) {
            const value = getSettingValue('exit_scam_multi_candle_percent');
            if (value !== undefined) {
                exitScamMultiPercentEl.value = value;
            }
        }

        const trendDetectionEnabledEl = document.getElementById('trendDetectionEnabledDup');
        if (trendDetectionEnabledEl) {
            const value = getSettingValue('trend_detection_enabled');
            if (value !== undefined) {
                trendDetectionEnabledEl.checked = Boolean(value);
            }
        }

        const trendAnalysisPeriodEl = document.getElementById('trendAnalysisPeriodDup');
        if (trendAnalysisPeriodEl) {
            const value = getSettingValue('trend_analysis_period');
            if (value !== undefined) {
                trendAnalysisPeriodEl.value = value;
            }
        }

        const trendPriceChangeEl = document.getElementById('trendPriceChangeThresholdDup');
        if (trendPriceChangeEl) {
            const value = getSettingValue('trend_price_change_threshold');
            if (value !== undefined) {
                trendPriceChangeEl.value = value;
            }
        }

        const trendCandlesThresholdEl = document.getElementById('trendCandlesThresholdDup');
        if (trendCandlesThresholdEl) {
            const value = getSettingValue('trend_candles_threshold');
            if (value !== undefined) {
                trendCandlesThresholdEl.value = value;
            }
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
        
        const leverageCoinEl = document.getElementById('leverageCoinInput');
        if (leverageCoinEl) {
            // Загружаем из индивидуальных настроек, если есть, иначе из глобального конфига
            const leverageValue = getSettingValue('leverage');
            if (leverageValue !== undefined) {
                leverageCoinEl.value = leverageValue;
            }
        }
        
        // ✅ Enhanced RSI настройки для индивидуальных настроек монеты
        const enhancedRsiEnabledDupEl = document.getElementById('enhancedRsiEnabledDup');
        if (enhancedRsiEnabledDupEl) {
            const value = getSettingValue('enhanced_rsi_enabled');
            if (value !== undefined) {
                enhancedRsiEnabledDupEl.checked = Boolean(value);
            }
        }
        
        const enhancedRsiVolumeConfirmDupEl = document.getElementById('enhancedRsiVolumeConfirmDup');
        if (enhancedRsiVolumeConfirmDupEl) {
            const value = getSettingValue('enhanced_rsi_require_volume_confirmation');
            if (value !== undefined) {
                enhancedRsiVolumeConfirmDupEl.checked = Boolean(value);
            }
        }
        
        const enhancedRsiDivergenceConfirmDupEl = document.getElementById('enhancedRsiDivergenceConfirmDup');
        if (enhancedRsiDivergenceConfirmDupEl) {
            const value = getSettingValue('enhanced_rsi_require_divergence_confirmation');
            if (value !== undefined) {
                enhancedRsiDivergenceConfirmDupEl.checked = Boolean(value);
            }
        }
        
        const enhancedRsiUseStochRsiDupEl = document.getElementById('enhancedRsiUseStochRsiDup');
        if (enhancedRsiUseStochRsiDupEl) {
            const value = getSettingValue('enhanced_rsi_use_stoch_rsi');
            if (value !== undefined) {
                enhancedRsiUseStochRsiDupEl.checked = Boolean(value);
            }
        }
        
        this.highlightIndividualSettingDiffs(settings);
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
            this.pendingIndividualSettingsSymbol = symbol;
             const settings = await this.loadIndividualSettings(symbol);
            if (this.pendingIndividualSettingsSymbol !== symbol) {
                console.log('[BotsManager] ⏭️ Ответ для старой монеты, игнорируем');
                return;
            }
             
             if (settings) {
                 // Применяем настройки к UI и подсвечиваем отличия от основного конфига
                 this.applyIndividualSettingsToUI(settings);
                 this.updateIndividualSettingsStatus(true);
                 console.log(`[BotsManager] ✅ Индивидуальные настройки для ${symbol} применены`);
             } else {
                 // Сбрасываем UI к общим настройкам и убираем подсветку
                 this.clearIndividualSettingDiffHighlights();
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
        console.log('[BotsManager] 🔄 Сброс к общим настройкам');
        this.clearIndividualSettingDiffHighlights();
        const config = this.cachedAutoBotConfig || {};
        const fallback = {
            rsi_long_threshold: 29,
            rsi_short_threshold: 71,
            rsi_exit_long_with_trend: 65,
            rsi_exit_long_against_trend: 60,
            rsi_exit_short_with_trend: 35,
            rsi_exit_short_against_trend: 40,
            max_loss_percent: 15.0,
            take_profit_percent: 5.0,
            close_at_profit_enabled: true,
            trailing_stop_activation: 20.0,
            trailing_stop_distance: 5.0,
            trailing_take_distance: 0.5,
            trailing_update_interval: 3.0,
            max_position_hours: 0,
            break_even_protection: true,
            break_even_trigger: 20.0,
                    loss_reentry_protection: true,
                    loss_reentry_count: 1,
                    loss_reentry_candles: 3,
            avoid_down_trend: config.avoid_down_trend !== false,
            loss_reentry_protection: config.loss_reentry_protection !== false,
            loss_reentry_count: config.loss_reentry_count || 1,
            loss_reentry_candles: config.loss_reentry_candles || 3,
            avoid_up_trend: config.avoid_up_trend !== false,
            enable_maturity_check: config.enable_maturity_check !== false,
            min_candles_for_maturity: (config.min_candles_for_maturity !== undefined ? config.min_candles_for_maturity : 400),
            min_rsi_low: (config.min_rsi_low !== undefined ? config.min_rsi_low : 35),
            max_rsi_high: (config.max_rsi_high !== undefined ? config.max_rsi_high : 65),
            rsi_time_filter_enabled: (config.rsi_time_filter_enabled !== undefined ? config.rsi_time_filter_enabled : true),
            rsi_time_filter_candles: (config.rsi_time_filter_candles !== undefined ? config.rsi_time_filter_candles : 6),
            rsi_time_filter_upper: (config.rsi_time_filter_upper !== undefined ? config.rsi_time_filter_upper : 65),
            rsi_time_filter_lower: (config.rsi_time_filter_lower !== undefined ? config.rsi_time_filter_lower : 35),
            exit_scam_enabled: (config.exit_scam_enabled !== undefined ? config.exit_scam_enabled : true),
            exit_scam_candles: (config.exit_scam_candles !== undefined ? config.exit_scam_candles : 8),
            exit_scam_single_candle_percent: (config.exit_scam_single_candle_percent !== undefined ? config.exit_scam_single_candle_percent : 15),
            exit_scam_multi_candle_count: (config.exit_scam_multi_candle_count !== undefined ? config.exit_scam_multi_candle_count : 4),
            exit_scam_multi_candle_percent: (config.exit_scam_multi_candle_percent !== undefined ? config.exit_scam_multi_candle_percent : 50),
            trend_detection_enabled: (config.trend_detection_enabled !== undefined ? config.trend_detection_enabled : false),
            trend_analysis_period: (config.trend_analysis_period !== undefined ? config.trend_analysis_period : 30),
            trend_price_change_threshold: (config.trend_price_change_threshold !== undefined ? config.trend_price_change_threshold : 7),
            trend_candles_threshold: (config.trend_candles_threshold !== undefined ? config.trend_candles_threshold : 70)
        };

        const get = (key, defaultValue) => {
            const value = config[key];
            return value !== undefined ? value : defaultValue;
        };

        const setValue = (id, value) => {
            const el = document.getElementById(id);
            if (el !== null && value !== undefined) {
                el.value = value;
            }
        };

        setValue('rsiLongThresholdDup', get('rsi_long_threshold', fallback.rsi_long_threshold));
        setValue('rsiShortThresholdDup', get('rsi_short_threshold', fallback.rsi_short_threshold));
        setValue('rsiExitLongWithTrendDup', get('rsi_exit_long_with_trend', fallback.rsi_exit_long_with_trend));
        setValue('rsiExitLongAgainstTrendDup', get('rsi_exit_long_against_trend', fallback.rsi_exit_long_against_trend));
        setValue('rsiExitShortWithTrendDup', get('rsi_exit_short_with_trend', fallback.rsi_exit_short_with_trend));
        setValue('rsiExitShortAgainstTrendDup', get('rsi_exit_short_against_trend', fallback.rsi_exit_short_against_trend));
        setValue('maxLossPercentDup', get('max_loss_percent', fallback.max_loss_percent));
        setValue('takeProfitPercentDup', get('take_profit_percent', fallback.take_profit_percent));
        const closeAtProfitDupEl = document.getElementById('closeAtProfitEnabledDup');
        if (closeAtProfitDupEl) closeAtProfitDupEl.checked = get('close_at_profit_enabled', true) !== false;
        setValue('trailingStopActivationDup', get('trailing_stop_activation', fallback.trailing_stop_activation));
        setValue('trailingStopDistanceDup', get('trailing_stop_distance', fallback.trailing_stop_distance));
        setValue('trailingTakeDistanceDup', get('trailing_take_distance', fallback.trailing_take_distance));
        setValue('trailingUpdateIntervalDup', get('trailing_update_interval', fallback.trailing_update_interval));

        const maxHoursEl = document.getElementById('maxPositionHoursDup');
        if (maxHoursEl) {
            const hours = get('max_position_hours', fallback.max_position_hours);
            maxHoursEl.value = Math.round((hours || 0) * 3600);
        }

        const breakEvenEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenEl) {
            breakEvenEl.checked = get('break_even_protection', fallback.break_even_protection);
        }

        const breakEvenTriggerEl = document.getElementById('breakEvenTriggerDup');
        if (breakEvenTriggerEl) {
            breakEvenTriggerEl.value = get('break_even_trigger', fallback.break_even_trigger);
        }

        const avoidDownTrendEl = document.getElementById('avoidDownTrendDup');
        if (avoidDownTrendEl) {
            avoidDownTrendEl.checked = get('avoid_down_trend', fallback.avoid_down_trend);
        }

        const avoidUpTrendEl = document.getElementById('avoidUpTrendDup');
        if (avoidUpTrendEl) {
            avoidUpTrendEl.checked = get('avoid_up_trend', fallback.avoid_up_trend);
        }

        const maturityEl = document.getElementById('enableMaturityCheckDup');
        if (maturityEl) {
            maturityEl.checked = get('enable_maturity_check', fallback.enable_maturity_check);
        }

        const minCandlesMaturityEl = document.getElementById('minCandlesForMaturityDup');
        if (minCandlesMaturityEl) {
            minCandlesMaturityEl.value = get('min_candles_for_maturity', fallback.min_candles_for_maturity);
        }

        const minRsiLowEl = document.getElementById('minRsiLowDup');
        if (minRsiLowEl) {
            minRsiLowEl.value = get('min_rsi_low', fallback.min_rsi_low);
        }

        const maxRsiHighEl = document.getElementById('maxRsiHighDup');
        if (maxRsiHighEl) {
            maxRsiHighEl.value = get('max_rsi_high', fallback.max_rsi_high);
        }

        const rsiTimeFilterEnabledEl = document.getElementById('rsiTimeFilterEnabledDup');
        if (rsiTimeFilterEnabledEl) {
            rsiTimeFilterEnabledEl.checked = get('rsi_time_filter_enabled', fallback.rsi_time_filter_enabled);
        }

        const rsiTimeFilterCandlesEl = document.getElementById('rsiTimeFilterCandlesDup');
        if (rsiTimeFilterCandlesEl) {
            rsiTimeFilterCandlesEl.value = get('rsi_time_filter_candles', fallback.rsi_time_filter_candles);
        }

        const rsiTimeFilterUpperEl = document.getElementById('rsiTimeFilterUpperDup');
        if (rsiTimeFilterUpperEl) {
            rsiTimeFilterUpperEl.value = get('rsi_time_filter_upper', fallback.rsi_time_filter_upper);
        }

        const rsiTimeFilterLowerEl = document.getElementById('rsiTimeFilterLowerDup');
        if (rsiTimeFilterLowerEl) {
            rsiTimeFilterLowerEl.value = get('rsi_time_filter_lower', fallback.rsi_time_filter_lower);
        }

        const exitScamEnabledEl = document.getElementById('exitScamEnabledDup');
        if (exitScamEnabledEl) {
            exitScamEnabledEl.checked = get('exit_scam_enabled', fallback.exit_scam_enabled);
        }

        const exitScamCandlesEl = document.getElementById('exitScamCandlesDup');
        if (exitScamCandlesEl) {
            exitScamCandlesEl.value = get('exit_scam_candles', fallback.exit_scam_candles);
        }

        const exitScamSingleEl = document.getElementById('exitScamSingleCandleDup');
        if (exitScamSingleEl) {
            exitScamSingleEl.value = get('exit_scam_single_candle_percent', fallback.exit_scam_single_candle_percent);
        }

        const exitScamMultiCountEl = document.getElementById('exitScamMultiCountDup');
        if (exitScamMultiCountEl) {
            exitScamMultiCountEl.value = get('exit_scam_multi_candle_count', fallback.exit_scam_multi_candle_count);
        }

        const exitScamMultiPercentEl = document.getElementById('exitScamMultiPercentDup');
        if (exitScamMultiPercentEl) {
            exitScamMultiPercentEl.value = get('exit_scam_multi_candle_percent', fallback.exit_scam_multi_candle_percent);
        }

        const trendDetectionEnabledEl = document.getElementById('trendDetectionEnabledDup');
        if (trendDetectionEnabledEl) {
            trendDetectionEnabledEl.checked = get('trend_detection_enabled', fallback.trend_detection_enabled);
        }

        const trendAnalysisPeriodEl = document.getElementById('trendAnalysisPeriodDup');
        if (trendAnalysisPeriodEl) {
            trendAnalysisPeriodEl.value = get('trend_analysis_period', fallback.trend_analysis_period);
        }

        const trendPriceChangeEl = document.getElementById('trendPriceChangeThresholdDup');
        if (trendPriceChangeEl) {
            trendPriceChangeEl.value = get('trend_price_change_threshold', fallback.trend_price_change_threshold);
        }

        const trendCandlesThresholdEl = document.getElementById('trendCandlesThresholdDup');
        if (trendCandlesThresholdEl) {
            trendCandlesThresholdEl.value = get('trend_candles_threshold', fallback.trend_candles_threshold);
        }
        
        // Объем торговли и плечо
        const volumeModeEl = document.getElementById('volumeModeSelect');
        if (volumeModeEl) {
            volumeModeEl.value = get('default_position_mode', 'usdt');
        }
        
        const volumeValueEl = document.getElementById('volumeValueInput');
        if (volumeValueEl) {
            volumeValueEl.value = get('default_position_size', 10);
        }
        
        const leverageCoinEl = document.getElementById('leverageCoinInput');
        if (leverageCoinEl) {
            leverageCoinEl.value = get('leverage', 10);
        }
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
                // Добавляем основные настройки торговли (volume_mode, volume_value, leverage)
                const volumeModeEl = document.getElementById('volumeModeSelect');
                if (volumeModeEl) settings.volume_mode = volumeModeEl.value;
                const volumeValueEl = document.getElementById('volumeValueInput');
                if (volumeValueEl) settings.volume_value = parseFloat(volumeValueEl.value) || 10;
                const leverageCoinEl = document.getElementById('leverageCoinInput');
                if (leverageCoinEl) settings.leverage = parseInt(leverageCoinEl.value) || 10;
                const success = await this.saveIndividualSettings(this.selectedCoin.symbol, settings);
                if (success) {
                    this.highlightIndividualSettingDiffs(settings);
                    this.updateIndividualSettingsStatus(true);
                }
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
        
        // Кнопка «Подобрать ExitScam по истории» для выбранной монеты
        const learnExitScamBtn = document.getElementById('learnExitScamForCoinBtn');
        if (learnExitScamBtn) {
            learnExitScamBtn.addEventListener('click', () => this.learnExitScamForCoin());
        }
        // Кнопка «Индивидуальный ExitScam для всех монет» — расчёт по истории для каждой монеты
        const learnExitScamAllBtn = document.getElementById('learnExitScamForAllCoinsBtn');
        if (learnExitScamAllBtn) {
            learnExitScamAllBtn.addEventListener('click', () => this.learnExitScamForAllCoins());
        }
        const resetExitScamToConfigBtn = document.getElementById('resetExitScamToConfigForAllBtn');
        if (resetExitScamToConfigBtn) {
            resetExitScamToConfigBtn.addEventListener('click', () => this.resetExitScamToConfigForAll());
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
        
        // Обработчики для кнопок ручного запуска в секции настроек
        const manualLaunchLongBtn = document.getElementById('manualLaunchLongBtn');
        if (manualLaunchLongBtn) {
            manualLaunchLongBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для запуска', 'warning');
                    return;
                }
                
                await this.quickLaunchBot('LONG');
            });
        }
        
        const manualLaunchShortBtn = document.getElementById('manualLaunchShortBtn');
        if (manualLaunchShortBtn) {
            manualLaunchShortBtn.addEventListener('click', async () => {
                if (!this.selectedCoin) {
                    this.showNotification('⚠️ Выберите монету для запуска', 'warning');
                    return;
                }
                
                await this.quickLaunchBot('SHORT');
            });
        }
        
        console.log('[BotsManager] ✅ Кнопки быстрого запуска инициализированы');
    }

    async quickLaunchBot(direction) {
        if (!this.selectedCoin) return;
        
        try {
            console.log(`[BotsManager] 🚀 Быстрый запуск ${direction} бота для ${this.selectedCoin.symbol}`);
            await this.createBot(direction);
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
    }

    getBotDeleteButtonHtml(bot) {
        return `<span onclick="event.stopPropagation(); window.app.botsManager.deleteBot('${bot.symbol}')" title="${window.languageUtils.translate('delete_btn')}" class="bot-icon-btn bot-icon-delete">🗑</span>`;
    }

    getBotControlButtonsHtml(bot) {
        return (this.getBotStopButtonHtml(bot) || '') + this.getBotDeleteButtonHtml(bot);
    }

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
                    blacklist: data.config.blacklist || [],
                    scope: ['all', 'whitelist', 'blacklist'].includes(data.config.scope) ? data.config.scope : 'all'
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
        const filtersSearchInput = document.getElementById('filtersSearchInput');
        if (filtersSearchInput && !filtersSearchInput.dataset.filterInit) {
            filtersSearchInput.dataset.filterInit = '1';
            filtersSearchInput.addEventListener('input', (e) => {
                this.performFiltersSearch(e.target.value);
            });
        }
        const filtersTab = document.getElementById('filtersTab');
        if (!filtersTab || filtersTab.dataset.controlsInit) return;
        filtersTab.dataset.controlsInit = '1';
        const exportBtn = document.getElementById('exportFiltersBtn');
        const importBtn = document.getElementById('importFiltersBtn');
        const importFile = document.getElementById('importFiltersFile');
        if (exportBtn) exportBtn.addEventListener('click', () => this.exportFiltersToJson());
        if (importBtn) importBtn.addEventListener('click', () => importFile && importFile.click());
        if (importFile) importFile.addEventListener('change', (e) => { this.importFiltersFromJson(e.target.files[0]); e.target.value = ''; });
        const clearWhitelistBtn = document.getElementById('clearWhitelistBtn');
        const clearBlacklistBtn = document.getElementById('clearBlacklistBtn');
        if (clearWhitelistBtn) clearWhitelistBtn.addEventListener('click', () => this.clearWhitelist());
        if (clearBlacklistBtn) clearBlacklistBtn.addEventListener('click', () => this.clearBlacklist());
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

    async clearWhitelist() {
        const whitelist = this.filtersData?.whitelist || [];
        if (whitelist.length === 0) {
            this.showNotification('Белый список уже пуст', 'info');
            return;
        }
        const msg = 'Удалить все ' + whitelist.length + ' монет из белого списка?';
        if (!confirm(msg)) return;
        try {
            await this.updateFilters({ whitelist: [] });
            this.showNotification('Белый список очищен', 'success');
        } catch (error) {
            console.error('[BotsManager] Ошибка очистки белого списка:', error);
            this.showNotification('Ошибка очистки белого списка', 'error');
        }
    }

    async clearBlacklist() {
        const blacklist = this.filtersData?.blacklist || [];
        if (blacklist.length === 0) {
            this.showNotification('Черный список уже пуст', 'info');
            return;
        }
        const msg = 'Удалить все ' + blacklist.length + ' монет из черного списка?';
        if (!confirm(msg)) return;
        try {
            await this.updateFilters({ blacklist: [] });
            this.showNotification('Черный список очищен', 'success');
        } catch (error) {
            console.error('[BotsManager] Ошибка очистки черного списка:', error);
            this.showNotification('Ошибка очистки черного списка', 'error');
        }
    }

    exportFiltersToJson() {
        const w = this.filtersData?.whitelist || [];
        const b = this.filtersData?.blacklist || [];
        const scope = this.filtersData?.scope || 'all';
        const payload = { whitelist: w, blacklist: b, scope };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const iso = new Date().toISOString().slice(0, 19).replace('T', '_').replace(/:/g, '-');
        a.download = 'coin_filters_' + iso + '.json';
        a.click();
        URL.revokeObjectURL(url);
        this.showNotification('Списки выгружены в JSON', 'success');
    }

    async importFiltersFromJson(file) {
        if (!file) return;
        if (!this.serviceOnline) {
            this.showNotification('Сервис ботов недоступен. Запустите bots.py', 'error');
            return;
        }
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            const w = Array.isArray(data.whitelist) ? data.whitelist : [];
            const b = Array.isArray(data.blacklist) ? data.blacklist : [];
            const scope = ['all', 'whitelist', 'blacklist'].includes(data.scope) ? data.scope : 'all';
            const toSymbols = (arr) => arr.map(x => typeof x === 'string' ? x.trim().toUpperCase() : (x && x.symbol ? String(x.symbol).trim().toUpperCase() : '')).filter(Boolean);
            const whitelist = [...new Set(toSymbols(w))];
            const blacklist = [...new Set(toSymbols(b))];
            await this.updateFilters({ whitelist, blacklist, scope });
            this.showNotification('Списки загружены из JSON в БД', 'success');
        } catch (err) {
            console.error('[BotsManager] Ошибка импорта фильтров:', err);
            this.showNotification('Ошибка: неверный формат JSON или чтение файла', 'error');
        }
    }

    async updateFilters(updates) {
        // Убеждаемся что filtersData инициализирован
        if (!this.filtersData) {
            this.filtersData = { whitelist: [], blacklist: [], scope: 'all' };
        }
        
        // Обновляем локальные данные
        if (updates.whitelist !== undefined) {
            this.filtersData.whitelist = updates.whitelist;
        }
        if (updates.blacklist !== undefined) {
            this.filtersData.blacklist = updates.blacklist;
        }
        if (updates.scope !== undefined) {
            this.filtersData.scope = updates.scope;
        }
        
        // Отправляем на сервер (в БД через API)
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
        console.log(`[BotsManager] 🔔 showNotification ВЫЗВАН [${type}]:`, message);
        console.log(`[BotsManager] 🔍 this:`, this);
        console.log(`[BotsManager] 🔍 window.toastManager:`, window.toastManager);
        
        // ✅ Принудительно инициализируем toastManager, если его нет
        if (!window.toastManager) {
            if (typeof ToastManager !== 'undefined') {
                window.toastManager = new ToastManager();
            } else if (window.showToast) {
                window.showToast(message, type, 4000);
                return;
            } else {
                return;
            }
        }
        
        try {
            // ✅ Принудительно инициализируем контейнер
            if (!window.toastManager.container) {
                console.log('[BotsManager] 🔧 Инициализация контейнера toast...');
                window.toastManager.init();
            }
            
            // ✅ Проверяем, что контейнер в DOM
            if (!window.toastManager.container || !document.body.contains(window.toastManager.container)) {
                console.log('[BotsManager] 🔧 Добавление контейнера toast в DOM...');
                if (document.body) {
                    if (!window.toastManager.container) {
                        window.toastManager.init();
                    }
                    if (window.toastManager.container && !document.body.contains(window.toastManager.container)) {
                        document.body.appendChild(window.toastManager.container);
                        console.log('[BotsManager] ✅ Контейнер добавлен в DOM');
                    }
                } else {
                    console.error('[BotsManager] ❌ document.body не доступен! Пропускаем уведомление.');
                    return; // ❌ НЕ используем alert - просто пропускаем
                }
            }
            
            // ✅ Принудительно устанавливаем стили контейнера
            if (window.toastManager.container) {
                const container = window.toastManager.container;
                container.style.position = 'fixed';
                container.style.top = '20px';
                container.style.right = '20px';
                container.style.zIndex = '999999';
                container.style.display = 'flex';
                container.style.flexDirection = 'column';
                container.style.gap = '10px';
                container.style.maxWidth = '400px';
                container.style.pointerEvents = 'none';
                container.style.visibility = 'visible';
                container.style.opacity = '1';
            }
            
            // ✅ Показываем уведомление (автоматически скрывается через 4-5 секунд)
            switch(type) {
                case 'success':
                    window.toastManager.success(message, 4500);
                    console.log('[BotsManager] ✅ Вызван toastManager.success()');
                    break;
                case 'error':
                    window.toastManager.error(message, 5000); // 5 секунд для ошибок
                    console.log('[BotsManager] ❌ Вызван toastManager.error()');
                    break;
                case 'warning':
                    window.toastManager.warning(message, 4000); // 4 секунды
                    console.log('[BotsManager] ⚠️ Вызван toastManager.warning()');
                    break;
                case 'info':
                default:
                    window.toastManager.info(message, 3000); // 3 секунды
                    console.log('[BotsManager] ℹ️ Вызван toastManager.info()');
                    break;
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка при показе уведомления:', error);
            if (window.showToast) {
                try { window.showToast(message, type, 4000); } catch (e) { /* ignore */ }
            }
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

        // Если уже в белом списке — сообщаем и подсвечиваем
        if (whitelist.includes(symbol)) {
            this.showNotification('⚠️ Монета уже в белом списке', 'warning');
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

        // Если уже в черном списке — сообщаем и подсвечиваем
        if (blacklist.includes(symbol)) {
            this.showNotification('⚠️ Монета уже в черном списке', 'warning');
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

        // Если уже в белом списке — сообщаем и подсвечиваем
        if (whitelist.includes(symbol)) {
            this.showNotification('⚠️ Монета уже в белом списке', 'warning');
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

        // Если уже в черном списке — сообщаем и подсвечиваем
        if (blacklist.includes(symbol)) {
            this.showNotification('⚠️ Монета уже в черном списке', 'warning');
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
                    
                    const rightPanelHtml = displayListForDetails.map(bot => {
                    const isVirtual = !!bot.is_virtual;
                    const isActive = isVirtual || bot.status === 'running' || bot.status === 'idle' || 
                                    bot.status === 'in_position_long' || bot.status === 'in_position_short' ||
                                    bot.status === 'armed_up' || bot.status === 'armed_down';
                    const statusColor = isActive ? '#4caf50' : '#ff5722';
                    const statusText = isVirtual ? (window.languageUtils?.translate('fullai_virtual_position') || 'Виртуальная') : (isActive ? window.languageUtils.translate('active_status') : (bot.status === 'paused' ? window.languageUtils.translate('paused_status') : (bot.status === 'idle' ? window.languageUtils.translate('waiting_status') : window.languageUtils.translate('stopped_status'))));
                    
                    const d = this.getCompactCardData(bot);
                    const t = k => window.languageUtils?.translate(k) || this.getTranslation(k);
                    const exchangeUrl = this.getExchangeLink(bot.symbol, 'bybit');
                    // Цвет карточки по PnL: зелёный — прибыль, красный — убыток (направление Long/Short уже показано подписью)
                    const pnlValue = isVirtual ? (bot.unrealized_pnl ?? 0) : (bot.unrealized_pnl_usdt ?? bot.unrealized_pnl ?? 0);
                    const isProfit = Number(pnlValue) >= 0;
                    const cardBg = isVirtual ? 'rgba(156, 39, 176, 0.12)' : (isProfit ? 'rgba(76, 175, 80, 0.08)' : 'rgba(244, 67, 54, 0.08)');
                    const virtualAttrs = isVirtual ? ` data-is-virtual="true" data-virtual-index="${bot._virtualIndex || 0}"` : '';
                    const pnlVal = isVirtual ? (bot.unrealized_pnl != null ? `${(bot.unrealized_pnl || 0).toFixed(2)}%` : '-') : `$${(bot.unrealized_pnl_usdt || bot.unrealized_pnl || 0).toFixed(3)}`;
                    const htmlResult = `
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
                    
                    return htmlResult;
                }).join('');

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
    }

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
    }

    startPeriodicUpdate() {
        // Обновляем данные с единым интервалом
        this.updateInterval = setInterval(() => {
            if (this.serviceOnline) {
                this.logDebug('[BotsManager] 🔄 Автообновление данных...');
                
                // Обновляем основные данные
                this.loadCoinsRsiData();
                this.loadDelistedCoins(); // Загружаем делистинговые монеты
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
    }
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
            button.addEventListener('click', async () => {
                // Убираем активность со всех кнопок
                scopeButtons.forEach(btn => btn.classList.remove('active'));
                
                // Добавляем активность на нажатую кнопку
                button.classList.add('active');
                
                // Обновляем скрытое поле
                const value = button.getAttribute('data-value');
                const oldValue = scopeInput.value;
                scopeInput.value = value;
                
                console.log('[BotsManager] 🎯 Область действия изменена на:', value, '(было:', oldValue + ')');
                console.log('[BotsManager] 🔍 Проверка: autoBotScope.value =', scopeInput.value);
                
                if (oldValue !== value) this.scheduleToggleAutoSave(scopeInput);
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
            // Один раз загружаем ВСЁ параллельно: auto-bot, system-config, fullai-config — чтобы не было подмены (100→10, ПРИИ выкл→вкл)
            const [autoBotResponse, systemResponse, fullaiResponse] = await Promise.all([
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`),
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`),
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/fullai-config`)
            ]);
            
            if (!autoBotResponse.ok || !systemResponse.ok) {
                throw new Error(`HTTP ${autoBotResponse.status} или ${systemResponse.status}`);
            }
            
            const autoBotData = await autoBotResponse.json();
            const systemData = await systemResponse.json();
            const fullaiData = fullaiResponse.ok ? await fullaiResponse.json() : { success: false, config: {} };
            
            if (autoBotData.success && systemData.success) {
                // Мержим fullai-config в autoBot ДО первой отрисовки — чтобы тумблер ПРИИ и «Свечей без сделок» сразу были правильные, без подмены
                const autoBotMerged = { ...(autoBotData.config || {}) };
                if (fullaiData.success && fullaiData.config && typeof fullaiData.config === 'object') {
                    const fc = fullaiData.config;
                    if (fc.full_ai_control !== undefined) autoBotMerged.full_ai_control = fc.full_ai_control;
                    if (fc.fullai_adaptive_enabled !== undefined) autoBotMerged.fullai_adaptive_enabled = fc.fullai_adaptive_enabled;
                    if (fc.fullai_adaptive_dead_candles !== undefined) autoBotMerged.fullai_adaptive_dead_candles = fc.fullai_adaptive_dead_candles;
                    if (fc.fullai_adaptive_virtual_success_count !== undefined) autoBotMerged.fullai_adaptive_virtual_success_count = fc.fullai_adaptive_virtual_success_count;
                    if (fc.fullai_adaptive_real_loss_to_retry !== undefined) autoBotMerged.fullai_adaptive_real_loss_to_retry = fc.fullai_adaptive_real_loss_to_retry;
                    if (fc.fullai_adaptive_virtual_round_size !== undefined) autoBotMerged.fullai_adaptive_virtual_round_size = fc.fullai_adaptive_virtual_round_size;
                    if (fc.fullai_adaptive_virtual_max_failures !== undefined) autoBotMerged.fullai_adaptive_virtual_max_failures = fc.fullai_adaptive_virtual_max_failures;
                }
                
                const config = {
                    autoBot: autoBotMerged,
                    system: systemData.config
                };
                
                const timeframeData = await this.loadTimeframe();
                if (timeframeData) {
                    config.system = config.system || {};
                    config.system.timeframe = timeframeData;
                }
                
                // Одна отрисовка с уже правильными данными — без подмены 100→10 и без переключения ПРИИ при смене вкладки
                this.populateConfigurationForm(config);
                this.syncDuplicateSettings(autoBotMerged);
                
                // КРИТИЧЕСКИ ВАЖНО: Инициализируем глобальный переключатель Auto Bot
                console.log('[BotsManager] 🤖 Инициализация глобального переключателя Auto Bot...');
                this.initializeGlobalAutoBotToggle();
            this.initializeMobileAutoBotToggle();
                
                // Обновляем интерфейс с текущим таймфреймом
                if (config.system && config.system.timeframe) {
                    this.updateTimeframeInUI(config.system.timeframe);
                }
                
                // Обновляем блок AI из /api/ai/config, чтобы переключатели показывали сохранённые значения
                // (они пишутся в RiskConfig/AIConfig, а не в auto-bot)
                if (window.aiConfigManager && typeof window.aiConfigManager.loadAIConfig === 'function') {
                    try {
                        await window.aiConfigManager.loadAIConfig();
                    } catch (aiErr) {
                        console.warn('[BotsManager] Обновление AI-блока:', aiErr);
                    }
                }
                
                console.log('[BotsManager] ✅ Конфигурация загружена и применена');
                this.aiConfigDirty = false;
                this.updateFloatingSaveButtonVisibility();
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
        // Устанавливаем флаг, чтобы предотвратить автосохранение при программном изменении
        this.isProgrammaticChange = true;
        
        this.logDebug('[BotsManager] 🔧 Заполнение формы конфигурации:', config);
        this.logDebug('[BotsManager] 🔍 DOM готовность:', document.readyState);
        this.logDebug('[BotsManager] 🔍 Элемент positionSyncInterval существует:', !!document.getElementById('positionSyncInterval'));
        this.logDebug('[BotsManager] 🔍 Детали конфигурации:');
        this.logDebug('   autoBot:', config.autoBot);
        this.logDebug('   system:', config.system);
        
        const autoBotConfig = config.autoBot || config;
        if (!autoBotConfig.default_position_mode) {
            autoBotConfig.default_position_mode = 'usdt';
        }
        
        // ✅ Кэшируем конфигурацию Auto Bot для быстрого доступа (для updateCoinInfo и др.)
        this.cachedAutoBotConfig = autoBotConfig;
        
        // ✅ ВСЕГДА обновляем originalConfig при загрузке конфигурации из бэкенда
        // Это гарантирует, что после сохранения и перезагрузки конфигурации originalConfig синхронизирован
        this.originalConfig = {
            autoBot: JSON.parse(JSON.stringify(autoBotConfig)), // Глубокое копирование
            system: JSON.parse(JSON.stringify(config.system || {}))
        };
        console.log(`[BotsManager] 💾 originalConfig обновлен из бэкенда для отслеживания изменений`);
        console.log(`[BotsManager] 🔍 originalConfig ключи:`, Object.keys(this.originalConfig.autoBot));
        console.log(`[BotsManager] 🔍 trailing_stop_activation в originalConfig:`, this.originalConfig.autoBot.trailing_stop_activation);
        console.log(`[BotsManager] 🔍 trailing_stop_distance в originalConfig:`, this.originalConfig.autoBot.trailing_stop_distance);
        console.log(`[BotsManager] 🔍 break_even_trigger в originalConfig:`, this.originalConfig.autoBot.break_even_trigger_percent ?? this.originalConfig.autoBot.break_even_trigger);
            
            // Защита от повторных входов после убытка
            const lossReentryProtectionEl = document.getElementById('lossReentryProtection');
            if (lossReentryProtectionEl) {
                lossReentryProtectionEl.checked = autoBotConfig.loss_reentry_protection !== false;
                console.log('[BotsManager] 🛡️ Защита от повторных входов:', lossReentryProtectionEl.checked);
            }

            const lossReentryCountEl = document.getElementById('lossReentryCount');
            if (lossReentryCountEl) {
                lossReentryCountEl.value = autoBotConfig.loss_reentry_count || 1;
                console.log('[BotsManager] 🔢 Количество убыточных позиций (N):', lossReentryCountEl.value);
            }

            const lossReentryCandlesEl = document.getElementById('lossReentryCandles');
            if (lossReentryCandlesEl) {
                lossReentryCandlesEl.value = autoBotConfig.loss_reentry_candles || 3;
                console.log('[BotsManager] 🕯️ ВХОД ЧЕРЕЗ X СВЕЧЕЙ:', lossReentryCandlesEl.value);
            }
        console.log(`[BotsManager] 🔍 avoid_down_trend в originalConfig:`, this.originalConfig.autoBot.avoid_down_trend);
        console.log(`[BotsManager] 🔍 avoid_up_trend в originalConfig:`, this.originalConfig.autoBot.avoid_up_trend);
        
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
        
        // ai_enabled в auto-bot конфиге задаётся мастер-переключателем aiEnabled (см. mapElementIdToConfigKey)
        const aiConfidenceEl = document.getElementById('aiMinConfidence');
        if (aiConfidenceEl) {
            const value = Number.parseFloat(autoBotConfig.ai_min_confidence);
            aiConfidenceEl.value = Number.isFinite(value) ? value : 0.7;
        }
        
        const aiOverrideEl = document.getElementById('aiOverrideOriginal');
        if (aiOverrideEl) {
            const overrideValue = autoBotConfig.ai_override_original;
            aiOverrideEl.checked = overrideValue !== false;
        }
        
        // ✅ AI оптимальный вход (может быть в AI секции, но сохраняется в auto-bot конфиге)
        const optimalEntryEl = document.getElementById('optimalEntryEnabled');
        if (optimalEntryEl) {
            optimalEntryEl.checked = Boolean(autoBotConfig.ai_optimal_entry_enabled);
            console.log('[BotsManager] 🎯 AI оптимальный вход:', optimalEntryEl.checked);
        }
        
        // ✅ FullAI адаптивные параметры (из auto-bot ответа; GET auto-bot уже подмешивает значения из AutoBotConfig)
        const deadCandles = autoBotConfig.fullai_adaptive_dead_candles;
        if (deadCandles !== undefined && document.getElementById('fullaiAdaptiveDeadCandles')) {
            document.getElementById('fullaiAdaptiveDeadCandles').value = parseInt(deadCandles, 10) || 10;
        }
        const virtualSuccess = autoBotConfig.fullai_adaptive_virtual_success_count ?? autoBotConfig.fullai_adaptive_virtual_success;
        if (virtualSuccess !== undefined && document.getElementById('fullaiAdaptiveVirtualSuccess')) {
            document.getElementById('fullaiAdaptiveVirtualSuccess').value = parseInt(virtualSuccess, 10) || 3;
        }
        const realLoss = autoBotConfig.fullai_adaptive_real_loss_to_retry ?? autoBotConfig.fullai_adaptive_real_loss;
        if (realLoss !== undefined && document.getElementById('fullaiAdaptiveRealLoss')) {
            document.getElementById('fullaiAdaptiveRealLoss').value = parseInt(realLoss, 10) || 1;
        }
        const roundSize = autoBotConfig.fullai_adaptive_virtual_round_size ?? autoBotConfig.fullai_adaptive_round_size;
        if (roundSize !== undefined && document.getElementById('fullaiAdaptiveRoundSize')) {
            document.getElementById('fullaiAdaptiveRoundSize').value = parseInt(roundSize, 10) || 3;
        }
        const maxFailures = autoBotConfig.fullai_adaptive_virtual_max_failures ?? autoBotConfig.fullai_adaptive_max_failures;
        if (maxFailures !== undefined && document.getElementById('fullaiAdaptiveMaxFailures')) {
            document.getElementById('fullaiAdaptiveMaxFailures').value = parseInt(maxFailures, 10) || 0;
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
            console.log('[BotsManager] 📈 RSI SHORT порог:', rsiShortEl.value);
        }
        
        const rsiLimitEntryEl = document.getElementById('rsiLimitEntryEnabled');
        if (rsiLimitEntryEl) {
            rsiLimitEntryEl.checked = autoBotConfig.rsi_limit_entry_enabled === true;
        }
        const rsiLimitOffsetEl = document.getElementById('rsiLimitOffsetPercentGlobal');
        if (rsiLimitOffsetEl) {
            const v = parseFloat(autoBotConfig.rsi_limit_offset_percent);
            rsiLimitOffsetEl.value = (!isNaN(v) && v >= 0) ? v : 0.2;
        }
        const rsiLimitExitEl = document.getElementById('rsiLimitExitEnabled');
        if (rsiLimitExitEl) {
            rsiLimitExitEl.checked = autoBotConfig.rsi_limit_exit_enabled === true;
        }
        const rsiLimitExitOffsetEl = document.getElementById('rsiLimitExitOffsetPercentGlobal');
        if (rsiLimitExitOffsetEl) {
            const v = parseFloat(autoBotConfig.rsi_limit_exit_offset_percent);
            rsiLimitExitOffsetEl.value = (!isNaN(v) && v >= 0) ? v : 0.2;
        }
        
        const positionSizeEl = document.getElementById('defaultPositionSize');
        if (positionSizeEl) {
            positionSizeEl.value = autoBotConfig.default_position_size || 10;
            console.log('[BotsManager] 💰 Размер позиции:', positionSizeEl.value);
        }
        const positionModeEl = document.getElementById('defaultPositionMode');
        if (positionModeEl) {
            positionModeEl.value = autoBotConfig.default_position_mode || 'usdt';
            console.log('[BotsManager] 🔄 Режим размера позиции:', positionModeEl.value);
        }
        
        const leverageEl = document.getElementById('leverage');
        if (leverageEl) {
            leverageEl.value = autoBotConfig.leverage || 10;
            console.log('[BotsManager] ⚡ Кредитное плечо:', leverageEl.value);
        }
        
        const checkIntervalEl = document.getElementById('checkInterval');
        if (checkIntervalEl && autoBotConfig.check_interval !== undefined) {
            checkIntervalEl.value = autoBotConfig.check_interval;
            console.log('[BotsManager] ⏱️ Интервал проверки установлен:', autoBotConfig.check_interval, '(из API)');
        } else if (checkIntervalEl) {
            console.warn('[BotsManager] ⚠️ Интервал проверки не найден в API, оставляем поле пустым');
        }
        

        
        // ✅ Новые параметры RSI выхода с учетом тренда
        const rsiExitLongWithTrendEl = document.getElementById('rsiExitLongWithTrendGlobal');
        if (rsiExitLongWithTrendEl && rsiExitLongWithTrendEl.value) {
            rsiExitLongWithTrendEl.value = autoBotConfig.rsi_exit_long_with_trend || 65;
            console.log('[BotsManager] 🟢📈 RSI выход LONG (по тренду):', rsiExitLongWithTrendEl.value);
        }
        
        const rsiExitLongAgainstTrendEl = document.getElementById('rsiExitLongAgainstTrendGlobal');
        if (rsiExitLongAgainstTrendEl) {
            rsiExitLongAgainstTrendEl.value = autoBotConfig.rsi_exit_long_against_trend || 60;
            console.log('[BotsManager] 🟢📉 RSI выход LONG (против тренда):', rsiExitLongAgainstTrendEl.value);
        }
        
        const rsiExitShortWithTrendEl = document.getElementById('rsiExitShortWithTrendGlobal');
        if (rsiExitShortWithTrendEl) {
            rsiExitShortWithTrendEl.value = autoBotConfig.rsi_exit_short_with_trend || 35;
            console.log('[BotsManager] 🔴📉 RSI выход SHORT (по тренду):', rsiExitShortWithTrendEl.value);
        }
        
        const rsiExitShortAgainstTrendEl = document.getElementById('rsiExitShortAgainstTrendGlobal');
        if (rsiExitShortAgainstTrendEl) {
            rsiExitShortAgainstTrendEl.value = autoBotConfig.rsi_exit_short_against_trend || 40;
            console.log('[BotsManager] 🔴📈 RSI выход SHORT (против тренда):', rsiExitShortAgainstTrendEl.value);
        }
        
        const rsiExitMinCandlesEl = document.getElementById('rsiExitMinCandlesGlobal');
        if (rsiExitMinCandlesEl) {
            const v = parseInt(autoBotConfig.rsi_exit_min_candles, 10);
            rsiExitMinCandlesEl.value = (!isNaN(v) && v >= 0) ? v : 0;
            console.log('[BotsManager] ⏱️ Мин. свечей до выхода по RSI:', rsiExitMinCandlesEl.value);
        }
        const rsiExitMinMinutesEl = document.getElementById('rsiExitMinMinutesGlobal');
        if (rsiExitMinMinutesEl) {
            const v = parseInt(autoBotConfig.rsi_exit_min_minutes, 10);
            rsiExitMinMinutesEl.value = (!isNaN(v) && v >= 0) ? v : 0;
        }
        const rsiExitMinMovePercentEl = document.getElementById('rsiExitMinMovePercentGlobal');
        if (rsiExitMinMovePercentEl) {
            const v = parseFloat(autoBotConfig.rsi_exit_min_move_percent);
            rsiExitMinMovePercentEl.value = (v !== undefined && !isNaN(v) && v >= 0) ? v : 0;
        }
        const exitWaitBreakevenEl = document.getElementById('exitWaitBreakevenWhenLoss');
        if (exitWaitBreakevenEl) {
            exitWaitBreakevenEl.checked = autoBotConfig.exit_wait_breakeven_when_loss === true;
        }
        
        // Торговые настройки (перенесены в блок Торговые параметры)
        const tradingEnabledEl = document.getElementById('tradingEnabled');
        if (tradingEnabledEl) {
            tradingEnabledEl.checked = autoBotConfig.trading_enabled !== false;
            console.log('[BotsManager] 🎛️ Реальная торговля:', tradingEnabledEl.checked);
        }
        
        const useTestServerEl1 = document.getElementById('useTestServer');
        if (useTestServerEl1) {
            useTestServerEl1.checked = autoBotConfig.use_test_server || false;
            console.log('[BotsManager] 🧪 Тестовый сервер:', useTestServerEl1.checked);
        }
        
        // ==========================================
        // ЗАЩИТНЫЕ МЕХАНИЗМЫ
        // ==========================================
        
        const maxLossPercentEl = document.getElementById('maxLossPercent');
        if (maxLossPercentEl) {
            maxLossPercentEl.value = autoBotConfig.max_loss_percent || 15.0;
            console.log('[BotsManager] 🛡️ Макс. убыток (стоп-лосс):', maxLossPercentEl.value);
        }
        
        const takeProfitPercentEl = document.getElementById('takeProfitPercent');
        if (takeProfitPercentEl) {
            takeProfitPercentEl.value = autoBotConfig.take_profit_percent ?? 5.0;
            console.log('[BotsManager] 🎯 Защитный TP (%):', takeProfitPercentEl.value);
        }
        
        const closeAtProfitEnabledEl = document.getElementById('closeAtProfitEnabled');
        if (closeAtProfitEnabledEl) {
            closeAtProfitEnabledEl.checked = autoBotConfig.close_at_profit_enabled !== false;
            console.log('[BotsManager] 🎯 Закрывать по % прибыли:', closeAtProfitEnabledEl.checked);
        }
        
        const trailingStopActivationEl = document.getElementById('trailingStopActivation');
        if (trailingStopActivationEl) {
            const value = Number.parseFloat(autoBotConfig.trailing_stop_activation);
            trailingStopActivationEl.value = Number.isFinite(value) ? value : 20.0;
            console.log('[BotsManager] 📈 Активация trailing stop:', trailingStopActivationEl.value);
        }
        
        const trailingStopDistanceEl = document.getElementById('trailingStopDistance');
        if (trailingStopDistanceEl) {
            const value = Number.parseFloat(autoBotConfig.trailing_stop_distance);
            trailingStopDistanceEl.value = Number.isFinite(value) ? value : 5.0;
            console.log('[BotsManager] 📉 Расстояние trailing stop:', trailingStopDistanceEl.value);
        }

        const trailingTakeDistanceEl = document.getElementById('trailingTakeDistance');
        if (trailingTakeDistanceEl) {
            const value = autoBotConfig.trailing_take_distance;
            trailingTakeDistanceEl.value = (value !== undefined && value !== null) ? value : 0.5;
            console.log('[BotsManager] 🎯 Резервный trailing take:', trailingTakeDistanceEl.value);
        }

        const trailingUpdateIntervalEl = document.getElementById('trailingUpdateInterval');
        if (trailingUpdateIntervalEl) {
            const value = autoBotConfig.trailing_update_interval;
            trailingUpdateIntervalEl.value = (value !== undefined && value !== null) ? value : 3.0;
            console.log('[BotsManager] ⏱️ Интервал обновления трейлинга:', trailingUpdateIntervalEl.value);
        }
        
        const maxPositionHoursEl = document.getElementById('maxPositionHours');
        if (maxPositionHoursEl) {
            const hours = autoBotConfig.max_position_hours || 0;
            maxPositionHoursEl.value = Math.round(hours * 3600);
            console.log('[BotsManager] ⏰ Макс. время позиции (сек):', maxPositionHoursEl.value);
        }
        
        const breakEvenProtectionEl = document.getElementById('breakEvenProtection');
        if (breakEvenProtectionEl) {
            breakEvenProtectionEl.checked = autoBotConfig.break_even_protection !== false;
            console.log('[BotsManager] 🛡️ Защита безубыточности:', breakEvenProtectionEl.checked);
        }
        
        const breakEvenTriggerEl = document.getElementById('breakEvenTrigger');
        if (breakEvenTriggerEl) {
            // ✅ ИСПОЛЬЗУЕМ РЕАЛЬНОЕ ЗНАЧЕНИЕ ИЗ КОНФИГА, А НЕ ДЕФОЛТНОЕ
            const triggerValue = autoBotConfig.break_even_trigger_percent ?? autoBotConfig.break_even_trigger ?? 20.0;
            breakEvenTriggerEl.value = triggerValue;
            console.log('[BotsManager] 🎯 Триггер безубыточности:', breakEvenTriggerEl.value, '(из конфига:', autoBotConfig.break_even_trigger_percent ?? autoBotConfig.break_even_trigger, ')');
        }
        
        // ==========================================
        // ФИЛЬТРЫ ПО ТРЕНДУ
        // ==========================================
        
        const avoidDownTrendEl = document.getElementById('avoidDownTrend');
        if (avoidDownTrendEl) {
            // ✅ ИСПОЛЬЗУЕМ РЕАЛЬНОЕ ЗНАЧЕНИЕ ИЗ КОНФИГА, А НЕ ДЕФОЛТНОЕ
            const configValue = autoBotConfig.avoid_down_trend;
            avoidDownTrendEl.checked = configValue === true;
            console.log('[BotsManager] 📉 Избегать DOWN тренд:', avoidDownTrendEl.checked, '(из конфига:', configValue, ')');
        }
        
        const avoidUpTrendEl = document.getElementById('avoidUpTrend');
        if (avoidUpTrendEl) {
            // ✅ ИСПОЛЬЗУЕМ РЕАЛЬНОЕ ЗНАЧЕНИЕ ИЗ КОНФИГА, А НЕ ДЕФОЛТНОЕ
            const configValue = autoBotConfig.avoid_up_trend;
            avoidUpTrendEl.checked = configValue === true;
            console.log('[BotsManager] 📈 Избегать UP тренд:', avoidUpTrendEl.checked, '(из конфига:', configValue, ')');
        }
        
        // ==========================================
        // ПАРАМЕТРЫ АНАЛИЗА ТРЕНДА
        // ==========================================
        
        const trendDetectionEnabledEl = document.getElementById('trendDetectionEnabled');
        if (trendDetectionEnabledEl) {
            // ✅ ИСПОЛЬЗУЕМ РЕАЛЬНОЕ ЗНАЧЕНИЕ ИЗ КОНФИГА, А НЕ ДЕФОЛТНОЕ
            const configValue = autoBotConfig.trend_detection_enabled;
            trendDetectionEnabledEl.checked = configValue === true;
            console.log('[BotsManager] 🔍 Анализ трендов включен:', trendDetectionEnabledEl.checked, '(из конфига:', configValue, ')');
        }
        
        const trendAnalysisPeriodEl = document.getElementById('trendAnalysisPeriod');
        if (trendAnalysisPeriodEl && autoBotConfig.trend_analysis_period !== undefined) {
            trendAnalysisPeriodEl.value = autoBotConfig.trend_analysis_period;
            console.log('[BotsManager] 📊 Период анализа тренда:', trendAnalysisPeriodEl.value);
        }
        
        const trendPriceChangeThresholdEl = document.getElementById('trendPriceChangeThreshold');
        if (trendPriceChangeThresholdEl && autoBotConfig.trend_price_change_threshold !== undefined) {
            trendPriceChangeThresholdEl.value = autoBotConfig.trend_price_change_threshold;
            console.log('[BotsManager] 📈 Порог изменения цены:', trendPriceChangeThresholdEl.value);
        }
        
        const trendCandlesThresholdEl = document.getElementById('trendCandlesThreshold');
        if (trendCandlesThresholdEl && autoBotConfig.trend_candles_threshold !== undefined) {
            trendCandlesThresholdEl.value = autoBotConfig.trend_candles_threshold;
            console.log('[BotsManager] 🕯️ Порог свечей:', trendCandlesThresholdEl.value);
        }
        
        // ==========================================
        // СИСТЕМНЫЕ НАСТРОЙКИ
        // ==========================================
        const systemConfig = config.system || {};
        
        // Загружаем таймфрейм в select
        const timeframeSelect = document.getElementById('systemTimeframe');
        if (timeframeSelect && systemConfig.timeframe) {
            timeframeSelect.value = systemConfig.timeframe;
            const applyBtn = document.getElementById('applyTimeframeBtn');
            if (applyBtn) {
                applyBtn.dataset.currentTimeframe = systemConfig.timeframe;
            }
            console.log('[BotsManager] ⏱️ Таймфрейм загружен:', systemConfig.timeframe);
        }
        
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
        
        // Миниграфики = интервал «Синхронизация позиций» (настройка убрана из UI). Автообновление UI всегда вкл.
        
        // Режим отладки
        const debugModeEl = document.getElementById('debugMode');
        if (debugModeEl) {
            debugModeEl.checked = systemConfig.debug_mode || false;
            console.log('[BotsManager] 🐛 Режим отладки:', debugModeEl.checked);
        }
        
        // Режим маржи Bybit (auto / cross / isolated)
        const bybitMarginModeEl = document.getElementById('bybitMarginMode');
        if (bybitMarginModeEl && systemConfig.bybit_margin_mode !== undefined) {
            const val = (systemConfig.bybit_margin_mode || 'auto').toLowerCase();
            bybitMarginModeEl.value = ['auto', 'cross', 'isolated'].includes(val) ? val : 'auto';
            console.log('[BotsManager] 📊 Режим маржи Bybit:', bybitMarginModeEl.value);
        } else if (bybitMarginModeEl) {
            bybitMarginModeEl.value = 'auto';
        }
        
        // ==========================================
        // ИНТЕРВАЛЫ СИНХРОНИЗАЦИИ И ОЧИСТКИ
        // ==========================================
        // Единый период обновления для всех RSI-зависимых данных в UI (боты, списки, фильтры, мониторинг) = position_sync_interval
        const positionSyncIntervalEl = document.getElementById('positionSyncInterval');
        console.log('[BotsManager] 🔍 Поиск элемента positionSyncInterval:', positionSyncIntervalEl);
        console.log('[BotsManager] 🔍 systemConfig.position_sync_interval:', systemConfig.position_sync_interval);
        if (positionSyncIntervalEl && systemConfig.position_sync_interval !== undefined) {
            positionSyncIntervalEl.value = systemConfig.position_sync_interval;
            // Минимум 5 сек — иначе интерфейс мигает как стробоскоп
            this.refreshInterval = Math.max(5000, systemConfig.position_sync_interval * 1000);
            console.log('[BotsManager] 🔄 Синхронизация позиций и период обновления UI (RSI, боты, мониторинг):', systemConfig.position_sync_interval, 'сек');
        } else if (positionSyncIntervalEl) {
            positionSyncIntervalEl.value = 600;
            this.refreshInterval = 600 * 1000;
            console.log('[BotsManager] 🔄 Position Sync и обновление UI по умолчанию: 600 сек');
        } else {
            console.error('[BotsManager] ❌ Элемент positionSyncInterval не найден!');
            this.refreshInterval = 600 * 1000;
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
        const exitScamAutoLearnEl = document.getElementById('exitScamAutoLearnEnabled');
        if (exitScamAutoLearnEl) {
            exitScamAutoLearnEl.checked = autoBotConfig.exit_scam_auto_learn_enabled === true;
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
        const exitScamTimeframeEl = document.getElementById('exitScamTimeframe');
        if (exitScamTimeframeEl) {
            const tf = autoBotConfig.exit_scam_timeframe || '1m';
            exitScamTimeframeEl.value = tf;
        }
        const exitScamEffectiveScaleEl = document.getElementById('exitScamEffectiveScale');
        if (exitScamEffectiveScaleEl) {
            const single = autoBotConfig.exit_scam_effective_single_pct ?? autoBotConfig.exit_scam_single_candle_percent ?? 15;
            const multi = autoBotConfig.exit_scam_effective_multi_pct ?? autoBotConfig.exit_scam_multi_candle_percent ?? 50;
            const n = autoBotConfig.exit_scam_multi_candle_count || 4;
            exitScamEffectiveScaleEl.textContent = `Одна свеча: ${Number(single)}% | суммарно за ${n} св.: ${Number(multi)}% (как в конфиге)`;
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
        // НАБОР ПОЗИЦИЙ ЛИМИТНЫМИ ОРДЕРАМИ
        // ==========================================
        
        const limitOrdersEnabledEl = document.getElementById('limitOrdersEntryEnabled');
        // Используем уже объявленные переменные positionSizeEl и positionModeEl из блока торговых параметров
        const limitPositionSizeEl = document.getElementById('defaultPositionSize');
        const limitPositionModeEl = document.getElementById('defaultPositionMode');
        
        if (limitOrdersEnabledEl) {
            const isEnabled = autoBotConfig.limit_orders_entry_enabled || false;
            // ✅ Устанавливаем значение БЕЗ триггера события change (чтобы не сработало автосохранение)
            // Используем прямую установку свойства, а не событие
            limitOrdersEnabledEl.checked = isEnabled;
            
            // ✅ Вручную обновляем UI без триггера события change
            const configDiv = document.getElementById('limitOrdersConfig');
            if (configDiv) {
                configDiv.style.display = isEnabled ? 'block' : 'none';
            }
            
            // Деактивируем настройку "Размер позиции" при включении лимитных ордеров
            if (limitPositionSizeEl) {
                limitPositionSizeEl.disabled = isEnabled;
                limitPositionSizeEl.style.opacity = isEnabled ? '0.5' : '1';
                limitPositionSizeEl.style.cursor = isEnabled ? 'not-allowed' : 'text';
            }
            if (limitPositionModeEl) {
                limitPositionModeEl.disabled = isEnabled;
                limitPositionModeEl.style.opacity = isEnabled ? '0.5' : '1';
                limitPositionModeEl.style.cursor = isEnabled ? 'not-allowed' : 'pointer';
            }
            
            // ✅ Обновляем состояние кнопки "По умолчанию"
            const resetBtn = document.getElementById('resetLimitOrdersBtn');
            if (resetBtn) {
                resetBtn.disabled = !isEnabled;
                resetBtn.style.opacity = isEnabled ? '1' : '0.5';
                resetBtn.style.cursor = isEnabled ? 'pointer' : 'not-allowed';
            }
            
            console.log('[BotsManager] 📊 Набор позиций лимитными ордерами:', isEnabled);
        }
        
        // Загружаем настройки лимитных ордеров
        const percentSteps = autoBotConfig.limit_orders_percent_steps || [1, 2, 3, 4, 5];
        const marginAmounts = autoBotConfig.limit_orders_margin_amounts || [5, 5, 5, 5, 5];
        const listEl = document.getElementById('limitOrdersList');
        if (listEl) {
            // ✅ Инициализируем UI ПЕРЕД загрузкой данных, но ПОСЛЕ установки значения toggle
            // Это гарантирует, что обработчики установлены, но не перезаписывают значение
            try {
                this.initializeLimitOrdersUI();
            } catch (e) {
                console.warn('[BotsManager] ⚠️ Ошибка инициализации UI лимитных ордеров:', e);
            }
            
            // ✅ Убеждаемся, что значение toggle не изменилось после инициализации UI
            if (limitOrdersEnabledEl) {
                const currentEnabled = limitOrdersEnabledEl.checked;
                const shouldBeEnabled = autoBotConfig.limit_orders_entry_enabled || false;
                if (currentEnabled !== shouldBeEnabled) {
                    // Если значение изменилось, восстанавливаем его
                    limitOrdersEnabledEl.checked = shouldBeEnabled;
                    const configDiv = document.getElementById('limitOrdersConfig');
                    if (configDiv) {
                        configDiv.style.display = shouldBeEnabled ? 'block' : 'none';
                    }
                }
            }
            
            listEl.innerHTML = ''; // Очищаем список
            for (let i = 0; i < Math.max(percentSteps.length, marginAmounts.length); i++) {
                try {
                    this.addLimitOrderRow(
                        percentSteps[i] || 0,
                        marginAmounts[i] || 0
                    );
                } catch (e) {
                    console.warn('[BotsManager] ⚠️ Ошибка добавления строки лимитного ордера:', e);
                }
            }
        }
        
        // ==========================================
        // ПАРАМЕТРЫ ОПРЕДЕЛЕНИЯ ТРЕНДА
        // ==========================================
        
        // ❌ УСТАРЕВШИЕ НАСТРОЙКИ EMA - УБРАНЫ (больше не используются)
        // Тренд теперь определяется простым анализом цены (% изменения и растущие/падающие свечи)
        
        // Сбрасываем флаг программного изменения после заполнения формы
        // Используем setTimeout чтобы гарантировать, что все события завершились
        setTimeout(() => {
            this.isProgrammaticChange = false;
        }, 100);
        
        console.log('[BotsManager] ✅ Форма заполнена данными из API');
    }
    
    // ==========================================
    // ИНДИКАТОР ЗАГРУЗКИ КОНФИГУРАЦИИ
    // ==========================================
    
    showConfigurationLoading(show) {
        // ✅ БЕЗ БЛОКИРОВКИ: Просто логируем, но не блокируем элементы
        const configContainer = document.getElementById('configTab');
        if (!configContainer) return;
        
        if (show) {
            // Добавляем класс загрузки для визуального индикатора
            configContainer.classList.add('loading');
            console.log('[BotsManager] ⏳ Конфигурация загружается...');
        } else {
            // Убираем класс загрузки
            configContainer.classList.remove('loading');
            console.log('[BotsManager] ✅ Конфигурация загружена');
            
            // ✅ КРИТИЧЕСКИ ВАЖНО: Убеждаемся что все элементы разблокированы
            const allInputs = configContainer.querySelectorAll('input, select, textarea, button');
            allInputs.forEach(el => {
                el.removeAttribute('disabled');
                el.disabled = false;
                el.style.pointerEvents = 'auto';
                el.style.opacity = '1';
                el.style.cursor = 'pointer';
            });
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
            // ✅ Проверяем, что есть данные для отправки
            if (!defaultConfig.autoBot || Object.keys(defaultConfig.autoBot).length === 0) {
                console.log('[BotsManager] ⚠️ Auto Bot конфигурация пуста, пропускаем сохранение');
            } else {
                // Сохраняем Auto Bot настройки
                const autoBotResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(defaultConfig.autoBot)
                });
                
                const autoBotData = await autoBotResponse.json();
                if (autoBotData.success) {
                    console.log('[BotsManager] ✅ Auto Bot конфигурация сохранена');
                }
            }
            
            // ✅ Проверяем, что есть данные для отправки
            if (!defaultConfig.system || Object.keys(defaultConfig.system).length === 0) {
                console.log('[BotsManager] ⚠️ System конфигурация пуста, пропускаем сохранение');
            } else {
                // Сохраняем системные настройки
                const systemResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(defaultConfig.system)
                });
                
                const systemData = await systemResponse.json();
                if (systemData.success) {
                    console.log('[BotsManager] ✅ System конфигурация сохранена');
                }
            }
            
            console.log('[BotsManager] ✅ Конфигурация по умолчанию обработана');
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения конфигурации по умолчанию:', error);
            throw error;
        }
    }
    /**
     * Конвертирует camelCase в snake_case для маппинга ID элементов на ключи конфигурации
     */
    camelToSnake(str) {
        return str.replace(/[A-Z]/g, letter => `_${letter.toLowerCase()}`);
    }
    
    /**
     * Автоматически маппит ID элемента на ключ конфигурации
     */
    mapElementIdToConfigKey(elementId) {
        // Прямые маппинги для элементов с нестандартными ID
        const directMappings = {
            'globalAutoBotToggle': 'enabled',
            'autoBotMaxConcurrent': 'max_concurrent',
            'autoBotRiskCap': 'risk_cap_percent',
            'autoBotScope': 'scope',  // ✅ КРИТИЧЕСКИ ВАЖНО: маппинг для scope
            'aiEnabled': 'ai_enabled',  // мастер-переключатель AI
            'aiMinConfidence': 'ai_min_confidence',
            'aiOverrideOriginal': 'ai_override_original',
            'fullAiControlToggle': 'full_ai_control',  // полный режим ИИ на вкладке Управление
            'fullAiControlToggleConfig': 'full_ai_control',  // дубль на вкладке Конфигурация
            'rsiLongThreshold': 'rsi_long_threshold',
            'rsiShortThreshold': 'rsi_short_threshold',
            'rsiExitLongWithTrendGlobal': 'rsi_exit_long_with_trend',
            'rsiExitLongAgainstTrendGlobal': 'rsi_exit_long_against_trend',
            'rsiExitShortWithTrendGlobal': 'rsi_exit_short_with_trend',
            'rsiExitShortAgainstTrendGlobal': 'rsi_exit_short_against_trend',
            'rsiExitMinCandlesGlobal': 'rsi_exit_min_candles',
            'rsiExitMinMinutesGlobal': 'rsi_exit_min_minutes',
            'rsiExitMinMovePercentGlobal': 'rsi_exit_min_move_percent',
            'exitWaitBreakevenWhenLoss': 'exit_wait_breakeven_when_loss',
            'rsiLimitEntryEnabled': 'rsi_limit_entry_enabled',
            'rsiLimitExitEnabled': 'rsi_limit_exit_enabled',
            'rsiLimitExitOffsetPercentGlobal': 'rsi_limit_exit_offset_percent',
            'rsiLimitOffsetPercentGlobal': 'rsi_limit_offset_percent',
            'defaultPositionSize': 'default_position_size',
            'defaultPositionMode': 'default_position_mode',
            'leverage': 'leverage',
            'checkInterval': 'check_interval',
            'maxLossPercent': 'max_loss_percent',
            'takeProfitPercent': 'take_profit_percent',
            'closeAtProfitEnabled': 'close_at_profit_enabled',
            'trailingStopActivation': 'trailing_stop_activation',
            'trailingStopDistance': 'trailing_stop_distance',
            'trailingTakeDistance': 'trailing_take_distance',
            'trailingUpdateInterval': 'trailing_update_interval',
            'maxPositionHours': 'max_position_hours',
            'breakEvenProtection': 'break_even_protection',
            'breakEvenTrigger': 'break_even_trigger_percent',
            'lossReentryProtection': 'loss_reentry_protection',
            'lossReentryCount': 'loss_reentry_count',
            'lossReentryCandles': 'loss_reentry_candles',
            'avoidDownTrend': 'avoid_down_trend',
            'avoidUpTrend': 'avoid_up_trend',
            'trendDetectionEnabled': 'trend_detection_enabled',
            'trendAnalysisPeriod': 'trend_analysis_period',
            'trendPriceChangeThreshold': 'trend_price_change_threshold',
            'trendCandlesThreshold': 'trend_candles_threshold',
            'enableMaturityCheck': 'enable_maturity_check',
            'minCandlesForMaturity': 'min_candles_for_maturity',
            'minRsiLow': 'min_rsi_low',
            'maxRsiHigh': 'max_rsi_high',
            'minVolatilityThreshold': 'min_volatility_threshold',
            'rsiTimeFilterEnabled': 'rsi_time_filter_enabled',
            'rsiTimeFilterCandles': 'rsi_time_filter_candles',
            'rsiTimeFilterUpper': 'rsi_time_filter_upper',
            'rsiTimeFilterLower': 'rsi_time_filter_lower',
            'exitScamEnabled': 'exit_scam_enabled',
            'exitScamCandles': 'exit_scam_candles',
            'exitScamSingleCandlePercent': 'exit_scam_single_candle_percent',
            'exitScamMultiCandleCount': 'exit_scam_multi_candle_count',
            'exitScamMultiCandlePercent': 'exit_scam_multi_candle_percent',
            'exitScamTimeframe': 'exit_scam_timeframe',
            'exitScamAutoLearnEnabled': 'exit_scam_auto_learn_enabled',
            'tradingEnabled': 'trading_enabled',
            'useTestServer': 'use_test_server',
            'enhancedRsiEnabled': 'enhanced_rsi_enabled',
            'enhancedRsiVolumeConfirm': 'enhanced_rsi_require_volume_confirmation',
            'enhancedRsiDivergenceConfirm': 'enhanced_rsi_require_divergence_confirmation',
            'enhancedRsiUseStochRsi': 'enhanced_rsi_use_stoch_rsi',
            'rsiExtremeZoneTimeout': 'rsi_extreme_zone_timeout',
            'rsiExtremeOversold': 'rsi_extreme_oversold',
            'rsiExtremeOverbought': 'rsi_extreme_overbought',
            'rsiVolumeMultiplier': 'rsi_volume_confirmation_multiplier',
            'rsiDivergenceLookback': 'rsi_divergence_lookback',
            'limitOrdersEntryEnabled': 'limit_orders_entry_enabled',
            'optimalEntryEnabled': 'ai_optimal_entry_enabled',
            'rsiUpdateInterval': 'rsi_update_interval',
            'autoSaveInterval': 'auto_save_interval',
            'debugMode': 'debug_mode',
            'positionSyncInterval': 'position_sync_interval',
            'inactiveBotCleanupInterval': 'inactive_bot_cleanup_interval',
            'inactiveBotTimeout': 'inactive_bot_timeout',
            'stopLossSetupInterval': 'stop_loss_setup_interval',
            'bybitMarginMode': 'bybit_margin_mode'
        };
        
        // Используем прямое маппинг если есть
        if (directMappings[elementId]) {
            return directMappings[elementId];
        }
        
        // Иначе конвертируем camelCase в snake_case
        return this.camelToSnake(elementId);
    }
    
    collectConfigurationData() {
        console.log('[BotsManager] 📋 Сбор данных конфигурации (автоматический режим)...');
        
        // ✅ РАБОТАЕМ НАПРЯМУЮ С КЭШИРОВАННОЙ КОНФИГУРАЦИЕЙ ИЗ БЭКЕНДА
        // Это гарантирует, что мы используем реальные значения из файла конфига, а не дефолтные из HTML
        if (!this.cachedAutoBotConfig) {
            console.warn('[BotsManager] ⚠️ cachedAutoBotConfig не загружен, используем пустой объект');
            return {
                autoBot: {},
                system: {}
            };
        }
        
        // ✅ ГЛУБОКОЕ КОПИРОВАНИЕ КЭШИРОВАННОЙ КОНФИГУРАЦИИ
        const autoBotConfig = JSON.parse(JSON.stringify(this.cachedAutoBotConfig));
        if (!autoBotConfig.default_position_mode) {
            autoBotConfig.default_position_mode = 'usdt';
        }
        
        // ✅ АВТОМАТИЧЕСКИЙ СБОР ВСЕХ ПОЛЕЙ КОНФИГУРАЦИИ
        const configTab = document.getElementById('configTab');
        if (!configTab) {
            console.warn('[BotsManager] ⚠️ configTab не найден');
            return { autoBot: autoBotConfig, system: {} };
        }
        
        // Находим ВСЕ поля конфигурации: input, select, checkbox
        // ✅ КРИТИЧЕСКИ ВАЖНО: Включаем скрытые input (hidden) для scope
        const autoBotInputs = configTab.querySelectorAll('input[type="number"], input[type="text"], input[type="hidden"], input[type="checkbox"], select');
        
        // Также добавляем поля из секции AI, если она существует
        const aiConfigSection = document.getElementById('aiConfigSection');
        if (aiConfigSection) {
            const aiInputs = aiConfigSection.querySelectorAll('input[type="number"], input[type="text"], input[type="hidden"], input[type="checkbox"], select');
            const uniqueInputs = new Set([...autoBotInputs, ...aiInputs]);
            this.collectFieldsFromElements(Array.from(uniqueInputs), autoBotConfig);
        } else {
            this.collectFieldsFromElements(Array.from(autoBotInputs), autoBotConfig);
        }
        
        // ✅ ОБРАБОТКА ДИНАМИЧЕСКИХ ПОЛЕЙ ЛИМИТНЫХ ОРДЕРОВ
        // Сначала обрабатываем toggle для limit_orders_entry_enabled
        const limitOrdersEntryEnabledEl = document.getElementById('limitOrdersEntryEnabled');
        if (limitOrdersEntryEnabledEl) {
            const enabled = limitOrdersEntryEnabledEl.checked;
            // Всегда обновляем значение, чтобы оно сохранялось при обычном сохранении конфигурации
            autoBotConfig.limit_orders_entry_enabled = enabled;
            console.log('[BotsManager] 🔄 Обновлен limit_orders_entry_enabled:', enabled);
        }
        // ✅ ExitScam: всегда берём из DOM, чтобы после перезапуска сервера сохранялось отключение
        const exitScamEnabledEl = document.getElementById('exitScamEnabled');
        const exitScamCandlesEl = document.getElementById('exitScamCandles');
        const exitScamSingleEl = document.getElementById('exitScamSingleCandlePercent');
        const exitScamMultiCountEl = document.getElementById('exitScamMultiCandleCount');
        const exitScamMultiPercentEl = document.getElementById('exitScamMultiCandlePercent');
        if (exitScamEnabledEl) {
            autoBotConfig.exit_scam_enabled = exitScamEnabledEl.checked;
        }
        if (exitScamCandlesEl && exitScamCandlesEl.value !== '') {
            const v = parseInt(exitScamCandlesEl.value, 10);
            if (!isNaN(v)) autoBotConfig.exit_scam_candles = v;
        }
        if (exitScamSingleEl && exitScamSingleEl.value !== '') {
            const v = parseFloat(exitScamSingleEl.value);
            if (!isNaN(v)) autoBotConfig.exit_scam_single_candle_percent = v;
        }
        if (exitScamMultiCountEl && exitScamMultiCountEl.value !== '') {
            const v = parseInt(exitScamMultiCountEl.value, 10);
            if (!isNaN(v)) autoBotConfig.exit_scam_multi_candle_count = v;
        }
        if (exitScamMultiPercentEl && exitScamMultiPercentEl.value !== '') {
            const v = parseFloat(exitScamMultiPercentEl.value);
            if (!isNaN(v)) autoBotConfig.exit_scam_multi_candle_percent = v;
        }
        const exitScamAutoLearnEl = document.getElementById('exitScamAutoLearnEnabled');
        if (exitScamAutoLearnEl) {
            autoBotConfig.exit_scam_auto_learn_enabled = exitScamAutoLearnEl.checked;
        }
        // ✅ КРИТИЧНО: exit_wait_breakeven_when_loss — всегда из DOM (иначе не сохраняется при переключении view)
        const exitWaitBreakevenEl = document.getElementById('exitWaitBreakevenWhenLoss');
        if (exitWaitBreakevenEl) {
            autoBotConfig.exit_wait_breakeven_when_loss = exitWaitBreakevenEl.checked;
        }
        const rsiLimitEntryEl = document.getElementById('rsiLimitEntryEnabled');
        if (rsiLimitEntryEl) {
            autoBotConfig.rsi_limit_entry_enabled = rsiLimitEntryEl.checked;
        }
        const rsiLimitOffsetEl = document.getElementById('rsiLimitOffsetPercentGlobal');
        if (rsiLimitOffsetEl && rsiLimitOffsetEl.value !== '') {
            const v = parseFloat(rsiLimitOffsetEl.value);
            if (!isNaN(v) && v >= 0) autoBotConfig.rsi_limit_offset_percent = v;
        }
        const rsiLimitExitEl = document.getElementById('rsiLimitExitEnabled');
        if (rsiLimitExitEl) {
            autoBotConfig.rsi_limit_exit_enabled = rsiLimitExitEl.checked;
        }
        const rsiLimitExitOffsetEl = document.getElementById('rsiLimitExitOffsetPercentGlobal');
        if (rsiLimitExitOffsetEl && rsiLimitExitOffsetEl.value !== '') {
            const v = parseFloat(rsiLimitExitOffsetEl.value);
            if (!isNaN(v) && v >= 0) autoBotConfig.rsi_limit_exit_offset_percent = v;
        }
        
        const limitOrderRows = document.querySelectorAll('.limit-order-row');
        if (limitOrderRows.length > 0) {
            const percentSteps = [];
            const marginAmounts = [];
            
            limitOrderRows.forEach(row => {
                const percentEl = row.querySelector('.limit-order-percent');
                const marginEl = row.querySelector('.limit-order-margin');
                
                if (percentEl) {
                    const percent = parseFloat(percentEl.value);
                    if (!isNaN(percent)) {
                        percentSteps.push(percent);
                    } else {
                        percentSteps.push(0); // Добавляем 0 если значение невалидно
                    }
                }
                
                if (marginEl) {
                    const margin = parseFloat(marginEl.value);
                    if (!isNaN(margin)) {
                        marginAmounts.push(margin);
                    } else {
                        marginAmounts.push(0); // Добавляем 0 если значение невалидно
                    }
                }
            });
            
            // ✅ ВСЕГДА обновляем значения лимитных ордеров (для автосохранения)
            // Это гарантирует, что изменения сохраняются даже если originalConfig не обновлен
            if (percentSteps.length > 0 || marginAmounts.length > 0) {
                autoBotConfig.limit_orders_percent_steps = percentSteps;
                autoBotConfig.limit_orders_margin_amounts = marginAmounts;
                console.log('[BotsManager] 🔄 Обновлены настройки лимитных ордеров:', { percentSteps, marginAmounts });
            }
        }
        
        // ✅ СБОР СИСТЕМНЫХ НАСТРОЕК (автоматически из системных полей)
        const systemConfig = {};
        
        // ✅ Список системных настроек Enhanced RSI и других системных настроек
        const systemConfigKeys = [
            'enhanced_rsi_enabled',
            'enhanced_rsi_require_volume_confirmation',
            'enhanced_rsi_require_divergence_confirmation',
            'enhanced_rsi_use_stoch_rsi',
            'rsi_extreme_zone_timeout',
            'rsi_extreme_oversold',
            'rsi_extreme_overbought',
            'rsi_volume_confirmation_multiplier',
            'rsi_divergence_lookback',
            'rsi_update_interval',
            'auto_save_interval',
            'debug_mode',
            'refresh_interval',
            'position_sync_interval',
            'inactive_bot_cleanup_interval',
            'inactive_bot_timeout',
            'stop_loss_setup_interval'
        ];
        
        // ✅ Находим все системные поля в configTab (используем более надежный подход)
        // Сначала собираем все поля Enhanced RSI по конкретным ID
        const enhancedRsiFields = [
            'enhancedRsiEnabled',
            'enhancedRsiVolumeConfirm',
            'enhancedRsiDivergenceConfirm',
            'enhancedRsiUseStochRsi',
            'rsiExtremeZoneTimeout',
            'rsiExtremeOversold',
            'rsiExtremeOverbought',
            'rsiVolumeMultiplier',
            'rsiDivergenceLookback'
        ];
        
        enhancedRsiFields.forEach(fieldId => {
            const element = document.getElementById(fieldId);
            if (element && !element.closest('#limitOrdersList') && !element.closest('.limit-order-row')) {
                const configKey = this.mapElementIdToConfigKey(fieldId);
                if (configKey && systemConfigKeys.includes(configKey)) {
                    let value;
                    if (element.type === 'checkbox') {
                        value = element.checked;
                    } else if (element.type === 'number') {
                        const numValue = parseFloat(element.value);
                        value = isNaN(numValue) ? undefined : numValue;
                    } else {
                        value = element.value;
                    }
                    
                    if (value !== undefined && value !== null) {
                        systemConfig[configKey] = value;
                        console.log(`[BotsManager] ✅ Собрана Enhanced RSI настройка ${configKey}:`, value);
                    }
                }
            }
        });
        
        // ✅ Находим остальные системные поля (интервалы, режимы и т.д.)
        // Используем селектор, который ищет по ID (нечувствительный к регистру через проверку)
        const allInputs = configTab.querySelectorAll('input, select');
        allInputs.forEach(element => {
            if (!element.id || element.closest('#limitOrdersList') || element.closest('.limit-order-row')) {
                return; // Пропускаем динамические поля лимитных ордеров
            }
            
            // Пропускаем поля Enhanced RSI, которые уже обработаны выше
            if (enhancedRsiFields.includes(element.id)) {
                return;
            }
            
            const configKey = this.mapElementIdToConfigKey(element.id);
            if (!configKey) {
                return;
            }
            
            // ✅ Проверяем, что это системная настройка (либо начинается с system_, либо в списке системных настроек)
            const isSystemConfig = configKey.startsWith('system_') || systemConfigKeys.includes(configKey);
            
            if (isSystemConfig) {
                const systemKey = configKey.startsWith('system_') ? configKey.replace('system_', '') : configKey;
                let value;
                if (element.type === 'checkbox') {
                    value = element.checked;
                } else if (element.type === 'number') {
                    const numValue = parseFloat(element.value);
                    value = isNaN(numValue) ? undefined : numValue;
                } else {
                    value = element.value;
                }
                
                if (value !== undefined && value !== null) {
                    systemConfig[systemKey] = value;
                    console.log(`[BotsManager] ✅ Собрана системная настройка ${systemKey}:`, value);
                }
            }
        });
        
        // Период обновления RSI/UI везде берётся из «Синхронизация позиций»
        if (systemConfig.position_sync_interval != null) {
            systemConfig.refresh_interval = systemConfig.position_sync_interval;
        }
        
        return {
            autoBot: autoBotConfig,
            system: systemConfig
        };
    }
    
    /**
     * Собирает значения из элементов формы и обновляет конфигурацию
     */
    collectFieldsFromElements(elements, config) {
        elements.forEach(element => {
            // Пропускаем кнопки и элементы управления
            if (element.type === 'button' || element.type === 'submit' || element.closest('button')) {
                return;
            }
            
            // Пропускаем элементы без ID (динамические поля лимитных ордеров обрабатываются отдельно)
            if (!element.id || element.classList.contains('limit-order-percent') || element.classList.contains('limit-order-margin')) {
                return;
            }
            
            const configKey = this.mapElementIdToConfigKey(element.id);
            if (!configKey) {
                return;
            }
            
            // Определяем значение в зависимости от типа элемента
            let value;
            if (element.type === 'checkbox') {
                value = element.checked;
            } else if (element.type === 'number') {
                const numValue = parseFloat(element.value);
                value = isNaN(numValue) ? undefined : numValue;
            } else if (element.tagName === 'SELECT') {
                value = element.value;
            } else {
                value = element.value;
            }
            
            // Применяем значение только если оно изменилось
            const originalValue = this.originalConfig?.autoBot?.[configKey];
            
            // ✅ Макс. время позиции: в UI в секундах, в конфиге — в часах
            if (configKey === 'max_position_hours' && typeof value === 'number') {
                value = value / 3600;
            }
            // ✅ КРИТИЧЕСКИ ВАЖНО: Специальная обработка для scope - всегда обновляем если значение изменилось
            if (configKey === 'scope') {
                if (value !== undefined && value !== null) {
                    config[configKey] = value;
                    console.log(`[BotsManager] 🔄 scope собран из UI: ${value} (было в originalConfig: ${originalValue || 'undefined'})`);
                }
                return; // Пропускаем остальную логику для scope
            }
            
            if (value !== undefined && value !== null) {
                // Если originalValue undefined (новое поле), всегда устанавливаем значение
                if (originalValue === undefined) {
                    config[configKey] = value;
                    console.log(`[BotsManager] 🔄 Авто-применено (новое поле): ${configKey} = ${value}`);
                }
                // Для булевых значений
                else if (typeof value === 'boolean') {
                    const normalizedOriginal = originalValue === true ? true : false;
                    if (value !== normalizedOriginal) {
                        config[configKey] = value;
                        console.log(`[BotsManager] 🔄 Авто-применено: ${configKey} = ${value} (было ${normalizedOriginal})`);
                    }
                }
                // Для чисел: сравниваем с точностью 0.01
                else if (typeof value === 'number' && typeof originalValue === 'number') {
                    if (Math.abs(value - originalValue) > 0.01) {
                        config[configKey] = value;
                        console.log(`[BotsManager] 🔄 Авто-применено: ${configKey} = ${value} (было ${originalValue})`);
                    }
                }
                // Для остальных типов: точное сравнение
                else if (value !== originalValue) {
                    config[configKey] = value;
                    console.log(`[BotsManager] 🔄 Авто-применено: ${configKey} = ${value} (было ${originalValue})`);
                }
            }
        });
    }

    // ✅ НОВЫЕ ФУНКЦИИ ДЛЯ СОХРАНЕНИЯ ОТДЕЛЬНЫХ БЛОКОВ
    
    async saveBasicSettings() {
        console.log('[BotsManager] 💾 Сохранение основных настроек...');
        try {
            // ✅ КРИТИЧЕСКИ ВАЖНО: Сначала получаем scope напрямую из UI
            const scopeInput = document.getElementById('autoBotScope');
            const scopeFromUI = scopeInput ? scopeInput.value : null;
            console.log('[BotsManager] 🔍 scope из UI (autoBotScope):', scopeFromUI);
            
            const config = this.collectConfigurationData();
            console.log('[BotsManager] 🔍 scope из collectConfigurationData():', config.autoBot.scope);
            
            // Полный Режим ИИ: тумблер на Управлении или дубль на Конфигурации
            const fullAiControlEl = document.getElementById('fullAiControlToggle');
            const fullAiControlConfigEl = document.getElementById('fullAiControlToggleConfig');
            const fullAiControl = (fullAiControlEl?.checked ?? fullAiControlConfigEl?.checked ?? config.autoBot.full_ai_control) === true;
            const basicSettings = {
                enabled: config.autoBot.enabled,
                max_concurrent: config.autoBot.max_concurrent,
                risk_cap_percent: config.autoBot.risk_cap_percent,
                scope: scopeFromUI || config.autoBot.scope || 'all',  // ✅ Приоритет UI значению
                ai_enabled: config.autoBot.ai_enabled,
                ai_min_confidence: config.autoBot.ai_min_confidence,
                ai_override_original: config.autoBot.ai_override_original,
                full_ai_control: fullAiControl
            };
            
            console.log('[BotsManager] 🔍 Основные настройки для сохранения:', basicSettings);
            console.log('[BotsManager] 🔍 originalConfig.autoBot.scope:', this.originalConfig?.autoBot?.scope);
            console.log('[BotsManager] 🔍 Сравнение scope: UI=' + basicSettings.scope + ', original=' + (this.originalConfig?.autoBot?.scope || 'undefined'));
            
            await this.sendConfigUpdate('auto-bot', basicSettings, 'Основные настройки');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения основных настроек:', error);
            this.showNotification('❌ Ошибка сохранения основных настроек: ' + error.message, 'error');
        }
    }
    
    _updateFullaiAdaptiveDependentFields() {
        const el = (id) => document.getElementById(id);
        const virtualSuccess = parseInt(el('fullaiAdaptiveVirtualSuccess')?.value, 10);
        const disabled = !Number.isFinite(virtualSuccess) || virtualSuccess <= 0;
        const ids = ['fullaiAdaptiveRealLoss', 'fullaiAdaptiveRoundSize', 'fullaiAdaptiveMaxFailures'];
        const groupIds = ['fullaiAdaptiveDependentGroup', 'fullaiAdaptiveDependentGroup2', 'fullaiAdaptiveDependentGroup3'];
        ids.forEach(id => { const i = el(id); if (i) i.disabled = disabled; });
        groupIds.forEach(id => { const g = el(id); if (g) g.style.opacity = disabled ? '0.6' : '1'; });
    }

    async loadFullaiAdaptiveConfig() {
        try {
            const res = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/fullai-config`, { method: 'GET' });
            const data = await res.json();
            if (!data.success || !data.config) return;
            const c = data.config;
            const el = (id) => document.getElementById(id);
            if (el('fullaiAdaptiveDeadCandles')) el('fullaiAdaptiveDeadCandles').value = c.fullai_adaptive_dead_candles ?? 100;
            if (el('fullaiAdaptiveVirtualSuccess')) el('fullaiAdaptiveVirtualSuccess').value = c.fullai_adaptive_virtual_success_count ?? 3;
            if (el('fullaiAdaptiveRealLoss')) el('fullaiAdaptiveRealLoss').value = c.fullai_adaptive_real_loss_to_retry ?? 1;
            if (el('fullaiAdaptiveRoundSize')) el('fullaiAdaptiveRoundSize').value = c.fullai_adaptive_virtual_round_size ?? 3;
            if (el('fullaiAdaptiveMaxFailures')) el('fullaiAdaptiveMaxFailures').value = c.fullai_adaptive_virtual_max_failures ?? 0;
            this._updateFullaiAdaptiveDependentFields();
        } catch (e) {
            console.warn('[BotsManager] loadFullaiAdaptiveConfig:', e);
        }
    }
    
    async saveFullaiAdaptiveConfig() {
        try {
            const el = (id) => document.getElementById(id);
            // Один переключатель: Full AI вкл → Adaptive вкл (второй выключатель убран)
            const fullAiOn = el('fullAiControlToggleConfig')?.checked ?? el('fullAiControlToggle')?.checked ?? false;
            const vs = parseInt(el('fullaiAdaptiveVirtualSuccess')?.value, 10);
            const payload = {
                fullai_adaptive_enabled: fullAiOn,
                fullai_adaptive_dead_candles: parseInt(el('fullaiAdaptiveDeadCandles')?.value, 10) || 100,
                fullai_adaptive_virtual_success_count: Number.isFinite(vs) ? vs : 3,
                fullai_adaptive_real_loss_to_retry: parseInt(el('fullaiAdaptiveRealLoss')?.value, 10) || 1,
                fullai_adaptive_virtual_round_size: parseInt(el('fullaiAdaptiveRoundSize')?.value, 10) || 3,
                fullai_adaptive_virtual_max_failures: parseInt(el('fullaiAdaptiveMaxFailures')?.value, 10) || 0
            };
            const res = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/fullai-config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                this.showNotification('Параметры Full AI сохранены', 'success');
            } else {
                this.showNotification('Ошибка сохранения параметров Full AI: ' + (data.error || res.status), 'error');
            }
        } catch (e) {
            console.error('[BotsManager] saveFullaiAdaptiveConfig:', e);
            this.showNotification('Ошибка сохранения параметров Full AI', 'error');
        }
    }
    
    async saveSystemSettings() {
        console.log('[BotsManager] 💾 Сохранение системных настроек...');
        try {
            const config = this.collectConfigurationData();
            const systemSettings = { ...config.system };
            const bybitMarginEl = document.getElementById('bybitMarginMode');
            if (bybitMarginEl) {
                const v = (bybitMarginEl.value || 'auto').toLowerCase();
                systemSettings.bybit_margin_mode = ['auto', 'cross', 'isolated'].includes(v) ? v : 'auto';
            }
            
            await this.sendConfigUpdate('system-config', systemSettings, 'Системные настройки');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения системных настроек:', error);
            this.showNotification('❌ Ошибка сохранения системных настроек', 'error');
        }
    }
    
    /**
     * Сохраняет весь блок: торговые параметры и RSI выходы (объединённая кнопка)
     */
    async saveTradingAndRsiExits() {
        console.log('[BotsManager] 💾 Сохранение торговых параметров и RSI выходов...');
        try {
            const config = this.collectConfigurationData();
            const params = {
                rsi_long_threshold: config.autoBot.rsi_long_threshold,
                rsi_short_threshold: config.autoBot.rsi_short_threshold,
                rsi_exit_long_with_trend: config.autoBot.rsi_exit_long_with_trend,
                rsi_exit_long_against_trend: config.autoBot.rsi_exit_long_against_trend,
                rsi_exit_short_with_trend: config.autoBot.rsi_exit_short_with_trend,
                rsi_exit_short_against_trend: config.autoBot.rsi_exit_short_against_trend,
                rsi_exit_min_candles: parseInt(config.autoBot.rsi_exit_min_candles, 10) || 0,
                rsi_exit_min_minutes: parseInt(config.autoBot.rsi_exit_min_minutes, 10) || 0,
                rsi_exit_min_move_percent: parseFloat(config.autoBot.rsi_exit_min_move_percent) || 0,
                exit_wait_breakeven_when_loss: (() => {
                    const el = document.getElementById('exitWaitBreakevenWhenLoss');
                    return el ? el.checked : (config.autoBot.exit_wait_breakeven_when_loss === true);
                })(),
                rsi_limit_entry_enabled: (() => {
                    const el = document.getElementById('rsiLimitEntryEnabled');
                    return el ? el.checked : (config.autoBot.rsi_limit_entry_enabled === true);
                })(),
                rsi_limit_offset_percent: (() => {
                    const el = document.getElementById('rsiLimitOffsetPercentGlobal');
                    if (el && el.value !== '') {
                        const v = parseFloat(el.value);
                        return !isNaN(v) && v >= 0 ? v : 0.2;
                    }
                    return parseFloat(config.autoBot.rsi_limit_offset_percent) || 0.2;
                })(),
                rsi_limit_exit_enabled: (() => {
                    const el = document.getElementById('rsiLimitExitEnabled');
                    return el ? el.checked : (config.autoBot.rsi_limit_exit_enabled === true);
                })(),
                rsi_limit_exit_offset_percent: (() => {
                    const el = document.getElementById('rsiLimitExitOffsetPercentGlobal');
                    if (el && el.value !== '') {
                        const v = parseFloat(el.value);
                        return !isNaN(v) && v >= 0 ? v : 0.2;
                    }
                    return parseFloat(config.autoBot.rsi_limit_exit_offset_percent) || 0.2;
                })(),
                default_position_size: config.autoBot.default_position_size,
                default_position_mode: config.autoBot.default_position_mode,
                leverage: config.autoBot.leverage,
                check_interval: config.autoBot.check_interval,
                trading_enabled: config.autoBot.trading_enabled,
                use_test_server: config.autoBot.use_test_server
            };
            await this.sendConfigUpdate('auto-bot', params, 'Торговые параметры и RSI выходы');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения:', error);
            this.showNotification('❌ Ошибка сохранения торговых параметров и RSI выходов', 'error');
        }
    }
    
    async saveRsiTimeFilter() {
        console.log('[BotsManager] 💾 Сохранение RSI временного фильтра...');
        try {
            const config = this.collectConfigurationData();
            const rsiTimeFilter = {
                rsi_time_filter_enabled: config.autoBot.rsi_time_filter_enabled,
                rsi_time_filter_candles: config.autoBot.rsi_time_filter_candles || 6,
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
            // ✅ Читаем значения напрямую из DOM, чтобы после перезапуска сервера сохранялось то, что в UI
            const exitScamEnabledEl = document.getElementById('exitScamEnabled');
            const exitScamCandlesEl = document.getElementById('exitScamCandles');
            const exitScamSingleEl = document.getElementById('exitScamSingleCandlePercent');
            const exitScamMultiCountEl = document.getElementById('exitScamMultiCandleCount');
            const exitScamMultiPercentEl = document.getElementById('exitScamMultiCandlePercent');
            const exitScamTimeframeEl = document.getElementById('exitScamTimeframe');
            const config = this.collectConfigurationData();
            const exitScamAutoLearnEl = document.getElementById('exitScamAutoLearnEnabled');
            const exitScamFilter = {
                exit_scam_enabled: exitScamEnabledEl ? exitScamEnabledEl.checked : (config.autoBot.exit_scam_enabled !== false),
                exit_scam_auto_learn_enabled: exitScamAutoLearnEl ? exitScamAutoLearnEl.checked : (config.autoBot.exit_scam_auto_learn_enabled === true),
                exit_scam_candles: exitScamCandlesEl && exitScamCandlesEl.value !== '' ? parseInt(exitScamCandlesEl.value, 10) : (config.autoBot.exit_scam_candles ?? 8),
                exit_scam_single_candle_percent: exitScamSingleEl && exitScamSingleEl.value !== '' ? parseFloat(exitScamSingleEl.value) : (config.autoBot.exit_scam_single_candle_percent ?? 15),
                exit_scam_multi_candle_count: exitScamMultiCountEl && exitScamMultiCountEl.value !== '' ? parseInt(exitScamMultiCountEl.value, 10) : (config.autoBot.exit_scam_multi_candle_count ?? 4),
                exit_scam_multi_candle_percent: exitScamMultiPercentEl && exitScamMultiPercentEl.value !== '' ? parseFloat(exitScamMultiPercentEl.value) : (config.autoBot.exit_scam_multi_candle_percent ?? 50),
                exit_scam_timeframe: exitScamTimeframeEl && exitScamTimeframeEl.value ? exitScamTimeframeEl.value : (config.autoBot.exit_scam_timeframe || '1m')
            };
            console.log('[BotsManager] 🔍 ExitScam из UI:', exitScamFilter.exit_scam_enabled, exitScamFilter.exit_scam_candles);
            await this.sendConfigUpdate('auto-bot', exitScamFilter, 'ExitScam фильтр');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения ExitScam фильтра:', error);
            this.showNotification('❌ Ошибка сохранения ExitScam фильтра', 'error');
        }
    }
    
    async saveEnhancedRsi() {
        console.log('[BotsManager] 💾 Сохранение Enhanced RSI...');
        try {
            // ✅ Сначала проверяем значения из UI напрямую
            const enhancedRsiEnabledEl = document.getElementById('enhancedRsiEnabled');
            const enhancedRsiVolumeConfirmEl = document.getElementById('enhancedRsiVolumeConfirm');
            const enhancedRsiDivergenceConfirmEl = document.getElementById('enhancedRsiDivergenceConfirm');
            const enhancedRsiUseStochRsiEl = document.getElementById('enhancedRsiUseStochRsi');
            
            console.log('[BotsManager] 🔍 Значения из UI напрямую:');
            console.log('  enhancedRsiEnabled:', enhancedRsiEnabledEl?.checked);
            console.log('  enhancedRsiVolumeConfirm:', enhancedRsiVolumeConfirmEl?.checked);
            console.log('  enhancedRsiDivergenceConfirm:', enhancedRsiDivergenceConfirmEl?.checked);
            console.log('  enhancedRsiUseStochRsi:', enhancedRsiUseStochRsiEl?.checked);
            
            const config = this.collectConfigurationData();
            console.log('[BotsManager] 🔍 Значения из collectConfigurationData():');
            console.log('  config.system:', config.system);
            
            const enhancedRsi = {
                enhanced_rsi_enabled: config.system.enhanced_rsi_enabled,
                enhanced_rsi_require_volume_confirmation: config.system.enhanced_rsi_require_volume_confirmation,
                enhanced_rsi_require_divergence_confirmation: config.system.enhanced_rsi_require_divergence_confirmation,
                enhanced_rsi_use_stoch_rsi: config.system.enhanced_rsi_use_stoch_rsi,
                rsi_extreme_zone_timeout: config.system.rsi_extreme_zone_timeout,
                rsi_extreme_oversold: config.system.rsi_extreme_oversold,
                rsi_extreme_overbought: config.system.rsi_extreme_overbought,
                rsi_volume_confirmation_multiplier: config.system.rsi_volume_confirmation_multiplier,
                rsi_divergence_lookback: config.system.rsi_divergence_lookback
            };
            
            console.log('[BotsManager] 📤 Отправляемые Enhanced RSI настройки:', enhancedRsi);
            
            await this.sendConfigUpdate('system-config', enhancedRsi, 'Enhanced RSI');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения Enhanced RSI:', error);
            this.showNotification('❌ Ошибка сохранения Enhanced RSI', 'error');
        }
    }
    async saveProtectiveMechanisms() {
        console.log('[BotsManager] 💾 Сохранение защитных механизмов...');
        try {
            const config = this.collectConfigurationData();
            
            const protectiveMechanisms = {
                max_loss_percent: config.autoBot.max_loss_percent,
                take_profit_percent: config.autoBot.take_profit_percent,
                close_at_profit_enabled: config.autoBot.close_at_profit_enabled !== false,
                trailing_stop_activation: config.autoBot.trailing_stop_activation,
                trailing_stop_distance: config.autoBot.trailing_stop_distance,
                trailing_take_distance: config.autoBot.trailing_take_distance,
                trailing_update_interval: config.autoBot.trailing_update_interval,
                max_position_hours: config.autoBot.max_position_hours,
                break_even_protection: config.autoBot.break_even_protection,
                break_even_trigger: config.autoBot.break_even_trigger,
                break_even_trigger_percent: config.autoBot.break_even_trigger_percent,
                loss_reentry_protection: config.autoBot.loss_reentry_protection !== false,
                loss_reentry_count: parseInt(config.autoBot.loss_reentry_count || 1),
                loss_reentry_candles: parseInt(config.autoBot.loss_reentry_candles || 3),
                avoid_down_trend: config.autoBot.avoid_down_trend,
                avoid_up_trend: config.autoBot.avoid_up_trend,
                // ✅ ПАРАМЕТРЫ АНАЛИЗА ТРЕНДА
                trend_detection_enabled: config.autoBot.trend_detection_enabled,
                trend_analysis_period: config.autoBot.trend_analysis_period,
                trend_price_change_threshold: config.autoBot.trend_price_change_threshold,
                trend_candles_threshold: config.autoBot.trend_candles_threshold
            };
            
            // sendConfigUpdate автоматически отфильтрует только измененные параметры
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
        // ❌ УСТАРЕВШИЕ НАСТРОЙКИ EMA - УБРАНЫ (больше не используются)
        // Тренд теперь определяется простым анализом цены - настройки не требуются
        this.showNotification('ℹ️ Настройки тренда больше не используются (тренд определяется автоматически по цене)', 'info');
    }

    hasUnsavedConfigChanges() {
        if (!this.originalConfig) return false;
        try {
            const config = this.collectConfigurationData();
            const autoBotChanges = this.filterChangedParams(config.autoBot || {}, 'autoBot');
            const systemChanges = this.filterChangedParams(config.system || {}, 'system');
            return Object.keys(autoBotChanges).length > 0 || Object.keys(systemChanges).length > 0 || this.aiConfigDirty;
        } catch (e) {
            return false;
        }
    }

    createFloatingSaveButton() {
        if (document.getElementById('floatingSaveConfigBtn')) return;
        const btn = document.createElement('button');
        btn.id = 'floatingSaveConfigBtn';
        btn.className = 'floating-save-config-btn';
        btn.innerHTML = '💾 ' + (this.getTranslation('save_all_config_btn') || 'Сохранить все настройки');
        btn.addEventListener('click', async () => {
            try {
                btn.disabled = true;
                await this.saveAllConfiguration();
            } finally {
                btn.disabled = false;
            }
        });
        document.body.appendChild(btn);
    }

    async saveAllConfiguration() {
        try {
            await this.saveConfiguration(false, true);
            if (window.aiConfigManager && typeof window.aiConfigManager.saveAIConfig === 'function') {
                await window.aiConfigManager.saveAIConfig(false, true);
            }
            this.aiConfigDirty = false;
            this.updateFloatingSaveButtonVisibility();
            this.showConfigNotification('✅ Сохранено', 'Все настройки сохранены', 'success');
        } catch (error) {
            console.error('[BotsManager] Ошибка при сохранении:', error);
            this.showConfigNotification('❌ Ошибка', 'Ошибка сохранения настроек: ' + error.message, 'error');
        }
    }

    hideFloatingSaveButton() {
        const btn = document.getElementById('floatingSaveConfigBtn');
        if (btn) btn.classList.remove('visible');
    }

    updateFloatingSaveButtonVisibility() {
        const btn = document.getElementById('floatingSaveConfigBtn');
        if (!btn) return;
        const configTab = document.getElementById('configTab');
        const isConfigTabActive = configTab && configTab.classList.contains('active');
        const botsContainer = document.getElementById('botsContainer');
        const isBotsPageVisible = botsContainer && botsContainer.style.display !== 'none';
        const hasChanges = this.hasUnsavedConfigChanges();
        if (isBotsPageVisible && isConfigTabActive) {
            btn.classList.add('visible');
            btn.disabled = !hasChanges;
        } else {
            btn.classList.remove('visible');
            btn.disabled = false;
        }
    }
    
    // ✅ ФИЛЬТРАЦИЯ ИЗМЕНЕННЫХ ПАРАМЕТРОВ
    filterChangedParams(data, configType = 'autoBot') {
        const originalGroup = configType === 'system'
            ? (this.originalConfig?.system)
            : (this.originalConfig?.autoBot);

        if (!originalGroup) {
            // Если нет исходной конфигурации, отправляем все данные
            console.log('[BotsManager] ⚠️ originalConfig не инициализирован, отправляем все параметры');
            return data;
        }
        
        const original = originalGroup;
        const filtered = {};
        let changedCount = 0;
        
        console.log(`[BotsManager] 🔍 filterChangedParams: сравниваем ${Object.keys(data).length} параметров`);
        // ✅ КРИТИЧЕСКИ ВАЖНО: Логируем scope для отладки
        if (data.scope !== undefined) {
            console.log(`[BotsManager] 🔍 SCOPE в data: "${data.scope}" (тип: ${typeof data.scope})`);
            console.log(`[BotsManager] 🔍 SCOPE в original: "${original.scope}" (тип: ${typeof original.scope})`);
            console.log(`[BotsManager] 🔍 SCOPE сравнение: ${data.scope} !== ${original.scope} = ${data.scope !== original.scope}`);
        }
        
        for (const [key, value] of Object.entries(data)) {
            const originalValue = original[key];
            
            // ✅ ОСОБАЯ ОБРАБОТКА ДЛЯ break_even_trigger_percent
            if (key === 'break_even_trigger_percent' && originalValue === undefined) {
                // Если в originalConfig нет break_even_trigger_percent, проверяем break_even_trigger
                const altOriginalValue = original['break_even_trigger'];
                if (altOriginalValue !== undefined) {
                    if (typeof value === 'number' && typeof altOriginalValue === 'number') {
                        if (Math.abs(value - altOriginalValue) > 0.01) {
                            filtered[key] = value;
                            changedCount++;
                            console.log(`[BotsManager] 🔄 Изменен ${key}: ${altOriginalValue} → ${value} (из break_even_trigger)`);
                        }
                    }
                } else {
                    // Если и break_even_trigger нет, считаем что значение изменилось
                    filtered[key] = value;
                    changedCount++;
                    console.log(`[BotsManager] 🔄 Изменен ${key}: undefined → ${value} (новый параметр)`);
                }
                continue;
            }
            
            // Для чисел: сравниваем с точностью 0.01
            if (typeof value === 'number' && typeof originalValue === 'number') {
                if (Math.abs(value - originalValue) > 0.01) {
                    filtered[key] = value;
                    changedCount++;
                    console.log(`[BotsManager] 🔄 Изменен ${key}: ${originalValue} → ${value}`);
                } else {
                    console.log(`[BotsManager] ⏭️ Пропущен ${key}: ${originalValue} == ${value} (не изменился)`);
                }
            }
            // Для булевых значений: точное сравнение
            else if (typeof value === 'boolean' && typeof originalValue === 'boolean') {
                if (value !== originalValue) {
                    filtered[key] = value;
                    changedCount++;
                    console.log(`[BotsManager] 🔄 Изменен ${key}: ${originalValue} → ${value}`);
                } else {
                    console.log(`[BotsManager] ⏭️ Пропущен ${key}: ${originalValue} == ${value} (не изменился)`);
                }
            }
            // ✅ ОСОБАЯ ОБРАБОТКА ДЛЯ scope - ВСЕГДА проверяем первым!
            else if (key === 'scope') {
                console.log(`[BotsManager] 🔍 [SCOPE] Сравнение scope: текущее="${value}" (тип: ${typeof value}), оригинальное="${originalValue}" (тип: ${typeof originalValue})`);
                console.log(`[BotsManager] 🔍 [SCOPE] Строгое сравнение: ${value} !== ${originalValue} = ${value !== originalValue}`);
                // ✅ КРИТИЧЕСКИ ВАЖНО: Для scope всегда проверяем изменение, даже если originalValue undefined
                if (originalValue === undefined || value !== originalValue) {
                    filtered[key] = value;
                    changedCount++;
                    console.log(`[BotsManager] ✅ [SCOPE] Изменен scope: ${originalValue || 'undefined'} → ${value} (ДОБАВЛЕН В ИЗМЕНЕННЫЕ!)`);
                } else {
                    console.log(`[BotsManager] ⏭️ [SCOPE] Пропущен scope: ${originalValue} == ${value} (не изменился)`);
                }
            }
            // Для остальных типов: точное сравнение
            else if (value !== originalValue) {
                filtered[key] = value;
                changedCount++;
                console.log(`[BotsManager] 🔄 Изменен ${key}: ${originalValue} → ${value}`);
            } else {
                console.log(`[BotsManager] ⏭️ Пропущен ${key}: ${originalValue} == ${value} (не изменился)`);
            }
        }
        
        console.log(`[BotsManager] 📊 Отфильтровано: ${changedCount} из ${Object.keys(data).length} параметров изменены`);
        // ✅ КРИТИЧЕСКИ ВАЖНО: Логируем scope в результате
        if (data.scope !== undefined) {
            if (filtered.scope !== undefined) {
                console.log(`[BotsManager] ✅ [SCOPE] scope ПОПАЛ В ОТПРАВЛЯЕМЫЕ ПАРАМЕТРЫ: "${filtered.scope}"`);
            } else {
                console.log(`[BotsManager] ❌ [SCOPE] scope НЕ ПОПАЛ В ОТПРАВЛЯЕМЫЕ ПАРАМЕТРЫ! data.scope="${data.scope}", original.scope="${original.scope}"`);
            }
        }
        if (changedCount > 0) {
            console.log(`[BotsManager] 📤 Отправляемые параметры:`, filtered);
        } else {
            console.log(`[BotsManager] ⚠️ НЕТ ИЗМЕНЕННЫХ ПАРАМЕТРОВ! Все ${Object.keys(data).length} параметров без изменений`);
        }
        return filtered;
    }
    
    // ✅ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ОТПРАВКИ КОНФИГУРАЦИИ
    // options: { forceSend: true } — не фильтровать по изменениям, всегда отправить (для тумблера FullAI)
    async sendConfigUpdate(endpoint, data, sectionName, options = {}) {
        // БЕЗ БЛОКИРОВКИ - элементы остаются активными!
        
        try {
            const configType = endpoint === 'system-config' ? 'system' : 'autoBot';
            const filteredData = options.forceSend ? data : this.filterChangedParams(data, configType);
            
            // Если нет изменений, не отправляем запрос (кроме forceSend)
            if (Object.keys(filteredData).length === 0) {
                console.log(`[BotsManager] ℹ️ Нет изменений в ${sectionName}, пропускаем отправку`);
                this.showNotification(`ℹ️ Нет изменений в ${sectionName}`, 'info');
                return;
            }
            
            console.log(`[BotsManager] 📤 Отправка ${sectionName}:`, filteredData);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(filteredData)
            });
            
            if (response.ok) {
                const responseData = await response.json();
                console.log(`[BotsManager] ✅ ${sectionName} сохранены успешно, ответ сервера:`, responseData);
                
                // ✅ Проверяем количество изменений из ответа сервера
                const changesCount = responseData.changes_count || 0;
                if (changesCount === 0) {
                    // Нет изменений - показываем соответствующее сообщение
                    this.showNotification(`ℹ️ Нет изменений в настройках`, 'info');
                } else {
                    // Есть изменения - показываем детальное сообщение из сервера
                    const message = responseData.message || `✅ ${sectionName} сохранены успешно`;
                    this.showNotification(message, 'success');
                    
                    // ✅ Логируем только измененные параметры
                    if (responseData.changed_params && responseData.changed_params.length > 0) {
                        console.log(`[BotsManager] 📋 Измененные параметры (${changesCount}):`, responseData.changed_params);
                    }
                }
                console.log(`[BotsManager] 🔔 Уведомление отправлено для ${sectionName}`);
                
                // ✅ ОБНОВЛЯЕМ originalConfig после успешного сохранения
                if (this.originalConfig) {
                    // Обновляем только сохраненные параметры
                    for (const [key, value] of Object.entries(filteredData)) {
                        if (configType === 'system') {
                            this.originalConfig.system[key] = value;
                        } else {
                            this.originalConfig.autoBot[key] = value;
                        }
                    }
                    console.log(`[BotsManager] 💾 originalConfig обновлен после сохранения ${sectionName}`);
                    console.log(`[BotsManager] 🔍 Обновленные параметры в originalConfig:`, Object.keys(filteredData));
                    // ✅ КРИТИЧЕСКИ ВАЖНО: Логируем scope для отладки
                    if (filteredData.scope !== undefined) {
                        console.log(`[BotsManager] ✅ scope обновлен в originalConfig: ${this.originalConfig.autoBot.scope}`);
                    }
                }
                
                // ✅ ПЕРЕЗАГРУЖАЕМ КОНФИГУРАЦИЮ ДЛЯ ОБНОВЛЕНИЯ UI (особенно важно для Enhanced RSI)
                setTimeout(() => {
                    console.log(`[BotsManager] 🔄 Перезагрузка конфигурации после сохранения ${sectionName}...`);
                    this.loadConfigurationData();
                    
                    // Если сохраняли Enhanced RSI - перезагружаем данные монет для применения новых фильтров
                    if (sectionName === 'Enhanced RSI' || (configType === 'system' && filteredData.enhanced_rsi_enabled !== undefined)) {
                        console.log('[BotsManager] 🔄 Перезагрузка RSI данных для применения Enhanced RSI настроек...');
                        setTimeout(() => {
                            this.loadCoinsRsiData();
                        }, 500);
                    }
                }, 300);
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка сохранения ${sectionName}:`, error);
            this.showNotification(`❌ Ошибка: ${error.message}`, 'error');
            throw error;
        }
    }

    async saveConfiguration(isAutoSave = false, skipNotification = false) {
        // Отменяем запланированное автосохранение при ручном сохранении
        if (!isAutoSave && this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
            this.autoSaveTimer = null;
            console.log('[BotsManager] ⏸️ Автосохранение отменено - выполняется ручное сохранение');
        }
        
        console.log('[BotsManager] 💾 Сохранение конфигурации...');
        
        try {
            const config = this.collectConfigurationData();
            
            // Отладочные логи для Enhanced RSI
            console.log('[BotsManager] 🔍 Отправляемая конфигурация Enhanced RSI:');
            console.log('  enhanced_rsi_enabled:', config.autoBot.enhanced_rsi_enabled);
            console.log('  enhanced_rsi_require_volume_confirmation:', config.autoBot.enhanced_rsi_require_volume_confirmation);
            console.log('  enhanced_rsi_require_divergence_confirmation:', config.autoBot.enhanced_rsi_require_divergence_confirmation);
            console.log('  enhanced_rsi_use_stoch_rsi:', config.autoBot.enhanced_rsi_use_stoch_rsi);
            
            // БЕЗ БЛОКИРОВКИ - элементы остаются активными!
            
            // ✅ Проверяем, что есть данные для отправки Auto Bot
            if (!config.autoBot || Object.keys(config.autoBot).length === 0) {
                console.log('[BotsManager] ⚠️ Auto Bot конфигурация пуста, пропускаем сохранение');
            } else {
                // ПРИИ (full_ai_control) не трогаем при «Сохранить все» — только при явном переключении тумблера (иначе баг UI может выключить ПРИИ)
                const autoBotPayload = { ...config.autoBot };
                delete autoBotPayload.full_ai_control;
                // Сохраняем Auto Bot настройки
                const autoBotResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(autoBotPayload)
                });
                const autoBotData = await autoBotResponse.json();
                if (!autoBotData.success) {
                    throw new Error(`Ошибка сохранения Auto Bot: ${autoBotData.message || 'Unknown error'}`);
                }
            }
            
            // ✅ Проверяем, что есть данные для отправки System
            if (!config.system || Object.keys(config.system).length === 0) {
                console.log('[BotsManager] ⚠️ System конфигурация пуста, пропускаем сохранение');
            } else {
                // Сохраняем системные настройки
                const systemResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config.system)
                });
                const systemData = await systemResponse.json();
                if (!systemData.success) {
                    throw new Error(`Ошибка сохранения System: ${systemData.message || 'Unknown error'}`);
                }
            }
            
            // Показываем уведомление только при ручном сохранении (при вызове из saveAllConfiguration — skipNotification)
            if (!isAutoSave && !skipNotification) {
                this.showNotification('✅ Настройки сохранены', 'success');
            }
            console.log('[BotsManager] ✅ Конфигурация сохранена в bot_config.py и перезагружена');
            
            // ✅ ОБНОВЛЯЕМ RSI ПОРОГИ (для фильтров и подписей)
            if (config.autoBot) {
                this.updateRsiThresholds(config.autoBot);
                console.log('[BotsManager] 🔄 RSI пороги обновлены после сохранения');
            }
            
            this.aiConfigDirty = false;
            this.updateFloatingSaveButtonVisibility();
            setTimeout(() => this.loadConfigurationData(), 500);
            
            // ✅ ПЕРЕЗАГРУЖАЕМ ДАННЫЕ RSI (чтобы применить новые фильтры)
            setTimeout(() => {
                console.log('[BotsManager] 🔄 Перезагрузка RSI данных для применения новых настроек...');
                this.loadCoinsRsiData();
            }, 1000);
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения конфигурации:', error);
            // Показываем уведомление об ошибке только если это не автосохранение
            if (!isAutoSave && !skipNotification) {
                this.showNotification('❌ Ошибка сохранения конфигурации: ' + error.message, 'error');
            }
            // Пробрасываем ошибку дальше для обработки в scheduleAutoSave
            throw error;
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
                    // ✅ Новые параметры RSI выхода с учетом тренда
                    rsi_exit_long_with_trend: 65,
                    rsi_exit_long_against_trend: 60,
                    rsi_exit_short_with_trend: 35,
                    rsi_exit_short_against_trend: 40,
                    rsi_exit_min_candles: 0,
                    rsi_exit_min_minutes: 0,
                    rsi_exit_min_move_percent: 0,
                    exit_wait_breakeven_when_loss: true,
                    default_position_size: 10,
                    default_position_mode: 'usdt',
                    check_interval: 180,
                    max_loss_percent: 15.0,
                    take_profit_percent: 5.0,
                    close_at_profit_enabled: true,
                    trailing_stop_activation: 20.0,
                    trailing_stop_distance: 5.0,
                    trailing_take_distance: 0.5,
                    trailing_update_interval: 3.0,
                    max_position_hours: 0,
                    break_even_protection: true,
                    loss_reentry_protection: true,
                    loss_reentry_count: 1,
                    loss_reentry_candles: 3,
                    avoid_down_trend: true,
                    avoid_up_trend: true,
                    // Параметры анализа тренда
                    trend_detection_enabled: true,
                    trend_analysis_period: 30,
                    trend_price_change_threshold: 7,
                    trend_candles_threshold: 70,
                    break_even_trigger: 20.0,
                    enable_maturity_check: true,
                    min_candles_for_maturity: 200,
                    min_rsi_low: 35,
                    max_rsi_high: 65,
                    trading_enabled: true,
                    use_test_server: false,
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

    /**
     * Экспорт полного конфига в InfoBot_Config_<TF>.json (Auto Bot + System + AI с сервера).
     * Имя файла по выбранному таймфрейму: InfoBot_Config_1m.json, InfoBot_Config_5m.json, InfoBot_Config_15m.json и т.д.
     */
    async exportConfig() {
        try {
            const res = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/export-config`);
            if (!res.ok) throw new Error('Не удалось загрузить конфигурацию');
            const data = await res.json();
            if (!data.success) throw new Error(data.error || 'Ошибка API');
            const tf = (data.timeframe || '1m').replace(/\s/g, '');
            const payload = {
                ...(data.config || {}),
                exportedAt: new Date().toISOString(),
                timeframe: tf,
                version: 1
            };
            const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `InfoBot_Config_${tf}.json`;
            a.click();
            URL.revokeObjectURL(url);
            this.showNotification(`✅ Конфигурация экспортирована в InfoBot_Config_${tf}.json`, 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка экспорта:', error);
            this.showNotification('❌ Ошибка экспорта: ' + error.message, 'error');
        }
    }

    /**
     * Импорт конфигурации из InfoBot_Config_<TF>.json (файл, выгруженный через «Экспорт»).
     * Поддерживает форматы: { autoBot, system, ai } и { config: { autoBot, system, ai } }.
     * Один запрос POST /api/bots/import-config — все блоки применяются и сохраняются в файл и БД.
     */
    async importConfig(file) {
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            if (!data || typeof data !== 'object') throw new Error('Неверный формат JSON');
            const config = data.config && typeof data.config === 'object' ? data.config : data;
            const hasAutoBot = config.autoBot && typeof config.autoBot === 'object';
            const hasSystem = config.system && typeof config.system === 'object';
            const hasAi = config.ai && typeof config.ai === 'object';
            if (!hasAutoBot && !hasSystem && !hasAi) throw new Error('В файле должны быть autoBot, system и/или ai');
            if (!confirm('Применить загруженную конфигурацию? Настройки будут сохранены в файл и БД.')) return;

            const baseUrl = this.BOTS_SERVICE_URL;
            const res = await fetch(`${baseUrl}/api/bots/import-config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config })
            });
            const result = await res.json();
            if (!result.success) throw new Error(result.error || 'Ошибка импорта');

            await this.loadConfigurationData();
            this.showNotification('✅ Конфигурация импортирована и сохранена в файл', 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка импорта:', error);
            this.showNotification('❌ Ошибка импорта: ' + error.message, 'error');
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
        
        // ✅ Валидация новых параметров RSI выхода
        if (config.autoBot.rsi_exit_long_with_trend && config.autoBot.rsi_exit_long_with_trend <= config.autoBot.rsi_long_threshold) {
            errors.push('RSI выход из LONG (по тренду) должен быть больше порога входа');
        }
        
        if (config.autoBot.rsi_exit_long_against_trend && config.autoBot.rsi_exit_long_against_trend <= config.autoBot.rsi_long_threshold) {
            errors.push('RSI выход из LONG (против тренда) должен быть больше порога входа');
        }
        
        if (config.autoBot.rsi_exit_short_with_trend && config.autoBot.rsi_exit_short_with_trend >= config.autoBot.rsi_short_threshold) {
            errors.push('RSI выход из SHORT (по тренду) должен быть меньше порога входа');
        }
        
        if (config.autoBot.rsi_exit_short_against_trend && config.autoBot.rsi_exit_short_against_trend >= config.autoBot.rsi_short_threshold) {
            errors.push('RSI выход из SHORT (против тренда) должен быть меньше порога входа');
        }
        
        if (config.autoBot.max_loss_percent <= 0 || config.autoBot.max_loss_percent > 50) {
            errors.push('Стоп-лосс должен быть от 1% до 50%');
        }
        
        if (config.autoBot.close_at_profit_enabled !== false && (config.autoBot.take_profit_percent <= 0 || config.autoBot.take_profit_percent > 100)) {
            errors.push('При включённом «Закрывать по % прибыли» укажите Take Profit от 1% до 100%');
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
        // Синхронизируем переключатель «Полный Режим ИИ» на вкладке Управление
        const fullAiControlToggleEl = document.getElementById('fullAiControlToggle');
        if (fullAiControlToggleEl) {
            const fullAiOn = config.full_ai_control === true;
            fullAiControlToggleEl.checked = fullAiOn;
            const aiEnabled = config.ai_enabled === true;
            const aiLicenseValid = config.ai_license_valid === true;
            // Тумблер FullAI всегда активен — при включении бэкенд при необходимости включит ИИ; при невалидной лицензии FullAI сбросится при загрузке
            fullAiControlToggleEl.disabled = false;
            if (!aiEnabled) {
                fullAiControlToggleEl.title = (window.languageUtils?.translate?.('full_ai_control_disabled_hint') || 'При включении FullAI ИИ будет включён автоматически, если лицензия валидна');
            } else if (!aiLicenseValid) {
                fullAiControlToggleEl.title = (window.languageUtils?.translate?.('full_ai_control_license_warning') || 'При невалидной лицензии FullAI будет сброшен при загрузке');
            } else {
                fullAiControlToggleEl.title = (window.languageUtils?.translate?.('full_ai_control_tooltip') || 'ИИ сам решает когда входить и выходить');
            }
            const fullAiLabel = fullAiControlToggleEl.closest('.full-ai-control-toggle')?.querySelector('.toggle-label');
            if (fullAiLabel) {
                fullAiLabel.textContent = fullAiOn ? '🧠 Полный Режим ИИ (ВКЛ)' : '🧠 Полный Режим ИИ';
            }
            const fullAiModeBadge = document.getElementById('fullAiModeBadge');
            if (fullAiModeBadge) {
                fullAiModeBadge.textContent = fullAiOn
                    ? (window.languageUtils?.translate?.('fullai_mode_full_ai') || 'Режим: FullAI')
                    : (window.languageUtils?.translate?.('fullai_mode_standard') || 'Режим: Стандартный');
                fullAiModeBadge.className = 'full-ai-mode-badge ' + (fullAiOn ? 'mode-full-ai' : 'mode-standard');
            }
            // Дубль переключателя и бейджа на вкладке Конфигурация
            const fullAiControlToggleConfigEl = document.getElementById('fullAiControlToggleConfig');
            if (fullAiControlToggleConfigEl) {
                fullAiControlToggleConfigEl.checked = fullAiOn;
            }
            const fullAiModeBadgeConfig = document.getElementById('fullAiModeBadgeConfig');
            if (fullAiModeBadgeConfig) {
                fullAiModeBadgeConfig.textContent = fullAiOn
                    ? (window.languageUtils?.translate?.('fullai_mode_full_ai') || 'Режим: FullAI')
                    : (window.languageUtils?.translate?.('fullai_mode_standard') || 'Режим: Стандартный');
                fullAiModeBadgeConfig.className = 'full-ai-mode-badge ' + (fullAiOn ? 'mode-full-ai' : 'mode-standard');
            }
            // Параметры обкатки (ниже переключателя) — подтягиваем при загрузке конфига
            if (fullAiOn) this.loadFullaiAdaptiveConfig();
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
        
        const takeProfitDupEl = document.getElementById('takeProfitPercentDup');
        if (takeProfitDupEl) takeProfitDupEl.value = config.take_profit_percent || 20.0;
        
        const trailingActivationDupEl = document.getElementById('trailingStopActivationDup');
        if (trailingActivationDupEl) {
            const value = Number.parseFloat(config.trailing_stop_activation);
            trailingActivationDupEl.value = Number.isFinite(value) ? value : 20.0;
        }
        
        const trailingDistanceDupEl = document.getElementById('trailingStopDistanceDup');
        if (trailingDistanceDupEl) {
            const value = Number.parseFloat(config.trailing_stop_distance);
            trailingDistanceDupEl.value = Number.isFinite(value) ? value : 5.0;
        }

        const trailingTakeDupEl = document.getElementById('trailingTakeDistanceDup');
        if (trailingTakeDupEl) {
            const value = config.trailing_take_distance;
            trailingTakeDupEl.value = (value !== undefined && value !== null) ? value : 0.5;
        }

        const trailingIntervalDupEl = document.getElementById('trailingUpdateIntervalDup');
        if (trailingIntervalDupEl) {
            const value = config.trailing_update_interval;
            trailingIntervalDupEl.value = (value !== undefined && value !== null) ? value : 3.0;
        }
        
        const maxHoursDupEl = document.getElementById('maxPositionHoursDup');
        if (maxHoursDupEl) {
            const hours = config.max_position_hours || 0;
            maxHoursDupEl.value = Math.round(hours * 3600);
        }
        
        const breakEvenDupEl = document.getElementById('breakEvenProtectionDup');
        if (breakEvenDupEl) breakEvenDupEl.checked = config.break_even_protection !== false;

        const lossReentryProtectionDupEl = document.getElementById('lossReentryProtection');
        if (lossReentryProtectionDupEl) lossReentryProtectionDupEl.checked = config.loss_reentry_protection !== false;

        const lossReentryCountDupEl = document.getElementById('lossReentryCount');
        if (lossReentryCountDupEl) lossReentryCountDupEl.value = config.loss_reentry_count || 1;

        const lossReentryCandlesDupEl = document.getElementById('lossReentryCandles');
        if (lossReentryCandlesDupEl) lossReentryCandlesDupEl.value = config.loss_reentry_candles || 3;
        
        const avoidDownTrendDupEl = document.getElementById('avoidDownTrendDup');
        if (avoidDownTrendDupEl) avoidDownTrendDupEl.checked = config.avoid_down_trend !== false;
        
        const avoidUpTrendDupEl = document.getElementById('avoidUpTrendDup');
        if (avoidUpTrendDupEl) avoidUpTrendDupEl.checked = config.avoid_up_trend !== false;
        
        const enableMaturityCheckDupEl = document.getElementById('enableMaturityCheckDup');
        if (enableMaturityCheckDupEl) enableMaturityCheckDupEl.checked = config.enable_maturity_check !== false;
        
        const breakEvenTriggerDupEl = document.getElementById('breakEvenTriggerDup');
        if (breakEvenTriggerDupEl) {
            // Используем значение из конфига, если оно есть, иначе не меняем поле (оставляем текущее значение)
            const triggerValue = config.break_even_trigger_percent ?? config.break_even_trigger;
            if (triggerValue !== undefined && triggerValue !== null) {
                breakEvenTriggerDupEl.value = triggerValue;
            }
        }
        
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
            // Используем account-info сервиса ботов (баланс + флаг недостатка средств)
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/account-info`);
            const data = await response.json();
            
            if (data.success && (data.total_wallet_balance !== undefined || data.total_available_balance !== undefined)) {
                const accountData = {
                    success: true,
                    total_wallet_balance: data.total_wallet_balance,
                    total_available_balance: data.total_available_balance,
                    total_unrealized_pnl: data.total_unrealized_pnl,
                    active_positions: data.active_positions ?? 0,
                    active_bots: data.active_bots ?? this.activeBots?.length ?? 0,
                    insufficient_funds: !!data.insufficient_funds
                };
                this.updateAccountDisplay(accountData);
                this.logDebug('[BotsManager] ✅ Информация о счете загружена:', accountData);
            } else if (data.wallet_data) {
                // Fallback: ответ в формате /api/positions
                const accountData = {
                    success: true,
                    total_wallet_balance: data.wallet_data.total_balance,
                    total_available_balance: data.wallet_data.available_balance,
                    total_unrealized_pnl: data.wallet_data.realized_pnl,
                    active_positions: data.stats?.total_trades || 0,
                    active_bots: this.activeBots?.length || 0,
                    insufficient_funds: !!data.insufficient_funds
                };
                this.updateAccountDisplay(accountData);
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
        const balance = accountData && accountData.success ? parseFloat(accountData.total_wallet_balance || 0) : null;
        const available = accountData && accountData.success ? parseFloat(accountData.total_available_balance || 0) : null;
        const pnl = accountData && accountData.success ? parseFloat(accountData.total_unrealized_pnl || 0) : null;
        const positions = accountData && accountData.success ? parseInt(accountData.active_positions || 0) : null;
        const insufficient_funds = !!(accountData && accountData.insufficient_funds);
        const key = [balance, available, pnl, positions, insufficient_funds].join('|');
        if (this._lastAccountDisplay === key) {
            return;
        }
        this._lastAccountDisplay = key;
        
        const activeBotsHeader = document.querySelector('.active-bots-header h3');
        if (!activeBotsHeader) return;
        
        if (accountData && accountData.success) {
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
        
        const showInsufficient = insufficient_funds;
        const trInsufficient = (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[document.documentElement.lang || 'ru'] && TRANSLATIONS[document.documentElement.lang || 'ru']['insufficient_funds']);
        document.querySelectorAll('.insufficient-funds-alert').forEach(function (el) {
            el.style.display = showInsufficient ? 'block' : 'none';
            if (showInsufficient && trInsufficient) el.textContent = trInsufficient;
        });
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

    /** Применить сохранённый вид настроек (Карточки / Списком) */
    applyConfigViewMode() {
        const wrapper = document.getElementById('configViewWrapper');
        const mode = (typeof localStorage !== 'undefined' && localStorage.getItem('configViewMode')) || 'cards';
        if (!wrapper) return;
        wrapper.classList.remove('config-view-cards', 'config-view-list');
        wrapper.classList.add(mode === 'list' ? 'config-view-list' : 'config-view-cards');
        document.querySelectorAll('.config-view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === mode);
        });
    }

    /** Инициализация переключателя вида настроек (Карточки / Списком) */
    _initConfigViewSwitcher() {
        const wrapper = document.getElementById('configViewWrapper');
        const btns = document.querySelectorAll('.config-view-btn');
        if (!wrapper || !btns.length) return;
        this.applyConfigViewMode();
        btns.forEach(btn => {
            if (btn.hasAttribute('data-initialized')) return;
            btn.setAttribute('data-initialized', 'true');
            btn.addEventListener('click', () => {
                const view = btn.dataset.view;
                if (typeof localStorage !== 'undefined') localStorage.setItem('configViewMode', view);
                wrapper.classList.remove('config-view-cards', 'config-view-list');
                wrapper.classList.add(view === 'list' ? 'config-view-list' : 'config-view-cards');
                btns.forEach(b => b.classList.toggle('active', b.dataset.view === view));
            });
        });
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

        // Экспорт конфигурации в config.json
        const exportConfigBtn = document.getElementById('exportConfigBtn');
        if (exportConfigBtn && !exportConfigBtn.hasAttribute('data-initialized')) {
            exportConfigBtn.setAttribute('data-initialized', 'true');
            exportConfigBtn.addEventListener('click', () => this.exportConfig());
        }

        // Импорт конфигурации из config.json
        const importConfigBtn = document.getElementById('importConfigBtn');
        const importConfigFileInput = document.getElementById('importConfigFileInput');
        if (importConfigBtn && importConfigFileInput && !importConfigBtn.hasAttribute('data-initialized')) {
            importConfigBtn.setAttribute('data-initialized', 'true');
            importConfigBtn.addEventListener('click', () => importConfigFileInput.click());
            importConfigFileInput.addEventListener('change', (e) => {
                const file = e.target.files?.[0];
                if (file) this.importConfig(file);
                e.target.value = '';
            });
        }

        // Переключатель вида настроек (Карточки / Списком)
        this._initConfigViewSwitcher();
        
        // ✅ ОБРАБОТЧИКИ ДЛЯ КНОПОК СОХРАНЕНИЯ ОТДЕЛЬНЫХ БЛОКОВ
        
        // Основные настройки
        const saveBasicBtn = document.querySelector('.config-section-save-btn[data-section="basic"]');
        if (saveBasicBtn && !saveBasicBtn.hasAttribute('data-initialized')) {
            saveBasicBtn.setAttribute('data-initialized', 'true');
            saveBasicBtn.addEventListener('click', () => this.saveBasicSettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить основные настройки" инициализирована');
        }
        
        const applyFullAiControl = async (value) => {
            try {
                await this.sendConfigUpdate('auto-bot', { full_ai_control: value }, value ? 'Полный Режим ИИ включён' : 'Полный Режим ИИ выключен', { forceSend: true });
                const autoBot = this.collectConfigurationData().autoBot || {};
                this.syncDuplicateSettings({ ...autoBot, full_ai_control: value });
                // Один переключатель управляет и Adaptive: синхронизируем fullai_config
                await this.saveFullaiAdaptiveConfig();
            } catch (e) {
                console.error('[BotsManager] Ошибка сохранения FullAI:', e);
                this.showNotification('Ошибка сохранения переключателя FullAI', 'error');
            }
        };
        // Тумблер «Полный Режим ИИ» на вкладке Управление и дубль на Конфигурации — синхронизируем при изменении любого
        const fullAiToggleEl = document.getElementById('fullAiControlToggle');
        const fullAiToggleConfigEl = document.getElementById('fullAiControlToggleConfig');
        const syncFullAiToggles = (sourceEl, value) => {
            if (fullAiToggleEl && fullAiToggleEl !== sourceEl) fullAiToggleEl.checked = value;
            if (fullAiToggleConfigEl && fullAiToggleConfigEl !== sourceEl) fullAiToggleConfigEl.checked = value;
        };
        if (fullAiToggleEl && !fullAiToggleEl.hasAttribute('data-fullai-listener')) {
            fullAiToggleEl.setAttribute('data-fullai-listener', 'true');
            fullAiToggleEl.addEventListener('change', () => {
                const value = fullAiToggleEl.checked;
                syncFullAiToggles(fullAiToggleEl, value);
                applyFullAiControl(value);
            });
        }
        if (fullAiToggleConfigEl && !fullAiToggleConfigEl.hasAttribute('data-fullai-listener')) {
            fullAiToggleConfigEl.setAttribute('data-fullai-listener', 'true');
            fullAiToggleConfigEl.addEventListener('change', () => {
                const value = fullAiToggleConfigEl.checked;
                syncFullAiToggles(fullAiToggleConfigEl, value);
                applyFullAiControl(value);
            });
        }
        
        let fullaiAdaptiveSaveTimer = null;
        const scheduleFullaiAdaptiveSave = () => {
            if (fullaiAdaptiveSaveTimer) clearTimeout(fullaiAdaptiveSaveTimer);
            fullaiAdaptiveSaveTimer = setTimeout(() => this.saveFullaiAdaptiveConfig(), 800);
        };
        const fullaiAdaptiveIds = ['fullaiAdaptiveDeadCandles', 'fullaiAdaptiveVirtualSuccess', 'fullaiAdaptiveRealLoss', 'fullaiAdaptiveRoundSize', 'fullaiAdaptiveMaxFailures'];
        fullaiAdaptiveIds.forEach(id => {
            const el = document.getElementById(id);
            if (el && !el.hasAttribute('data-fullai-adaptive-listener')) {
                el.setAttribute('data-fullai-adaptive-listener', 'true');
                el.addEventListener('change', () => {
                    if (id === 'fullaiAdaptiveVirtualSuccess') this._updateFullaiAdaptiveDependentFields();
                    scheduleFullaiAdaptiveSave();
                });
                el.addEventListener('input', () => {
                    if (id === 'fullaiAdaptiveVirtualSuccess') this._updateFullaiAdaptiveDependentFields();
                    scheduleFullaiAdaptiveSave();
                });
            }
        });
        this._updateFullaiAdaptiveDependentFields();
        
        // Кнопка сброса всех монет к глобальным настройкам
        const resetAllCoinsBtn = document.getElementById('resetAllCoinsToGlobalBtn');
        if (resetAllCoinsBtn && !resetAllCoinsBtn.hasAttribute('data-initialized')) {
            resetAllCoinsBtn.setAttribute('data-initialized', 'true');
            resetAllCoinsBtn.addEventListener('click', () => this.resetAllCoinsToGlobalSettings());
            console.log('[BotsManager] ✅ Кнопка "Сбросить все монеты к глобальным настройкам" инициализирована');
        }
        
        // Системные настройки
        const saveSystemBtn = document.querySelector('.config-section-save-btn[data-section="system"]');
        if (saveSystemBtn && !saveSystemBtn.hasAttribute('data-initialized')) {
            saveSystemBtn.setAttribute('data-initialized', 'true');
            saveSystemBtn.addEventListener('click', () => this.saveSystemSettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить системные настройки" инициализирована');
        }
        
        // Торговые параметры и RSI выходы (объединённая кнопка)
        const saveTradingRsiBtn = document.querySelector('.config-section-save-btn[data-section="trading-rsi"]');
        if (saveTradingRsiBtn && !saveTradingRsiBtn.hasAttribute('data-initialized')) {
            saveTradingRsiBtn.setAttribute('data-initialized', 'true');
            saveTradingRsiBtn.addEventListener('click', () => this.saveTradingAndRsiExits());
        }
        
        // RSI временной фильтр
        const saveRsiTimeBtn = document.querySelector('.config-section-save-btn[data-section="rsi-time-filter"]');
        if (saveRsiTimeBtn && !saveRsiTimeBtn.hasAttribute('data-initialized')) {
            saveRsiTimeBtn.setAttribute('data-initialized', 'true');
            saveRsiTimeBtn.addEventListener('click', () => this.saveRsiTimeFilter());
            console.log('[BotsManager] ✅ Кнопка "Сохранить RSI временной фильтр" инициализирована');
        }
        
        // ExitScam фильтр — сохраняется по общим правилам (авто при переключении чекбоксов/select, числа — через общее сохранение)
        
        // Enhanced RSI
        const saveEnhancedRsiBtn = document.querySelector('.config-section-save-btn[data-section="enhanced-rsi"]');
        if (saveEnhancedRsiBtn && !saveEnhancedRsiBtn.hasAttribute('data-initialized')) {
            saveEnhancedRsiBtn.setAttribute('data-initialized', 'true');
            saveEnhancedRsiBtn.addEventListener('click', () => this.saveEnhancedRsi());
            console.log('[BotsManager] ✅ Кнопка "Сохранить Enhanced RSI" инициализирована');
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
        
        // Набор позиций лимитными ордерами
        const saveLimitOrdersBtn = document.querySelector('.config-section-save-btn[data-section="limit-orders"]');
        if (saveLimitOrdersBtn && !saveLimitOrdersBtn.hasAttribute('data-initialized')) {
            saveLimitOrdersBtn.setAttribute('data-initialized', 'true');
            saveLimitOrdersBtn.addEventListener('click', () => this.saveLimitOrdersSettings());
            console.log('[BotsManager] ✅ Кнопка "Сохранить настройки набора позиций" инициализирована');
        }
        
        // Кнопка "По умолчанию" для лимитных ордеров
        const resetLimitOrdersBtn = document.getElementById('resetLimitOrdersBtn');
        if (resetLimitOrdersBtn && !resetLimitOrdersBtn.hasAttribute('data-initialized')) {
            resetLimitOrdersBtn.setAttribute('data-initialized', 'true');
            resetLimitOrdersBtn.addEventListener('click', () => this.resetLimitOrdersToDefault());
            console.log('[BotsManager] ✅ Кнопка "По умолчанию" для лимитных ордеров инициализирована');
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
    
    /**
     * Инициализация автосохранения конфигурации
     * Автоматически сохраняет изменения через 2 секунды после внесения в поле
     */
    initializeAutoSave() {
        console.log('[BotsManager] ⚙️ Инициализация автосохранения конфигурации...');
        
        // Находим контейнер конфигурации
        const configTab = document.getElementById('configTab');
        if (!configTab) {
            console.warn('[BotsManager] ⚠️ Вкладка конфигурации не найдена, автосохранение не инициализировано');
            return;
        }
        
        // Находим все поля конфигурации: input, select, checkbox
        // Включая поля в секции AI (aiConfigSection), которая может быть скрыта
        const configInputs = configTab.querySelectorAll('input[type="number"], input[type="text"], input[type="checkbox"], select');
        
        // Также добавляем поля из секции AI, если она существует
        const aiConfigSection = document.getElementById('aiConfigSection');
        let allInputs = Array.from(configInputs);
        
        if (aiConfigSection) {
            const aiInputs = aiConfigSection.querySelectorAll('input[type="number"], input[type="text"], input[type="checkbox"], select');
            console.log(`[BotsManager] 🔍 Найдено полей в секции AI: ${aiInputs.length}`);
            // Добавляем поля из AI секции
            allInputs = Array.from(new Set([...allInputs, ...Array.from(aiInputs)]));
        }
        
        console.log(`[BotsManager] 🔍 Всего полей конфигурации: ${allInputs.length}`);
        
        // Добавляем обработчики для всех полей
        this.addAutoSaveHandlers(allInputs);
        
        // ✅ Явно добавляем обработчик для toggle лимитных ордеров (может не попасть в querySelectorAll)
        const limitOrdersToggle = document.getElementById('limitOrdersEntryEnabled');
        if (limitOrdersToggle && !limitOrdersToggle.hasAttribute('data-autosave-initialized')) {
            limitOrdersToggle.setAttribute('data-autosave-initialized', 'true');
            limitOrdersToggle.addEventListener('change', () => {
                if (!this.isProgrammaticChange) this.scheduleToggleAutoSave(limitOrdersToggle);
            });
            console.log('[BotsManager] ✅ Обработчик автосохранения добавлен для toggle лимитных ордеров');
        }
    }
    
    /**
     * Сохраняет ТОЛЬКО одно значение переключателя (checkbox/select) — предотвращает сброс других настроек
     */
    async saveSingleToggleToBackend(input) {
        if (!input || !input.id) return false;
        const configKey = this.mapElementIdToConfigKey(input.id);
        if (!configKey) return false;

        const systemConfigKeys = [
            'enhanced_rsi_enabled', 'enhanced_rsi_require_volume_confirmation', 'enhanced_rsi_require_divergence_confirmation',
            'enhanced_rsi_use_stoch_rsi', 'rsi_extreme_zone_timeout', 'rsi_extreme_oversold', 'rsi_extreme_overbought',
            'rsi_volume_confirmation_multiplier', 'rsi_divergence_lookback', 'rsi_update_interval', 'auto_save_interval',
            'debug_mode', 'refresh_interval', 'position_sync_interval',
            'inactive_bot_cleanup_interval', 'inactive_bot_timeout', 'stop_loss_setup_interval',
            'bybit_margin_mode'
        ];
        const isSystem = configKey.startsWith('system_') || systemConfigKeys.includes(configKey);

        let value;
        if (input.type === 'checkbox') {
            value = input.checked;
        } else if (input.tagName === 'SELECT' || input.type === 'hidden') {
            value = input.value;
        } else {
            return false;
        }

        try {
            if (isSystem) {
                const systemKey = configKey.startsWith('system_') ? configKey.replace('system_', '') : configKey;
                const payload = { [systemKey]: value };
                const res = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/system-config`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!data.success) throw new Error(data.message || 'System config save failed');
            } else {
                const payload = { [configKey]: value };
                const res = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!data.success) throw new Error(data.message || 'Auto-bot config save failed');
            }
            if (this.originalConfig) {
                const group = isSystem ? this.originalConfig.system : this.originalConfig.autoBot;
                const key = isSystem ? configKey.replace('system_', '') : configKey;
                if (group) group[key] = value;
            }
            if (this.cachedAutoBotConfig && !isSystem) {
                this.cachedAutoBotConfig[configKey] = value;
            }
            return true;
        } catch (e) {
            console.error('[BotsManager] saveSingleToggleToBackend:', e);
            throw e;
        }
    }

    /**
     * Автосохранение при изменении переключателя (checkbox/select) — сохраняет ТОЛЬКО изменённое поле
     */
    scheduleToggleAutoSave(input) {
        if (this.toggleAutoSaveTimer) clearTimeout(this.toggleAutoSaveTimer);
        const self = this;
        this.toggleAutoSaveTimer = setTimeout(async () => {
            self.toggleAutoSaveTimer = null;
            try {
                if (input && input.closest('#aiConfigSection')) {
                    if (window.aiConfigManager && typeof window.aiConfigManager.saveAIConfig === 'function') {
                        await window.aiConfigManager.saveAIConfig(false, false);
                    }
                    self.aiConfigDirty = false;
                } else {
                    const ok = await self.saveSingleToggleToBackend(input);
                    if (!ok) {
                        await self.saveConfiguration(false, true);
                    }
                }
                self.updateFloatingSaveButtonVisibility();
                self.showConfigNotification('✅ Сохранено', 'Настройка сохранена', 'success');
            } catch (err) {
                console.error('[BotsManager] Ошибка автосохранения переключателя:', err);
                self.showConfigNotification('❌ Ошибка', 'Ошибка сохранения: ' + err.message, 'error');
            }
        }, this.toggleAutoSaveDelay);
    }

    /**
     * Добавляет обработчики автосохранения для списка полей
     */
    addAutoSaveHandlers(inputs) {
        // Добавляем обработчики для каждого поля
        inputs.forEach((input, index) => {
            // Пропускаем кнопки и элементы управления
            if (input.type === 'button' || input.type === 'submit' || input.closest('button')) {
                return;
            }
            
            // Проверяем, не добавлен ли уже обработчик
            if (input.hasAttribute('data-autosave-initialized')) {
                return;
            }
            
            input.setAttribute('data-autosave-initialized', 'true');
            
            // Числа и текст: сохраняем только при blur (уход с поля) или Enter — не при каждом нажатии клавиши
            if (input.type === 'number' || input.type === 'text') {
                input.addEventListener('blur', () => {
                    if (!this.isProgrammaticChange) {
                        if (input.closest('#aiConfigSection')) this.aiConfigDirty = true;
                        this.updateFloatingSaveButtonVisibility();
                    }
                });
                input.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') e.target.blur();
                });
            }
            if (input.type === 'checkbox' || input.tagName === 'SELECT') {
                input.addEventListener('change', () => {
                    if (!this.isProgrammaticChange) {
                        this.scheduleToggleAutoSave(input);
                    }
                });
            }
        });
        
        console.log(`[BotsManager] ✅ Обработчики автосохранения добавлены для ${inputs.length} полей`);
    }
    
    /**
     * Добавляет кнопки +/- к числовым полям конфигурации (изменение по step, с учётом min/max).
     */
    addStepperButtons() {
        try {
            const configTab = document.getElementById('configTab');
            const aiSection = document.getElementById('aiConfigSection');
            const containers = [configTab, aiSection].filter(Boolean);
            let added = 0;
            containers.forEach(container => {
                if (!container || !container.querySelectorAll) return;
                const inputs = container.querySelectorAll('.config-input-with-unit input[type="number"].config-input');
                inputs.forEach((input) => {
                    try {
                        const parent = input.closest('.config-input-with-unit');
                        if (!parent || parent.hasAttribute('data-stepper-initialized')) return;
                        parent.setAttribute('data-stepper-initialized', 'true');
                        parent.classList.add('config-input-stepper');
                        const step = parseFloat(input.getAttribute('step')) || 1;
                        const min = input.hasAttribute('min') ? parseFloat(input.getAttribute('min')) : null;
                        const max = input.hasAttribute('max') ? parseFloat(input.getAttribute('max')) : null;
                        const self = this;
                        const applyValue = (val) => {
                            if (min != null && val < min) val = min;
                            if (max != null && val > max) val = max;
                            input.value = val;
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                            input.dispatchEvent(new Event('change', { bubbles: true }));
                            if (!self.isProgrammaticChange) self.updateFloatingSaveButtonVisibility();
                        };
                        const minusBtn = document.createElement('button');
                        minusBtn.type = 'button';
                        minusBtn.className = 'config-step-btn config-step-minus';
                        minusBtn.setAttribute('aria-label', '-');
                        minusBtn.textContent = '−';
                        minusBtn.addEventListener('click', () => {
                            const v = parseFloat(input.value) || 0;
                            applyValue(v - step);
                        });
                        const plusBtn = document.createElement('button');
                        plusBtn.type = 'button';
                        plusBtn.className = 'config-step-btn config-step-plus';
                        plusBtn.setAttribute('aria-label', '+');
                        plusBtn.textContent = '+';
                        plusBtn.addEventListener('click', () => {
                            const v = parseFloat(input.value) || 0;
                            applyValue(v + step);
                        });
                        parent.insertBefore(minusBtn, input);
                        parent.insertBefore(plusBtn, input.nextSibling);
                        added++;
                    } catch (err) {
                        console.warn('[BotsManager] addStepperButtons: ошибка для поля', input?.id, err);
                    }
                });
            });
            if (added > 0) console.log('[BotsManager] ✅ Кнопки +/- добавлены для', added, 'полей');
        } catch (err) {
            console.warn('[BotsManager] addStepperButtons:', err);
        }
    }
    
    /**
     * Планирует автоматическое сохранение конфигурации с задержкой
     */
    scheduleAutoSave() {
        // ✅ Сохраняем контекст this для использования в setTimeout
        const self = this;
        
        // Очищаем предыдущий таймер
        if (this.autoSaveTimer) {
            clearTimeout(this.autoSaveTimer);
            this.autoSaveTimer = null;
        }
        
        // Устанавливаем новый таймер на 2 секунды
        this.autoSaveTimer = setTimeout(async () => {
            console.log('[BotsManager] ⏱️ Автосохранение конфигурации...');
            
            try {
                // Сохраняем конфигурацию с флагом автосохранения
                await self.saveConfiguration(true);
                console.log('[BotsManager] ✅ Конфигурация автосохранена');
                
                // ✅ ПРИНУДИТЕЛЬНО показываем toast-уведомление (прямой вызов toastManager)
                console.log('[BotsManager] 🔔 Показ toast-уведомления об автосохранении...');
                
                // ✅ Прямой вызов toastManager - гарантированно работает
                if (window.toastManager) {
                    // Инициализируем, если нужно
                    if (!window.toastManager.container) {
                        window.toastManager.init();
                    }
                    // Проверяем, что контейнер в DOM
                    if (window.toastManager.container && !document.body.contains(window.toastManager.container)) {
                        document.body.appendChild(window.toastManager.container);
                    }
                    // Показываем уведомление
                    window.toastManager.success('✅ Настройки автоматически сохранены', 3000);
                    console.log('[BotsManager] ✅ Toast-уведомление показано');
                } else {
                    console.warn('[BotsManager] ⚠️ toastManager не найден, пытаемся вызвать showNotification...');
                    // Fallback на showNotification
                    try {
                        self.showNotification('✅ Настройки автоматически сохранены', 'success');
                    } catch (e) {
                        console.error('[BotsManager] ❌ Ошибка показа уведомления:', e);
                    }
                }
            } catch (error) {
                console.error('[BotsManager] ❌ Ошибка автосохранения конфигурации:', error);
                // Показываем ошибку при автосохранении
                if (window.toastManager) {
                    window.toastManager.error('❌ Ошибка автосохранения: ' + error.message, 5000);
                } else {
                    try {
                        self.showNotification('❌ Ошибка автосохранения: ' + error.message, 'error');
                    } catch (e) {
                        console.error('[BotsManager] ❌ Ошибка показа уведомления об ошибке:', e);
                    }
                }
            } finally {
                self.autoSaveTimer = null;
            }
        }, this.autoSaveDelay);
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
            'max_loss_percent': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Max loss (%)' : 'Макс. убыток (%)',
            'trailing_stop_activation': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Trailing stop activation (%)' : 'Активация трейлинг-стопа (%)',
            'trailing_stop_distance': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Trailing stop distance (%)' : 'Расстояние трейлинг-стопа (%)',
            'max_position_hours': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Max time in position (sec)' : 'Макс. время в позиции (сек)',
            'break_even_protection': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Break-even protection' : 'Защита безубыточности',
            'break_even_trigger': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Break-even trigger (%)' : 'Триггер безубыточности (%)',
            'avoid_down_trend': window.languageUtils?.getCurrentLanguage() === 'en' ? '🔻 Avoid downtrend (LONG)' : '🔻 Избегать нисходящий тренд (LONG)',
            'avoid_up_trend': window.languageUtils?.getCurrentLanguage() === 'en' ? '📈 Avoid uptrend (SHORT)' : '📈 Избегать восходящий тренд (SHORT)',
            
            // Системные настройки
            'rsi_update_interval': window.languageUtils?.getCurrentLanguage() === 'en' ? 'RSI update interval' : 'Интервал обновления RSI',
            'auto_save_interval': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Auto-save interval' : 'Интервал автосохранения',
            'mini_chart_update_interval': window.languageUtils?.getCurrentLanguage() === 'en' ? 'Mini charts update interval' : 'Интервал обновления миниграфиков',
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
    
    /** Возвращает компактные данные для карточки бота: объём, позиция, вход, тейк, стоп, текущая цена */
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
    }

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
     */
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

        if (!this.historyInitialized) {
            // Инициализируем фильтры
            this.initializeHistoryFilters();

            // Инициализируем подвкладки истории
            this.initializeHistorySubTabs();

            // Инициализируем кнопки действий
            this.initializeHistoryActionButtons();

            this.historyInitialized = true;
        }

        // Загружаем данные для текущей подвкладки
        this.loadHistoryData(this.currentHistoryTab);
    }

    /**
     * Инициализирует вкладку «Аналитика»: привязка кнопок и однократная привязка обработчиков
     */
    initializeAnalyticsTab() {
        const runBtn = document.getElementById('analyticsRunBtn');
        if (runBtn && !runBtn.hasAttribute('data-analytics-bound')) {
            runBtn.setAttribute('data-analytics-bound', 'true');
            runBtn.addEventListener('click', () => this.runTradingAnalytics());
        }
        const syncBtn = document.getElementById('analyticsSyncExchangeBtn');
        if (syncBtn && !syncBtn.hasAttribute('data-sync-bound')) {
            syncBtn.setAttribute('data-sync-bound', 'true');
            syncBtn.addEventListener('click', () => this.syncTradesFromExchange());
        }
        const rsiAuditBtn = document.getElementById('rsiAuditRunBtn');
        if (rsiAuditBtn && !rsiAuditBtn.hasAttribute('data-rsi-audit-bound')) {
            rsiAuditBtn.setAttribute('data-rsi-audit-bound', 'true');
            rsiAuditBtn.addEventListener('click', () => this.runRsiAudit());
        }
        const fullaiBtn = document.getElementById('fullaiAnalyticsRunBtn');
        if (fullaiBtn && !fullaiBtn.hasAttribute('data-fullai-bound')) {
            fullaiBtn.setAttribute('data-fullai-bound', 'true');
            fullaiBtn.addEventListener('click', () => this.loadFullaiAnalytics());
        }
        const aiReanalyzeBtn = document.getElementById('aiReanalyzeBtn');
        if (aiReanalyzeBtn && !aiReanalyzeBtn.hasAttribute('data-ai-reanalyze-bound')) {
            aiReanalyzeBtn.setAttribute('data-ai-reanalyze-bound', 'true');
            aiReanalyzeBtn.addEventListener('click', () => this.runAiReanalyze());
        }
        const subtabBtns = document.querySelectorAll('.analytics-subtab-btn');
        const subtabPanels = document.querySelectorAll('.analytics-subtab-content');
        if (subtabBtns.length && !document.getElementById('analyticsTab').hasAttribute('data-subtabs-bound')) {
            document.getElementById('analyticsTab').setAttribute('data-subtabs-bound', 'true');
            subtabBtns.forEach(btn => {
                btn.addEventListener('click', () => {
                    const id = btn.getAttribute('data-analytics-subtab');
                    subtabBtns.forEach(b => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
                    subtabPanels.forEach(p => {
                        const on = p.getAttribute('data-analytics-subtab') === id;
                        p.classList.toggle('active', on);
                        p.hidden = !on;
                    });
                    btn.classList.add('active');
                    btn.setAttribute('aria-selected', 'true');
                    if (id === 'fullai') this.loadFullaiAnalytics();
                    if (id === 'rsi') this.runRsiAudit();
                });
            });
        }
    }

    /**
     * Загрузка и отображение аналитики FullAI (события и сводка из data/fullai_analytics.db)
     */
    async loadFullaiAnalytics() {
        const loadingEl = document.getElementById('fullaiAnalyticsLoading');
        const summaryEl = document.getElementById('fullaiAnalyticsSummary');
        const eventsEl = document.getElementById('fullaiAnalyticsEvents');
        const periodHours = parseInt(document.getElementById('fullaiAnalyticsPeriod')?.value, 10) || 168;
        const symbol = (document.getElementById('fullaiAnalyticsSymbol')?.value || '').trim().toUpperCase() || undefined;
        const from_ts = (Date.now() / 1000) - periodHours * 3600;
        const to_ts = Date.now() / 1000;
        if (loadingEl) loadingEl.style.display = 'flex';
        if (summaryEl) summaryEl.innerHTML = '';
        if (eventsEl) eventsEl.innerHTML = '';
        try {
            const params = new URLSearchParams({ from_ts: String(from_ts), to_ts: String(to_ts), limit: '300' });
            if (symbol) params.set('symbol', symbol);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/analytics/fullai?${params}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            if (!data.success) throw new Error(data.error || 'Нет данных');
            this.renderFullaiAnalytics(data.summary || {}, data.events || [], summaryEl, eventsEl, {
                db_path: data.db_path,
                total_events: data.total_events,
                bot_trades_stats: data.bot_trades_stats || null,
                closed_trades: data.closed_trades || []
            });
        } catch (err) {
            if (summaryEl) summaryEl.innerHTML = `<div class="analytics-error">❌ ${(err && err.message) || String(err)}</div>`;
            if (eventsEl) eventsEl.innerHTML = '';
            console.error('[BotsManager] Ошибка аналитики FullAI:', err);
        } finally {
            if (loadingEl) loadingEl.style.display = 'none';
        }
    }

    renderFullaiAnalytics(summary, events, summaryEl, eventsEl, meta) {
        if (!summaryEl) return;
        const botStats = (meta && meta.bot_trades_stats) || null;
        const totalInDb = (meta && meta.total_events) != null ? meta.total_events : null;
        const dbPath = (meta && meta.db_path) || '';
        const s = summary;
        // Реальные сделки: используем bots_data.db (истинный источник), если есть — иначе fullai_analytics
        const realClose = (botStats != null) ? (botStats.total || 0) : (s.real_close || 0);
        const realWins = (botStats != null) ? (botStats.wins || 0) : (s.real_wins || 0);
        const realLosses = (botStats != null) ? (botStats.losses || 0) : (s.real_losses || 0);
        const winRate = (botStats != null && botStats.win_rate_pct != null) ? String(botStats.win_rate_pct) : (s.real_total > 0 ? ((s.real_wins / s.real_total) * 100).toFixed(1) : '—');
        const virtualRate = s.virtual_total > 0 ? ((s.virtual_ok / s.virtual_total) * 100).toFixed(1) : '—';
        let html = '';
        if (botStats && (botStats.total > 0 || botStats.total_pnl_usdt !== 0)) {
            const wr = botStats.win_rate_pct != null ? botStats.win_rate_pct + '%' : '—';
            const pnlClass = (botStats.total_pnl_usdt || 0) >= 0 ? 'positive' : 'negative';
            const pnlStr = (botStats.total_pnl_usdt != null ? (botStats.total_pnl_usdt >= 0 ? '+' : '') + botStats.total_pnl_usdt : '—') + ' USDT';
            html += '<div class="fullai-bot-trades-block" style="margin-bottom:1rem;padding:0.75rem;background:var(--bg-secondary, #1a1a2e);border-radius:8px;border:1px solid var(--border, #333);">';
            html += '<strong>По сделкам бота (bots_data.db)</strong> — совпадает с монитором «Закрытые PNL»:<br>';
            html += '<span>Сделок: ' + botStats.total + '</span> · <span class="positive">В плюс: ' + (botStats.wins || 0) + '</span> · <span class="negative">В минус: ' + (botStats.losses || 0) + '</span> · Win rate: ' + wr + ' · Суммарный PnL: <span class="' + pnlClass + '">' + pnlStr + '</span></div>';
        }
        let cards = '<div class="fullai-cards">';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Реальные входы</span><span class="fullai-card-value">' + (s.real_open || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Виртуальные входы</span><span class="fullai-card-value">' + (s.virtual_open || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Реальные закрытия</span><span class="fullai-card-value">' + realClose + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Реальные в плюс</span><span class="fullai-card-value positive">' + realWins + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Реальные в минус</span><span class="fullai-card-value negative">' + realLosses + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Win rate (реал.)</span><span class="fullai-card-value">' + winRate + '%</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Вирт. закрытий удачных</span><span class="fullai-card-value">' + (s.virtual_ok || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Вирт. закрытий неудачных</span><span class="fullai-card-value">' + (s.virtual_fail || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Успешность вирт.</span><span class="fullai-card-value">' + virtualRate + '%</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Блокировок входа</span><span class="fullai-card-value">' + (s.blocked || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Отказов ИИ</span><span class="fullai-card-value">' + (s.refused || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Смен параметров</span><span class="fullai-card-value">' + (s.params_change || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Раундов → реал.</span><span class="fullai-card-value">' + (s.round_success || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Решений держать</span><span class="fullai-card-value">' + (s.exit_hold || 0) + '</span></div>';
        cards += '</div>';
        html += '<p class="fullai-events-note" style="font-size:0.85rem;color:var(--text-muted,#888);margin-top:0.25rem;">Карточки «Реальные закрытия/в плюс/в минус/Win rate» — из bots_data.db (история ботов). Остальные карточки — события FullAI (записываются только при включённом FullAI).</p>';
        summaryEl.innerHTML = html + cards;

        let closedTradesHtml = '';
        const closedTrades = (meta && meta.closed_trades) || [];
        if (closedTrades.length > 0) {
            closedTradesHtml = '<h4 style="margin-top:0.5rem;">Закрытые сделки (PnL и вывод)</h4>';
            closedTradesHtml += '<table class="fullai-events-table"><thead><tr><th>Время</th><th>Символ</th><th>Напр.</th><th>Вход</th><th>Выход</th><th>PnL %</th><th>PnL USDT</th><th>Причина</th><th>Вывод</th></tr></thead><tbody>';
            closedTrades.forEach(tr => {
                const pnlUsdt = tr.pnl_usdt != null ? Number(tr.pnl_usdt) : null;
                const roiPct = tr.roi_pct != null ? Number(tr.roi_pct) : null;
                const pnlClass = pnlUsdt != null ? (pnlUsdt >= 0 ? 'positive' : 'negative') : '';
                const pnlPctStr = roiPct != null ? ((roiPct >= 0 ? '+' : '') + roiPct.toFixed(2) + '%') : '—';
                const pnlUsdtStr = pnlUsdt != null ? ((pnlUsdt >= 0 ? '+' : '') + pnlUsdt.toFixed(2)) : '—';
                const entryPrice = tr.entry_price != null ? Number(tr.entry_price).toFixed(6) : '—';
                const exitPrice = tr.exit_price != null ? Number(tr.exit_price).toFixed(6) : '—';
                const conclusion = tr.conclusion || (pnlUsdt >= 0 ? 'Прибыль' : 'Убыток');
                closedTradesHtml += '<tr><td>' + (tr.ts_iso || tr.exit_time || '') + '</td><td>' + (tr.symbol || '') + '</td><td>' + (tr.direction || '') + '</td><td>' + entryPrice + '</td><td>' + exitPrice + '</td><td class="' + pnlClass + '">' + pnlPctStr + '</td><td class="' + pnlClass + '">' + pnlUsdtStr + '</td><td>' + (tr.close_reason || '—') + '</td><td>' + (conclusion || '—') + '</td></tr>';
            });
            closedTradesHtml += '</tbody></table><h4 style="margin-top:1.5rem;">Последние события FullAI</h4>';
        }

        if (!eventsEl) return;
        const eventLabels = { real_open: 'Вход реал.', virtual_open: 'Вход вирт.', real_close: 'Закрытие реал.', virtual_close: 'Закрытие вирт.', blocked: 'Блок', refused: 'Отказ ИИ', params_change: 'Смена параметров', round_success: 'Раунд → реал.', exit_hold: 'ИИ держать' };
        if (events.length === 0 && closedTrades.length === 0) {
            let hint = 'Нет событий и закрытых сделок за выбранный период.';
            if (totalInDb === 0) {
                hint = 'В БД 0 событий. Путь: ' + (dbPath || 'data/fullai_analytics.db') + '. Перезапустите сервис ботов после включения FullAI. В логах ботов при записи должна появиться строка «FullAI analytics: запись в БД». Если её нет — решения FullAI не доходят до записи (проверьте, что боты запущены и FullAI включён в Конфигурации).';
            } else if (totalInDb != null && totalInDb > 0) {
                hint = 'В БД всего событий: ' + totalInDb + '. За выбранный период — нет (попробуйте увеличить период).';
            }
            eventsEl.innerHTML = '<p class="analytics-placeholder">' + hint + '</p>';
            return;
        }
        if (events.length === 0 && closedTrades.length > 0) {
            eventsEl.innerHTML = closedTradesHtml;
            return;
        }
        let table = '<table class="fullai-events-table"><thead><tr><th>Время</th><th>Символ</th><th>Событие</th><th>Направление</th><th>Вход</th><th>Выход</th><th>PnL %</th><th>Лимит выхода</th><th>Тип</th><th>Время заявки</th><th>Проскальз.%</th><th>Задержка с</th><th>Детали</th><th>Вывод</th></tr></thead><tbody>';
        events.forEach(ev => {
            const label = eventLabels[ev.event_type] || ev.event_type;
            const dir = ev.direction || '—';
            const ex = ev.extra || {};
            const entryPrice = ex.entry_price != null ? Number(ex.entry_price).toFixed(6) : (ev.event_type === 'real_open' || ev.event_type === 'refused' ? (ex.price != null ? Number(ex.price).toFixed(6) : '—') : '—');
            const exitPrice = ex.exit_price != null ? Number(ex.exit_price).toFixed(6) : '—';
            const limitExit = ex.limit_price_exit != null ? Number(ex.limit_price_exit).toFixed(6) : '—';
            const orderType = ex.order_type_exit || '—';
            const tsPlaced = ex.ts_order_placed_exit != null ? (function() { const d = new Date(ex.ts_order_placed_exit * 1000); return d.toISOString ? d.toISOString().slice(0, 19).replace('T', ' ') : d.toLocaleString(); })() : '—';
            const slippage = ex.slippage_exit_pct != null ? Number(ex.slippage_exit_pct).toFixed(2) + '%' : '—';
            const delay = ex.delay_sec != null ? String(Number(ex.delay_sec).toFixed(1)) : '—';
            const pnlPct = ev.pnl_percent != null ? Number(ev.pnl_percent) : (ex.pnl_percent != null ? Number(ex.pnl_percent) : null);
            const pnlClass = pnlPct != null ? (pnlPct >= 0 ? 'positive' : 'negative') : '';
            const pnlStr = pnlPct != null ? ((pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(2) + '%') : '—';
            const details = ev.reason || (ev.extra && ev.extra.success !== undefined ? (ev.extra.success ? 'успех' : 'убыток') : '') || '—';
            const conclusion = pnlPct != null ? (pnlPct >= 0 ? 'Прибыль. ' + (ev.reason || '') : 'Убыток. ' + (ev.reason || '')) : '—';
            table += '<tr><td>' + (ev.ts_iso || '') + '</td><td>' + (ev.symbol || '') + '</td><td>' + label + '</td><td>' + dir + '</td><td>' + entryPrice + '</td><td>' + exitPrice + '</td><td class="' + pnlClass + '">' + pnlStr + '</td><td>' + limitExit + '</td><td>' + orderType + '</td><td>' + tsPlaced + '</td><td>' + slippage + '</td><td>' + delay + '</td><td>' + details + '</td><td>' + conclusion + '</td></tr>';
        });
        table += '</tbody></table>';
        eventsEl.innerHTML = closedTradesHtml + table;
    }

    /**
     * Запускает аудит RSI входа/выхода и отображает отчёт
     */
    async runRsiAudit() {
        const loadingEl = document.getElementById('rsiAuditLoading');
        const resultEl = document.getElementById('rsiAuditResult');
        const limitEl = document.getElementById('rsiAuditLimit');
        const limit = (limitEl && parseInt(limitEl.value, 10)) || 500;
        if (loadingEl) loadingEl.style.display = 'flex';
        if (resultEl) resultEl.innerHTML = '';
        try {
            const response = await fetch(this.BOTS_SERVICE_URL + '/api/bots/analytics/rsi-audit?limit=' + Math.min(2000, Math.max(50, limit)));
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            if (!data.success || !data.report) throw new Error(data.error || 'Нет данных отчёта');
            this.renderRsiAuditReport(data.report, resultEl);
        } catch (err) {
            if (resultEl) resultEl.innerHTML = '<div class="analytics-error">❌ ' + ((err && err.message) || String(err)) + '</div>';
            console.error('[BotsManager] Ошибка аудита RSI:', err);
        } finally {
            if (loadingEl) loadingEl.style.display = 'none';
        }
    }

    /**
     * Рендер отчёта аудита RSI: сводка, конфиг, таблица сделок (ошибочные входы/выходы подсвечены)
     */
    renderRsiAuditReport(report, container) {
        if (!container) return;
        const cfg = report.config || {};
        const tf = report.timeframe || '1m';
        const sum = report.summary || {};
        const trades = report.trades || [];
        let html = '<div class="rsi-audit-report">';
        html += '<div class="rsi-audit-summary">';
        html += '<h4>Сводка</h4>';
        html += `<p><strong>Всего сделок:</strong> ${sum.total || 0}</p>`;
        html += '<p><strong>Вход:</strong> ';
        html += `✅ по порогу: ${sum.entry_ok || 0} · `;
        html += `<span class="rsi-audit-error">❌ ошибочных (вне порога): ${sum.entry_error || 0}</span> · `;
        html += `без RSI: ${sum.entry_no_rsi || 0}</p>`;
        html += '<p><strong>Выход:</strong> ';
        html += `✅ по порогу: ${sum.exit_ok || 0} · `;
        html += `<span class="rsi-audit-error">❌ вне порога: ${sum.exit_error || 0}</span> · `;
        html += `без RSI: ${sum.exit_no_rsi || 0}</p>`;
        html += '</div>';
        html += '<div class="rsi-audit-config">';
        html += '<h4>Текущий конфиг (эталон)</h4>';
        html += `<p>Таймфрейм: <strong>${tf}</strong> · LONG: RSI ≤ ${cfg.rsi_long_threshold ?? 29} · SHORT: RSI ≥ ${cfg.rsi_short_threshold ?? 71}</p>`;
        html += `<p>Выход LONG: RSI ≥ ${cfg.rsi_exit_long_with_trend ?? 65} (по тренду) / ${cfg.rsi_exit_long_against_trend ?? 60} (против) · Выход SHORT: RSI ≤ ${cfg.rsi_exit_short_with_trend ?? 35} / ${cfg.rsi_exit_short_against_trend ?? 40}</p>`;
        html += '</div>';
        html += '<div class="rsi-audit-table-wrap"><h4>Сделки</h4><table class="rsi-audit-table"><thead><tr>';
        html += '<th>Символ</th><th>Направление</th><th>Вход (время)</th><th>RSI входа</th><th>Порог входа</th><th>Вход</th>';
        html += '<th>Выход (время)</th><th>RSI выхода</th><th>Порог выхода</th><th>Выход</th><th>PnL</th></tr></thead><tbody>';
        trades.forEach((t, i) => {
            const entryStatus = t.entry_rsi == null ? '—' : (t.entry_ok ? '✅ OK' : '<span class="rsi-audit-error">❌ Ошибка</span>');
            const exitStatus = t.exit_rsi == null ? '—' : (t.exit_ok ? '✅ OK' : '<span class="rsi-audit-error">❌ Ошибка</span>');
            const rowClass = (t.entry_error || t.exit_error) ? 'rsi-audit-row-error' : '';
            html += `<tr class="${rowClass}">`;
            html += `<td>${t.symbol || ''}</td><td>${t.direction || ''}</td>`;
            html += `<td>${t.entry_time_iso || ''}</td><td>${t.entry_rsi != null ? t.entry_rsi : '—'}</td><td>${t.entry_threshold != null ? t.entry_threshold : ''}</td><td>${entryStatus}</td>`;
            html += `<td>${t.exit_time_iso || ''}</td><td>${t.exit_rsi != null ? t.exit_rsi : '—'}</td><td>${t.exit_threshold != null ? t.exit_threshold : ''}</td><td>${exitStatus}</td>`;
            html += `<td>${t.pnl != null ? Number(t.pnl).toFixed(4) : ''}</td>`;
            html += '</tr>';
        });
        html += '</tbody></table></div>';
        html += `<div class="rsi-audit-meta">Отчёт: ${report.generated_at || ''}</div>`;
        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Синхронизирует bot_trades_history с данными биржи (обновляет цены и PnL в БД)
     */
    async syncTradesFromExchange() {
        const syncBtn = document.getElementById('analyticsSyncExchangeBtn');
        const origText = syncBtn ? syncBtn.textContent : '';
        if (syncBtn) syncBtn.disabled = true;
        try {
            const response = await fetch(this.BOTS_SERVICE_URL + '/api/bots/analytics/sync-from-exchange', { method: 'POST' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            const msg = data.updated != null ? ('Обновлено ' + data.updated + ' из ' + (data.matched || 0) + ' совпавших') : (data.message || 'Готово');
            alert('Синхронизация с биржей: ' + msg);
            if (data.updated > 0) this.runTradingAnalytics();
        } catch (err) {
            alert('Ошибка синхронизации: ' + ((err && err.message) || String(err)));
        } finally {
            if (syncBtn) { syncBtn.disabled = false; syncBtn.textContent = origText; }
        }
    }

    /**
     * Запускает ручной анализ ИИ: обновление данных, подход к сделкам и переобучение (в фоне).
     * Показывает изменения в формате «старое → новое».
     */
    async runAiReanalyze() {
        const btn = document.getElementById('aiReanalyzeBtn');
        const resultEl = document.getElementById('aiReanalyzeResult');
        const origText = btn ? btn.textContent : '';
        if (btn) { btn.disabled = true; btn.textContent = '⏳ Запуск...'; }
        if (resultEl) { resultEl.style.display = 'none'; resultEl.innerHTML = ''; }
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/analytics/ai-reanalyze`, { method: 'POST' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            if (!data.success) throw new Error(data.error || 'Не удалось запустить');

            const changes = data.changes || [];
            if (resultEl) {
                resultEl.style.display = 'block';
                if (changes.length > 0) {
                    const paramNames = {
                        take_profit_percent: 'TP%',
                        max_loss_percent: 'SL%',
                        rsi_long_threshold: 'RSI long',
                        rsi_short_threshold: 'RSI short'
                    };
                    const isPercent = (p) => p === 'take_profit_percent' || p === 'max_loss_percent';
                    let html = '<strong>🧠 Изменения ИИ:</strong><ul style="margin: 6px 0 0 16px;">';
                    changes.forEach(c => {
                        const p = paramNames[c.param] || c.param;
                        const suf = isPercent(c.param) ? '%' : '';
                        html += `<li><code>${c.symbol}</code> ${p}: <span style="text-decoration:line-through">${c.old}${suf}</span> → <strong>${c.new}${suf}</strong></li>`;
                    });
                    html += '</ul>';
                    html += '<p style="margin: 8px 0 0; color: var(--text-muted, #666); font-size: 0.85em;">' + (data.message || '') + '</p>';
                    resultEl.innerHTML = html;
                } else {
                    resultEl.innerHTML = '<strong>🧠</strong> ' + (data.message || 'Готово. Изменений параметров нет.');
                }
            } else {
                alert(data.message || 'ИИ анализирует и обновляет данные в фоне.');
            }
        } catch (err) {
            if (resultEl) {
                resultEl.style.display = 'block';
                resultEl.innerHTML = '<span class="analytics-error">❌ ' + ((err && err.message) || String(err)) + '</span>';
            } else {
                alert('Ошибка: ' + ((err && err.message) || String(err)));
            }
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = origText; }
        }
    }

    /**
     * Запускает аналитику торговли и отображает результат во вкладке «Аналитика»
     */
    async runTradingAnalytics() {
        const loadingEl = document.getElementById('analyticsLoading');
        const resultEl = document.getElementById('analyticsResult');
        const includeExchange = document.getElementById('analyticsIncludeExchange') && document.getElementById('analyticsIncludeExchange').checked;
        if (loadingEl) loadingEl.style.display = 'flex';
        if (resultEl) resultEl.innerHTML = '';
        try {
            const params = new URLSearchParams({ limit: '10000', include_exchange: includeExchange ? '1' : '0' });
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/analytics?${params}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            if (!data.success || !data.report) throw new Error(data.error || 'Нет данных отчёта');
            this.renderAnalyticsReport(data.report, resultEl);
        } catch (err) {
            if (resultEl) resultEl.innerHTML = '<div class="analytics-error">❌ ' + ((err && err.message) || String(err)) + '</div>';
            console.error('[BotsManager] Ошибка аналитики:', err);
        } finally {
            if (loadingEl) loadingEl.style.display = 'none';
        }
    }

    /**
     * Формирует HTML отчёта аналитики с переключаемыми категориями и вставляет в контейнер
     */
    renderAnalyticsReport(report, container) {
        if (!container) return;
        const s = report.summary || {};
        const bot = report.bot_analytics || {};
        const categories = [
            { id: 'summary', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_summary')) || 'Сводка' },
            { id: 'bots', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_bots')) || 'Сделки ботов' },
            { id: 'trades_table', label: 'Таблица сделок' },
            { id: 'by_symbol', label: 'По символам' },
            { id: 'by_bot', label: 'По ботам' },
            { id: 'by_decision_source', label: 'По источникам решений' },
            { id: 'reasons', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_reasons')) || 'Причины закрытия' },
            { id: 'unsuccessful_coins', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_unsuccessful_coins')) || 'Неудачные монеты' },
            { id: 'unsuccessful_settings', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_unsuccessful_settings')) || 'Неудачные настройки' },
            { id: 'successful_coins', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_successful_coins')) || 'Удачные монеты' },
            { id: 'successful_settings', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_successful_settings')) || 'Удачные настройки' }
        ];
        let tabsHtml = '<div class="analytics-category-tabs">';
        categories.forEach((cat, i) => {
            tabsHtml += `<button type="button" class="analytics-cat-btn ${i === 0 ? 'active' : ''}" data-category="${cat.id}">${cat.label}</button>`;
        });
        tabsHtml += '</div>';

        let bodyHtml = '<div class="analytics-report">';
        const exchangeCount = s.exchange_trades_count ?? 0;
        const botCountRaw = s.bot_trades_count ?? 0;
        const botCountUnique = (bot.total_trades != null ? bot.total_trades : botCountRaw);
        const onlyBots = s.reconciliation_only_bots ?? 0;
        let summaryNote = '';
        if (botCountRaw > exchangeCount && exchangeCount > 0) {
            summaryNote = '<p class="analytics-summary-note">В БД записей больше, чем биржа вернула по API: у биржи ограничена история (например 2 года или лимит страниц). «Только в БД» — сделки из БД без пары в ответе API (часто старые). В БД учтены закрытия ботов и ручные через интерфейс.</p>';
        }
        const botCountNote = (botCountUnique < botCountRaw) ? ` <small>(уникальных: ${botCountUnique}, всего записей в БД: ${botCountRaw})</small>` : ` <small>(всего записей в БД)</small>`;
        const series = bot.consecutive_series || {};
        const dd = bot.drawdown || {};
        const pfStr = bot.profit_factor != null ? (bot.profit_factor >= 999 ? '∞' : bot.profit_factor.toFixed(2)) : '—';
        var possibleErrorsHtml = '';
        if ((bot.possible_errors_count || 0) > 0) {
            var errs = Array.isArray(bot.possible_errors) ? bot.possible_errors.slice(0, 20) : [];
            possibleErrorsHtml = '<h4>⚠ Возможные ошибки по сделкам</h4><p>Найдено: <strong>' + bot.possible_errors_count + '</strong>.</p>';
            if (errs.length > 0) {
                possibleErrorsHtml += '<div class="analytics-stats-table-wrap"><table class="analytics-stats-table"><thead><tr><th>Символ</th><th>Время</th><th>PnL</th><th>Причина</th></tr></thead><tbody>';
                for (var i = 0; i < errs.length; i++) {
                    var e = errs[i];
                    var ts = e.exit_timestamp ? new Date(e.exit_timestamp * 1000).toISOString().slice(0, 19) : '—';
                    var reason = String(e.close_reason != null ? e.close_reason : '—').slice(0, 30);
                    possibleErrorsHtml += '<tr><td>' + (e.symbol || '—') + '</td><td>' + ts + '</td><td>' + (e.pnl != null ? e.pnl : '—') + '</td><td>' + reason + '</td></tr>';
                }
                possibleErrorsHtml += '</tbody></table></div>';
            }
        }
        bodyHtml += '<div class="analytics-section" data-category="summary">' +
            '<h3>' + categories[0].label + '</h3>' +
            '<h4 style="margin-top:0;">Метрики торговли</h4>' +
            '<p>Сделок: <strong>' + (bot.total_trades != null ? bot.total_trades : botCountUnique) + '</strong> · Прибыльных: <strong>' + (bot.win_count ?? '—') + '</strong> · Убыточных: <strong>' + (bot.loss_count ?? '—') + '</strong> · Нулевых: <strong>' + (bot.neutral_count ?? '—') + '</strong><br>' +
            'Win Rate: <strong>' + (s.bot_win_rate_pct != null ? s.bot_win_rate_pct + '%' : '—') + '</strong> · Суммарный PnL: <strong>' + (s.bot_total_pnl_usdt != null ? s.bot_total_pnl_usdt + ' USDT' : '—') + '</strong> · Profit Factor: <strong>' + pfStr + '</strong></p>' +
            '<p>Средняя прибыль на сделку: <strong>' + (bot.avg_win_usdt != null ? bot.avg_win_usdt + ' USDT' : '—') + '</strong> · Средний убыток: <strong>' + (bot.avg_loss_usdt != null ? bot.avg_loss_usdt + ' USDT' : '—') + '</strong></p>' +
            '<p>Макс. серия побед: <strong>' + (series.max_consecutive_wins ?? '—') + '</strong> · Макс. серия убытков: <strong>' + (series.max_consecutive_losses ?? '—') + '</strong> · Просадка: <strong>' + (dd.max_drawdown_usdt != null ? dd.max_drawdown_usdt + ' USDT' : '—') + (dd.max_drawdown_pct != null ? ' (' + dd.max_drawdown_pct + '%)' : '') + '</strong></p>' +
            possibleErrorsHtml +
            '<h4>Сверка с биржей</h4>' +
            '<p><strong>С биржи (по API):</strong> ' + exchangeCount + ' · <strong>В БД</strong> (закрытия ботов и ручные): <strong>' + botCountUnique + '</strong>' + botCountNote + '<br>' +
            'Совпадений: <strong>' + (s.reconciliation_matched ?? 0) + '</strong> · Только в ответе биржи: <strong>' + (s.reconciliation_only_exchange ?? 0) + '</strong> · ' +
            'Только в БД (нет пары в ответе API): <strong>' + onlyBots + '</strong> · Расхождений PnL: <strong>' + (s.reconciliation_pnl_mismatches ?? 0) + '</strong></p>' +
            summaryNote +
            '<p class="analytics-summary-note" style="margin-top: 6px;">В отчёте учтены только уникальные сделки: дубликаты отброшены по времени закрытия.</p>' +
            '</div>';

        bodyHtml += '<div class="analytics-section" data-category="bots">';
        if (bot.total_trades != null) {
            const series = bot.consecutive_series || {};
            const dd = bot.drawdown || {};
            const pfVal = bot.profit_factor != null ? (bot.profit_factor >= 999 ? '∞' : bot.profit_factor.toFixed(2)) : '—';
            bodyHtml += '<h3>' + (categories[1].label || '') + '</h3><p>Всего сделок: <strong>' + bot.total_trades + '</strong> · Прибыльных: <strong>' + (bot.win_count ?? 0) + '</strong> · Убыточных: <strong>' + (bot.loss_count ?? 0) + '</strong> · Нулевых: <strong>' + (bot.neutral_count ?? 0) + '</strong></p>';
            bodyHtml += '<p>PnL: <strong>' + bot.total_pnl_usdt + ' USDT</strong> · Win Rate: <strong>' + bot.win_rate_pct + '%</strong> · Profit Factor: <strong>' + pfVal + '</strong></p>';
            bodyHtml += '<p>Средняя прибыль: <strong>' + (bot.avg_win_usdt != null ? bot.avg_win_usdt + ' USDT' : '—') + '</strong> · Средний убыток: <strong>' + (bot.avg_loss_usdt != null ? bot.avg_loss_usdt + ' USDT' : '—') + '</strong></p>';
            bodyHtml += '<p>Макс. серия побед: <strong>' + (series.max_consecutive_wins ?? 0) + '</strong> · Макс. серия убытков: <strong>' + (series.max_consecutive_losses ?? 0) + '</strong> · Просадка: <strong>' + (dd.max_drawdown_usdt ?? 0) + ' USDT</strong> (' + (dd.max_drawdown_pct ?? 0) + '%)</p>';
        } else {
            bodyHtml += '<p>Нет данных</p>';
        }
        bodyHtml += '</div>';

        const tradesList = bot.trades || [];
        bodyHtml += '<div class="analytics-section" data-category="trades_table"><h3>Таблица сделок</h3><p>Показано последних <strong>' + tradesList.length + '</strong> сделок (символ, дата выхода, направление, цены, объём, PnL, причина, источник, RSI, тренд).</p>';
        bodyHtml += '<div class="analytics-trades-table-wrap"><table class="analytics-trades-table"><thead><tr>';
        bodyHtml += '<th>Дата выхода</th><th>Символ</th><th>Направление</th><th>Вход</th><th>Выход</th><th>Объём USDT</th><th>PnL</th><th>Причина</th><th>Источник</th><th>RSI</th><th>Тренд</th></tr></thead><tbody>';
        tradesList.slice(-500).reverse().forEach(tr => {
            const pnlClass = (tr.pnl || 0) > 0 ? 'pnl-win' : ((tr.pnl || 0) < 0 ? 'pnl-loss' : '');
            bodyHtml += '<tr>';
            bodyHtml += '<td>' + (tr.exit_time_iso || '').replace('T', ' ').slice(0, 19) + '</td>';
            bodyHtml += '<td>' + (tr.symbol || '') + '</td><td>' + (tr.direction || '') + '</td>';
            bodyHtml += '<td>' + (tr.entry_price != null ? Number(tr.entry_price).toFixed(6) : '—') + '</td><td>' + (tr.exit_price != null ? Number(tr.exit_price).toFixed(6) : '—') + '</td>';
            bodyHtml += '<td>' + (tr.position_size_usdt != null ? Number(tr.position_size_usdt).toFixed(2) : '—') + '</td>';
            bodyHtml += '<td class="' + pnlClass + '">' + (tr.pnl != null ? Number(tr.pnl).toFixed(4) : '—') + '</td>';
            bodyHtml += '<td>' + (tr.close_reason || '—').slice(0, 20) + '</td><td>' + (tr.decision_source || '—').slice(0, 15) + '</td>';
            bodyHtml += '<td>' + (tr.entry_rsi != null ? tr.entry_rsi : '—') + '</td><td>' + (tr.entry_trend || '—') + '</td>';
            bodyHtml += '</tr>';
        });
        bodyHtml += '</tbody></table></div></div>';

        const bySymbol = bot.by_symbol || {};
        bodyHtml += '<div class="analytics-section" data-category="by_symbol"><h3>По символам</h3><p>Сделок, PnL, победы/убытки/нулевые, Win Rate по каждому символу.</p>';
        bodyHtml += '<div class="analytics-stats-table-wrap"><table class="analytics-stats-table"><thead><tr><th>Символ</th><th>Сделок</th><th>PnL USDT</th><th>Победы</th><th>Убытки</th><th>Нулевые</th><th>Win Rate %</th></tr></thead><tbody>';
        Object.entries(bySymbol).sort((a, b) => (b[1].count || 0) - (a[1].count || 0)).forEach(([sym, d]) => {
            const wr = (d.count && d.wins != null) ? ((d.wins / d.count) * 100).toFixed(1) : '—';
            const pnlClass = (d.pnl || 0) >= 0 ? 'pnl-win' : 'pnl-loss';
            bodyHtml += '<tr><td>' + sym + '</td><td>' + (d.count ?? 0) + '</td><td class="' + pnlClass + '">' + (d.pnl || 0).toFixed(2) + '</td><td>' + (d.wins ?? 0) + '</td><td>' + (d.losses ?? 0) + '</td><td>' + (d.neutral ?? 0) + '</td><td>' + wr + '</td></tr>';
        });
        bodyHtml += '</tbody></table></div></div>';

        const byBot = bot.by_bot || {};
        bodyHtml += '<div class="analytics-section" data-category="by_bot"><h3>По ботам</h3><p>Статистика по каждому bot_id.</p>';
        bodyHtml += '<div class="analytics-stats-table-wrap"><table class="analytics-stats-table"><thead><tr><th>Bot ID</th><th>Сделок</th><th>PnL USDT</th><th>Победы</th><th>Убытки</th><th>Нулевые</th><th>Win Rate %</th></tr></thead><tbody>';
        Object.entries(byBot).sort((a, b) => (b[1].count || 0) - (a[1].count || 0)).forEach(([bid, d]) => {
            const wr = (d.count && d.wins != null) ? ((d.wins / d.count) * 100).toFixed(1) : '—';
            const pnlClass = (d.pnl || 0) >= 0 ? 'pnl-win' : 'pnl-loss';
            bodyHtml += '<tr><td>' + bid + '</td><td>' + (d.count ?? 0) + '</td><td class="' + pnlClass + '">' + (d.pnl || 0).toFixed(2) + '</td><td>' + (d.wins ?? 0) + '</td><td>' + (d.losses ?? 0) + '</td><td>' + (d.neutral ?? 0) + '</td><td>' + wr + '</td></tr>';
        });
        bodyHtml += '</tbody></table></div></div>';

        const byDecision = bot.by_decision_source || {};
        bodyHtml += `<div class="analytics-section" data-category="by_decision_source"><h3>По источникам решений</h3><p>Статистика по источнику решения (FullAI, RSI, и т.д.).</p>`;
        bodyHtml += '<div class="analytics-stats-table-wrap"><table class="analytics-stats-table"><thead><tr><th>Источник</th><th>Сделок</th><th>PnL USDT</th><th>Победы</th><th>Убытки</th><th>Нулевые</th><th>Win Rate %</th></tr></thead><tbody>';
        Object.entries(byDecision).sort((a, b) => (b[1].count || 0) - (a[1].count || 0)).forEach(([src, d]) => {
            const wr = (d.count && d.wins != null) ? ((d.wins / d.count) * 100).toFixed(1) : '—';
            const pnlClass = (d.pnl || 0) >= 0 ? 'pnl-win' : 'pnl-loss';
            bodyHtml += `<tr><td>${src}</td><td>${d.count ?? 0}</td><td class="${pnlClass}">${(d.pnl || 0).toFixed(2)}</td><td>${d.wins ?? 0}</td><td>${d.losses ?? 0}</td><td>${d.neutral ?? 0}</td><td>${wr}</td></tr>`;
        });
        bodyHtml += '</tbody></table></div></div>';

        const byReason = bot.by_close_reason || {};
        bodyHtml += `<div class="analytics-section" data-category="reasons"><h3>Причины закрытия</h3>`;
        if (Object.keys(byReason).length) {
            bodyHtml += '<div class="analytics-stats-table-wrap"><table class="analytics-stats-table"><thead><tr><th>Причина</th><th>Сделок</th><th>PnL USDT</th><th>Победы</th><th>Убытки</th><th>Нулевые</th><th>Win Rate %</th></tr></thead><tbody>';
            for (const [reason, d] of Object.entries(byReason)) {
                const wr = (d.count && d.wins != null) ? ((d.wins / d.count) * 100).toFixed(1) : '—';
                const pnlClass = (d.pnl || 0) >= 0 ? 'pnl-win' : 'pnl-loss';
                bodyHtml += `<tr><td>${reason}</td><td>${d.count ?? 0}</td><td class="${pnlClass}">${(d.pnl || 0).toFixed(2)}</td><td>${d.wins ?? 0}</td><td>${d.losses ?? 0}</td><td>${d.neutral ?? 0}</td><td>${wr}</td></tr>`;
            }
            bodyHtml += '</tbody></table></div>';
        } else {
            bodyHtml += '<p>Нет данных</p>';
        }
        bodyHtml += '</div>';

        const uc = bot.unsuccessful_coins || [];
        bodyHtml += `<div class="analytics-section" data-category="unsuccessful_coins"><h3>${categories[7].label}</h3><p>(PnL &lt; 0 или Win Rate &lt; 45%, мин. 3 сделки)</p>`;
        if (uc.length) {
            bodyHtml += '<ul>';
            uc.forEach(c => {
                bodyHtml += `<li><strong>${c.symbol}</strong>: сделок ${c.trades_count}, PnL ${c.pnl_usdt} USDT, Win Rate ${c.win_rate_pct}%, причины: ${(c.reasons || []).join(', ')}</li>`;
            });
            bodyHtml += '</ul>';
        } else {
            bodyHtml += '<p>Нет неудачных монет по критериям</p>';
        }
        bodyHtml += '</div>';

        const us = bot.unsuccessful_settings || [];
        bodyHtml += `<div class="analytics-section" data-category="unsuccessful_settings"><h3>${categories[8].label}</h3>`;
        if (us.length) {
            us.forEach(u => {
                if (!u.bad_rsi_ranges?.length && !u.bad_trends?.length) return;
                bodyHtml += `<p><strong>${u.symbol}</strong></p><ul>`;
                (u.bad_rsi_ranges || []).forEach(r => {
                    bodyHtml += `<li>RSI ${r.rsi_range}: сделок ${r.trades_count}, PnL ${r.pnl_usdt}, Win Rate ${r.win_rate_pct}%</li>`;
                });
                (u.bad_trends || []).forEach(t => {
                    bodyHtml += `<li>Тренд ${t.trend}: сделок ${t.trades_count}, PnL ${t.pnl_usdt}, Win Rate ${t.win_rate_pct}%</li>`;
                });
                bodyHtml += '</ul>';
            });
        } else {
            bodyHtml += '<p>Нет данных</p>';
        }
        bodyHtml += '</div>';

        const sc = bot.successful_coins || [];
        bodyHtml += `<div class="analytics-section" data-category="successful_coins"><h3>${categories[9].label}</h3><p>(PnL &gt; 0 и Win Rate ≥ 55%, мин. 3 сделки)</p>`;
        if (sc.length) {
            bodyHtml += '<ul>';
            sc.forEach(c => {
                bodyHtml += `<li><strong>${c.symbol}</strong>: сделок ${c.trades_count}, PnL ${c.pnl_usdt} USDT, Win Rate ${c.win_rate_pct}%</li>`;
            });
            bodyHtml += '</ul>';
        } else {
            bodyHtml += '<p>Нет удачных монет по критериям</p>';
        }
        bodyHtml += '</div>';

        const ss = bot.successful_settings || [];
        bodyHtml += `<div class="analytics-section" data-category="successful_settings"><h3>${categories[10].label}</h3><p>(Диапазоны RSI и тренды с Win Rate ≥ 55% и PnL &gt; 0)</p>`;
        if (ss.length) {
            ss.forEach(u => {
                if (!u.good_rsi_ranges?.length && !u.good_trends?.length) return;
                bodyHtml += `<p><strong>${u.symbol}</strong></p><ul>`;
                (u.good_rsi_ranges || []).forEach(r => {
                    bodyHtml += `<li>RSI ${r.rsi_range}: сделок ${r.trades_count}, PnL ${r.pnl_usdt}, Win Rate ${r.win_rate_pct}%</li>`;
                });
                (u.good_trends || []).forEach(t => {
                    bodyHtml += `<li>Тренд ${t.trend}: сделок ${t.trades_count}, PnL ${t.pnl_usdt}, Win Rate ${t.win_rate_pct}%</li>`;
                });
                bodyHtml += '</ul>';
            });
        } else {
            bodyHtml += '<p>Нет данных</p>';
        }
        bodyHtml += '</div>';

        bodyHtml += `<div class="analytics-meta">Отчёт сформирован: ${report.generated_at || '—'}</div></div>`;

        container.innerHTML = tabsHtml + '<div class="analytics-report-wrap">' + bodyHtml + '</div>';
        container.querySelectorAll('.analytics-cat-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const cat = btn.dataset.category;
                container.querySelectorAll('.analytics-cat-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                container.querySelectorAll('.analytics-section').forEach(sec => {
                    sec.classList.toggle('active', sec.dataset.category === cat);
                });
            });
        });
        container.querySelectorAll('.analytics-section').forEach(sec => {
            sec.classList.toggle('active', sec.dataset.category === 'summary');
        });
    }

    /**
     * Инициализирует фильтры истории
     */
    initializeHistoryFilters() {
        // Фильтр по боту
        const botFilter = document.getElementById('historyBotFilter');
        if (botFilter && !botFilter.hasAttribute('data-listener-bound')) {
            botFilter.addEventListener('change', () => this.loadHistoryData(this.currentHistoryTab));
            botFilter.setAttribute('data-listener-bound', 'true');
        }

        // Фильтр по типу действия
        const actionFilter = document.getElementById('historyActionFilter');
        if (actionFilter && !actionFilter.hasAttribute('data-listener-bound')) {
            actionFilter.addEventListener('change', () => this.loadHistoryData(this.currentHistoryTab));
            actionFilter.setAttribute('data-listener-bound', 'true');
        }

        // Фильтр по периоду
        const dateFilter = document.getElementById('historyDateFilter');
        if (dateFilter && !dateFilter.hasAttribute('data-listener-bound')) {
            dateFilter.addEventListener('change', () => this.loadHistoryData(this.currentHistoryTab));
            dateFilter.setAttribute('data-listener-bound', 'true');
        }

        // Кнопки фильтров
        const applyBtn = document.getElementById('applyHistoryFilters');
        if (applyBtn && !applyBtn.hasAttribute('data-listener-bound')) {
            applyBtn.addEventListener('click', () => this.loadHistoryData(this.currentHistoryTab));
            applyBtn.setAttribute('data-listener-bound', 'true');
        }

        const clearBtn = document.getElementById('clearHistoryFilters');
        if (clearBtn && !clearBtn.hasAttribute('data-listener-bound')) {
            clearBtn.addEventListener('click', () => this.clearHistoryFilters());
            clearBtn.setAttribute('data-listener-bound', 'true');
        }

        const exportBtn = document.getElementById('exportHistoryBtn');
        if (exportBtn && !exportBtn.hasAttribute('data-listener-bound')) {
            exportBtn.addEventListener('click', () => this.exportHistoryData());
            exportBtn.setAttribute('data-listener-bound', 'true');
        }
    }

    /**
     * Инициализирует подвкладки истории
     */
    initializeHistorySubTabs() {
        const tabButtons = document.querySelectorAll('.history-tab-btn');
        const tabContents = document.querySelectorAll('.history-tab-content');

        tabButtons.forEach(button => {
            if (button.hasAttribute('data-listener-bound')) {
                return;
            }

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
                this.currentHistoryTab = tabName;
                this.loadHistoryData(tabName);
            });

            button.setAttribute('data-listener-bound', 'true');
        });
    }

    /**
     * Инициализирует кнопки действий истории
     */
    initializeHistoryActionButtons() {
        // Кнопка обновления
        const refreshBtn = document.getElementById('refreshHistoryBtn');
        if (refreshBtn && !refreshBtn.hasAttribute('data-listener-bound')) {
            refreshBtn.addEventListener('click', () => this.loadHistoryData(this.currentHistoryTab));
            refreshBtn.setAttribute('data-listener-bound', 'true');
        }

        // Кнопка создания демо-данных
        const demoBtn = document.getElementById('createDemoDataBtn');
        if (demoBtn && !demoBtn.hasAttribute('data-listener-bound')) {
            demoBtn.addEventListener('click', () => this.createDemoHistoryData());
            demoBtn.setAttribute('data-listener-bound', 'true');
        }

        // Кнопка очистки истории
        const clearBtn = document.getElementById('clearHistoryBtn');
        if (clearBtn && !clearBtn.hasAttribute('data-listener-bound')) {
            clearBtn.addEventListener('click', () => this.clearAllHistory());
            clearBtn.setAttribute('data-listener-bound', 'true');
        }
    }

    /**
     * Загружает данные истории
     */
    async loadHistoryData(tabName = null) {
        try {
            const targetTab = tabName || this.currentHistoryTab || 'actions';
            this.currentHistoryTab = targetTab;

            console.log(`[BotsManager] 📊 Загрузка данных истории: ${targetTab}`);
            
            // Получаем параметры фильтров
            const filters = this.getHistoryFilters();
            
            // Загружаем данные в зависимости от вкладки
            switch (targetTab) {
                case 'actions':
                    await this.loadBotActions(filters);
                    break;
                case 'trades':
                    await this.loadBotTrades(filters);
                    break;
                case 'signals':
                    await this.loadBotSignals(filters);
                    break;
                case 'ai':
                    await this.loadAIHistory();
                    break;
            }
            
            // Загружаем статистику (если не AI вкладка)
            if (targetTab !== 'ai') {
                await this.loadHistoryStatistics(filters);
            }
            
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
        
        const symbolValue = botFilter ? (botFilter.value || 'all') : 'all';
        const actionValueRaw = actionFilter ? (actionFilter.value || 'all') : 'all';
        const actionValue = actionValueRaw !== 'all' ? actionValueRaw.toUpperCase() : 'all';
        const periodValue = dateFilter ? (dateFilter.value || 'all') : 'all';

        const decisionSourceFilter = document.getElementById('historyDecisionSourceFilter');
        const resultFilter = document.getElementById('historyResultFilter');
        
        return {
            symbol: symbolValue,
            action_type: actionValue,
            trade_type: actionValue,
            period: periodValue,
            decision_source: decisionSourceFilter ? decisionSourceFilter.value : 'all',
            result: resultFilter ? resultFilter.value : 'all',
            limit: 100
        };
    }
    
    /**
     * Загружает AI историю
     */
    async loadAIHistory() {
        try {
            // Сначала загружаем статистику, чтобы использовать её как fallback для метрик
            await this.loadAIStats();
            // Затем загружаем остальные данные параллельно
            await Promise.all([
                this.loadAIDecisions(),
                this.loadAIOptimizerSummary(),
                this.loadAITrainingHistory(),
                this.loadAIPerformanceMetrics()
            ]);
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки AI истории:', error);
        }
    }
    /**
     * Загружает статистику AI vs скриптовые
     */
    async loadAIStats() {
        try {
            // Период из селектора
            const periodSelect = document.getElementById('aiPeriodSelect');
            const rawPeriod = periodSelect ? (periodSelect.value || '7d') : '7d';
            const periodMap = { '24h': 'today', '7d': 'week', '30d': 'month', 'all': 'all' };
            const period = periodMap[rawPeriod] || 'all';
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/stats?period=${encodeURIComponent(period)}`);
            const data = await response.json();
            
            if (data.success) {
                const aiStats = data.ai || {};
                const scriptStats = data.script || {};
                const comparisonStats = data.comparison || {};
                
                // Сохраняем данные AI для использования в метриках производительности
                this._lastAIStats = aiStats;
                
                // Обновляем UI
                const aiTotalEl = document.getElementById('aiTotalDecisions');
                const aiWinRateEl = document.getElementById('aiWinRate');
                const scriptTotalEl = document.getElementById('scriptTotalDecisions');
                const scriptWinRateEl = document.getElementById('scriptWinRate');
                const comparisonWinRateEl = document.getElementById('comparisonWinRate');
                const comparisonAvgPnlEl = document.getElementById('comparisonAvgPnl');
                const comparisonSummaryEl = document.getElementById('aiComparisonSummary');
                
                const aiTotal = Number(aiStats.total) || 0;
                const aiWinRate = typeof aiStats.win_rate === 'number' ? aiStats.win_rate : 0;
                const aiTotalPnL = Number(aiStats.total_pnl) || 0;
                const aiAvgPnL = Number(aiStats.avg_pnl) || 0;
                const scriptTotal = Number(scriptStats.total) || 0;
                const scriptWinRate = typeof scriptStats.win_rate === 'number' ? scriptStats.win_rate : 0;
                const scriptTotalPnL = Number(scriptStats.total_pnl) || 0;
                const scriptAvgPnL = Number(scriptStats.avg_pnl) || 0;
                
                // Обновляем карточку AI
                if (aiTotalEl) {
                    aiTotalEl.textContent = aiTotal;
                    const aiCard = aiTotalEl.closest('.stat-card');
                    if (aiCard) {
                        aiCard.classList.remove('profit', 'loss', 'neutral');
                        if (aiTotal > 0) {
                            aiCard.classList.add(aiWinRate >= 50 ? 'profit' : 'loss');
                        }
                    }
                }
                if (aiWinRateEl) {
                    aiWinRateEl.innerHTML = `Win Rate: <strong>${aiWinRate.toFixed(1)}%</strong>`;
                    if (aiTotalPnL !== 0) {
                        aiWinRateEl.innerHTML += `<br>Total PnL: <strong class="${aiTotalPnL >= 0 ? 'profit' : 'loss'}">${aiTotalPnL >= 0 ? '+' : ''}${aiTotalPnL.toFixed(2)} USDT</strong>`;
                    }
                }
                
                // Обновляем карточку Скриптовые
                if (scriptTotalEl) {
                    scriptTotalEl.textContent = scriptTotal;
                    const scriptCard = scriptTotalEl.closest('.stat-card');
                    if (scriptCard) {
                        scriptCard.classList.remove('profit', 'loss', 'neutral');
                        if (scriptTotal > 0) {
                            scriptCard.classList.add(scriptWinRate >= 50 ? 'profit' : 'loss');
                        }
                    }
                }
                if (scriptWinRateEl) {
                    scriptWinRateEl.innerHTML = `Win Rate: <strong>${scriptWinRate.toFixed(1)}%</strong>`;
                    if (scriptTotalPnL !== 0) {
                        scriptWinRateEl.innerHTML += `<br>Total PnL: <strong class="${scriptTotalPnL >= 0 ? 'profit' : 'loss'}">${scriptTotalPnL >= 0 ? '+' : ''}${scriptTotalPnL.toFixed(2)} USDT</strong>`;
                    }
                }
                
                const winRateDiff = Number(comparisonStats.win_rate_diff) || 0;
                const avgPnlDiff = Number(comparisonStats.avg_pnl_diff) || 0;
                const totalPnlDiff = Number(comparisonStats.total_pnl_diff) || 0;
                
                // Обновляем карточку Сравнение
                if (comparisonWinRateEl) {
                    const diffIcon = winRateDiff > 0 ? '📈' : winRateDiff < 0 ? '📉' : '➖';
                    comparisonWinRateEl.innerHTML = `${diffIcon} ${winRateDiff >= 0 ? '+' : ''}${winRateDiff.toFixed(1)}%`;
                    comparisonWinRateEl.className = `stat-value ${winRateDiff >= 0 ? 'profit' : winRateDiff < 0 ? 'loss' : 'neutral'}`;
                    
                    const comparisonCard = comparisonWinRateEl.closest('.stat-card');
                    if (comparisonCard) {
                        comparisonCard.classList.remove('profit', 'loss', 'neutral');
                        if (winRateDiff > 0) {
                            comparisonCard.classList.add('profit');
                        } else if (winRateDiff < 0) {
                            comparisonCard.classList.add('loss');
                        } else {
                            comparisonCard.classList.add('neutral');
                        }
                    }
                }
                
                if (comparisonAvgPnlEl) {
                    comparisonAvgPnlEl.innerHTML = `Avg PnL: <strong class="${avgPnlDiff >= 0 ? 'profit' : 'loss'}">${avgPnlDiff >= 0 ? '+' : ''}${avgPnlDiff.toFixed(2)} USDT</strong>`;
                    if (totalPnlDiff !== 0) {
                        comparisonAvgPnlEl.innerHTML += `<br>Total PnL: <strong class="${totalPnlDiff >= 0 ? 'profit' : 'loss'}">${totalPnlDiff >= 0 ? '+' : ''}${totalPnlDiff.toFixed(2)} USDT</strong>`;
                    }
                }

                if (comparisonSummaryEl) {
                    comparisonSummaryEl.textContent = this.buildAIComparisonSummary(aiStats, scriptStats, comparisonStats);
                    comparisonSummaryEl.classList.toggle('profit', winRateDiff > 0);
                    comparisonSummaryEl.classList.toggle('loss', winRateDiff < 0);
                }
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки статистики AI:', error);
            const summaryEl = document.getElementById('aiComparisonSummary');
            if (summaryEl) {
                summaryEl.textContent = 'Недостаточно данных для сравнения';
                summaryEl.classList.remove('profit', 'loss');
            }
        }
    }

    /**
     * Навешивает обработчик на селектор периода
     */
    initAIPeriodSelector() {
        const select = document.getElementById('aiPeriodSelect');
        if (!select || select._aiBound) return;
        select._aiBound = true;
        select.addEventListener('change', () => {
            this.loadAIHistory();
        });
    }
    
    /**
     * Загружает решения AI
     */
    async loadAIDecisions() {
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/decisions?limit=100`);
            const data = await response.json();
            
            if (data.success) {
                this.displayAIDecisions(data.decisions || []);
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки решений AI:', error);
        }
    }

    /**
     * Загружает результаты оптимизатора
     */
    async loadAIOptimizerSummary() {
        const paramsContainer = document.getElementById('optimizerParamsList');
        if (!paramsContainer) {
            return;
        }

        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/optimizer/results`);
            const data = await response.json();
            if (data.success) {
                this.displayAIOptimizerSummary(data);
            } else {
                this.displayAIOptimizerSummary(null);
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки результатов оптимизатора:', error);
            this.displayAIOptimizerSummary(null);
        }
    }

    /**
     * Отображает результаты оптимизатора
     */
    displayAIOptimizerSummary(data) {
        const paramsList = document.getElementById('optimizerParamsList');
        const topList = document.getElementById('optimizerTopSymbols');
        const patternsContainer = document.getElementById('optimizerPatternsSummary');
        const genomeVersionEl = document.getElementById('optimizerGenomeVersion');
        const updatedAtEl = document.getElementById('optimizerUpdatedAt');
        const maxTestsEl = document.getElementById('optimizerMaxTests');
        const symbolsCountEl = document.getElementById('optimizerSymbolsCount');

        const metadata = data?.metadata || {};
        if (genomeVersionEl) {
            genomeVersionEl.textContent = metadata.genome_version || '—';
        }
        if (updatedAtEl) {
            const updatedAt = metadata.optimized_params_updated_at || metadata.genome_updated_at;
            if (updatedAt) {
                updatedAtEl.textContent = `Обновлено: ${this.formatTimestamp(updatedAt)}`;
            } else {
                updatedAtEl.textContent = 'Обновлено: —';
            }
        }
        if (maxTestsEl) {
            maxTestsEl.textContent = metadata.max_tests || '—';
        }
        if (symbolsCountEl) {
            symbolsCountEl.textContent = `Оптимизировано монет: ${metadata.total_symbols_optimized || 0}`;
        }

        if (paramsList) {
            const optimizedParams = data?.optimized_params;
            if (optimizedParams && Object.keys(optimizedParams).length > 0) {
                // Словарь переводов и описаний параметров
                const paramLabels = {
                    'rsi_long_entry': { label: 'RSI вход LONG', desc: 'RSI для входа в длинную позицию' },
                    'rsi_long_exit': { label: 'RSI выход LONG', desc: 'RSI для выхода из длинной позиции' },
                    'rsi_short_entry': { label: 'RSI вход SHORT', desc: 'RSI для входа в короткую позицию' },
                    'rsi_short_exit': { label: 'RSI выход SHORT', desc: 'RSI для выхода из короткой позиции' },
                    'stop_loss_pct': { label: 'Стоп-лосс', desc: 'Процент стоп-лосса' },
                    'take_profit_pct': { label: 'Тейк-профит', desc: 'Процент тейк-профита' },
                    'position_size_pct': { label: 'Размер позиции', desc: 'Процент размера позиции от баланса' },
                    'best_trend': { label: 'Лучший тренд', desc: 'Наиболее прибыльный тренд' },
                    'trend_win_rate': { label: 'Win Rate тренда', desc: 'Процент прибыльных сделок по тренду' }
                };
                
                const formatValue = (value) => {
                    if (value === null || value === undefined) return '—';
                    if (typeof value === 'number') {
                        return Number.isInteger(value) ? value.toString() : value.toFixed(2);
                    }
                    return String(value);
                };
                
                paramsList.innerHTML = Object.entries(optimizedParams)
                    .filter(([key]) => key !== 'name') // Исключаем 'name' если есть
                    .map(([key, value]) => {
                        const paramInfo = paramLabels[key] || { label: key, desc: '' };
                        return `
                            <div class="optimizer-param" style="display:flex; justify-content:space-between; border-bottom:1px solid var(--border-color); padding:6px 0;">
                                <div style="flex:1;">
                                    <div style="font-weight:500;">${paramInfo.label}</div>
                                    ${paramInfo.desc ? `<small style="color:var(--text-muted,#888); font-size:11px;">${paramInfo.desc}</small>` : ''}
                                </div>
                                <strong style="margin-left:12px; font-size:14px;">${formatValue(value)}${typeof value === 'number' && (key.includes('pct') || key.includes('rate')) ? '%' : ''}</strong>
                            </div>
                        `;
                    }).join('');
            } else {
                paramsList.innerHTML = `
                    <div class="empty-history-state">
                        <div class="empty-icon">🧮</div>
                        <p>Параметры оптимизатора недоступны</p>
                        <small>Запустите оптимизацию стратегии для получения параметров</small>
                    </div>
                `;
            }
        }

        if (topList) {
            const topSymbols = Array.isArray(data?.top_symbols) ? data.top_symbols : [];
            if (topSymbols.length > 0) {
                const html = topSymbols.map(item => `
                    <div class="optimizer-symbol-item" style="border-bottom:1px solid var(--border-color); padding:6px 0;">
                        <div class="symbol-header" style="display:flex; justify-content:space-between; align-items:center;">
                            <strong>${item.symbol}</strong>
                            <span class="symbol-rating">⭐ ${item.rating?.toFixed(2) || '0.00'}</span>
                        </div>
                        <div class="symbol-details" style="display:flex; gap:12px; font-size:12px; color:var(--text-muted,#888);">
                            <span>Win Rate: ${item.win_rate?.toFixed(1) || '0.0'}%</span>
                            <span>Total PnL: ${item.total_pnl >= 0 ? '+' : ''}${(item.total_pnl || 0).toFixed(2)} USDT</span>
                        </div>
                        ${item.updated_at ? `<small style="color:var(--text-muted,#888);">Обновлено: ${this.formatTimestamp(item.updated_at)}</small>` : ''}
                    </div>
                `).join('');
                topList.innerHTML = html;
            } else {
                topList.innerHTML = `
                    <div class="empty-history-state">
                        <div class="empty-icon">📉</div>
                        <p>Нет оптимизированных монет</p>
                        <small>Запустите оптимизацию, чтобы увидеть результаты</small>
                    </div>
                `;
            }
        }

        if (patternsContainer) {
            const patterns = data?.trade_patterns;
            if (patterns) {
                const total = patterns.total_trades || 0;
                const winRate = patterns.win_rate || patterns.profitable_trades && total
                    ? (patterns.profitable_trades / total * 100)
                    : 0;
                patternsContainer.innerHTML = `
                    <div class="optimizer-patterns-card" style="background:var(--section-bg); border:1px solid var(--border-color); border-radius:12px; padding:12px;">
                        <div>Всего сделок: <strong>${total}</strong></div>
                        <div>Прибыльных: <strong>${patterns.profitable_trades || 0}</strong></div>
                        <div>Убыточных: <strong>${patterns.losing_trades || 0}</strong></div>
                        <div>Win Rate: <strong>${winRate?.toFixed(1) || '0.0'}%</strong></div>
                    </div>
                `;
            } else {
                patternsContainer.innerHTML = `
                    <div class="empty-history-state">
                        <div class="empty-icon">📊</div>
                        <p>Нет данных по паттернам</p>
                    </div>
                `;
            }
        }
    }

    /**
     * Загружает историю обучения AI
     */
    async loadAITrainingHistory() {
        const container = document.getElementById('aiTrainingHistoryList');
        if (!container) {
            return;
        }

        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/training-history?limit=10`);
            const data = await response.json();
            if (data.success) {
                this.displayAITrainingHistory(data.history || []);
            } else {
                this.displayAITrainingHistory([]);
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки истории обучения AI:', error);
            this.displayAITrainingHistory([]);
        }
    }

    /**
     * Отображает историю обучения AI
     */
    displayAITrainingHistory(history) {
        const container = document.getElementById('aiTrainingHistoryList');
        if (!container) return;

        if (!history || history.length === 0) {
            container.innerHTML = `
                <div class="empty-history-state">
                    <div class="empty-icon">🧠</div>
                    <p>История обучения не найдена</p>
                    <small>Запуски обучения AI появятся здесь</small>
                </div>
            `;
            this.updateAITrainingSummary(null);
            return;
        }

        const sorted = [...history].sort((a, b) => {
            return new Date(b.timestamp || b.started_at || 0) - new Date(a.timestamp || a.started_at || 0);
        });

        this.updateAITrainingSummary(sorted[0]);

        const html = sorted.map(record => {
            const startedAt = record.timestamp || record.started_at;
            const duration = record.duration_seconds ?? record.duration;
            
            // Извлекаем samples с учетом типа обучения
            let samples = record.samples || record.processed_samples || record.dataset_size;
            if (!samples && record.event_type === 'historical_data_training') {
                samples = record.candles || record.coins;
            }
            if (!samples && record.event_type === 'real_trades_training') {
                samples = record.trades;
            }
            
            const accuracy = record.accuracy !== undefined ? (record.accuracy * 100).toFixed(1) : record.metrics?.accuracy;
            const status = (record.status || 'done').toUpperCase();
            const { icon: statusIcon, className: statusClass } = this.getAITrainingStatusMeta(status);
            const eventLabel = this.getAITrainingEventLabel(record.event_type);

            const metrics = [];
            const trades = record.trades ?? record.processed_trades;
            if (typeof samples === 'number') {
                metrics.push(`Выборка: <strong>${samples}</strong>`);
            }
            if (typeof trades === 'number') {
                metrics.push(`Сделок: <strong>${trades}</strong>`);
            }
            if (typeof record.coins === 'number') {
                metrics.push(`Монет: <strong>${record.coins}</strong>`);
            }
            if (typeof record.candles === 'number') {
                metrics.push(`Свечей: <strong>${record.candles}</strong>`);
            }
            if (typeof record.models_saved === 'number') {
                metrics.push(`Моделей: <strong>${record.models_saved}</strong>`);
            }
            if (typeof record.errors === 'number') {
                metrics.push(`Ошибок: <strong>${record.errors}</strong>`);
            }
            if (record.accuracy !== undefined) {
                const accNumber = Number(record.accuracy);
                if (Number.isFinite(accNumber)) {
                    const accValue = accNumber <= 1 ? accNumber * 100 : accNumber;
                    metrics.push(`Точность: <strong>${accValue.toFixed(1)}%</strong>`);
                }
            } else if (accuracy) {
                metrics.push(`Точность: <strong>${accuracy}%</strong>`);
            }
            if (record.mse !== undefined) {
                metrics.push(`MSE: <strong>${Number(record.mse).toFixed(4)}</strong>`);
            }
            // Метрики ML модели параметров
            if (record.r2_score !== undefined) {
                metrics.push(`R²: <strong>${Number(record.r2_score).toFixed(3)}</strong>`);
            }
            if (record.avg_quality !== undefined) {
                metrics.push(`Качество: <strong>${Number(record.avg_quality).toFixed(3)}</strong>`);
            }
            if (typeof record.blocked_samples === 'number') {
                metrics.push(`Заблокировано: <strong>${record.blocked_samples}</strong>`);
            }
            if (typeof record.successful_samples === 'number') {
                metrics.push(`Успешных: <strong>${record.successful_samples}</strong>`);
            }
            // Использование ML модели для генерации параметров
            if (typeof record.ml_params_generated === 'number') {
                metrics.push(`🤖 ML параметров: <strong>${record.ml_params_generated}</strong>`);
            }
            if (record.ml_model_available === true) {
                metrics.push(`🤖 ML модель: <strong>активна</strong>`);
            } else if (record.ml_model_available === false) {
                metrics.push(`🤖 ML модель: <strong>недоступна</strong>`);
            }
            if (duration) {
                metrics.push(`Длительность: <strong>${this.formatDuration(duration)}</strong>`);
            }

            const metricsHtml = metrics.length
                ? `<div class="ai-training-metrics">${metrics.join(' • ')}</div>`
                : '';
            const reasonHtml = record.reason
                ? `<div class="history-details">Причина: ${record.reason}</div>`
                : '';
            const notesHtml = record.notes
                ? `<div class="history-details">${record.notes}</div>`
                : '';

            return `
                <div class="history-item ai-training-item ${statusClass}">
                    <div class="history-item-header">
                        <span>${statusIcon} ${status}</span>
                        <span class="history-timestamp">${this.formatTimestamp(startedAt)}</span>
                    </div>
                    <div class="history-item-subtitle">${eventLabel}</div>
                    <div class="history-item-content">
                        ${metricsHtml}
                        ${reasonHtml}
                        ${notesHtml}
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = html;
    }

    getAITrainingStatusMeta(status) {
        const normalized = (status || 'SUCCESS').toUpperCase();
        const meta = {
            'SUCCESS': { icon: '✅', className: 'success' },
            'FAILED': { icon: '❌', className: 'failed' },
            'SKIPPED': { icon: '⏸️', className: 'skipped' }
        };
        return meta[normalized] || meta.SUCCESS;
    }

    getAITrainingEventLabel(eventType) {
        if (!eventType) {
            return 'Обучение AI';
        }
        const normalized = eventType.toLowerCase();
        const labels = {
            'historical_data_training': '🗂️ Симуляция на истории',
            'history_trades_training': '📚 Обучение на истории сделок',
            'real_trades_training': '🤖 Реальные сделки с PnL',
            'ml_parameter_quality_training': '🤖 ML модель параметров'
        };
        return labels[normalized] || eventType;
    }

    /**
     * Обновляет карточку последнего обучения
     */
    updateAITrainingSummary(record) {
        const timeEl = document.getElementById('aiLastTrainingTime');
        const durationEl = document.getElementById('aiLastTrainingDuration');
        const samplesEl = document.getElementById('aiLastTrainingSamples');

        if (!record) {
            if (timeEl) timeEl.textContent = '—';
            if (durationEl) durationEl.textContent = 'Длительность: —';
            if (samplesEl) samplesEl.textContent = 'Выборка: —';
            return;
        }

        if (timeEl) {
            timeEl.textContent = this.formatTimestamp(record.timestamp || record.started_at) || '—';
        }
        if (durationEl) {
            const durationValue = record.duration || record.duration_seconds;
            durationEl.textContent = `Длительность: ${durationValue ? this.formatDuration(durationValue) : '—'}`;
        }
        if (samplesEl) {
            // Пробуем разные поля в зависимости от типа обучения
            let samples = record.samples || record.processed_samples || record.dataset_size;
            
            // Для historical_data_training может быть candles или coins
            if (!samples && record.event_type === 'historical_data_training') {
                // Приоритет: candles (более точный показатель), затем coins
                samples = record.candles || record.coins;
                if (samples && record.coins) {
                    // Показываем оба значения если есть
                    samplesEl.textContent = `Выборка: ${record.coins} монет, ${record.candles || 0} свечей`;
                    return;
                }
            }
            
            // Для real_trades_training может быть trades
            if (!samples && record.event_type === 'real_trades_training') {
                samples = record.trades;
            }
            
            if (samples !== undefined && samples !== null) {
                samplesEl.textContent = `Выборка: ${samples}`;
            } else {
                samplesEl.textContent = 'Выборка: —';
            }
        }
    }

    /**
     * Загружает метрики производительности AI
     */
    async loadAIPerformanceMetrics() {
        try {
            const periodSelect = document.getElementById('aiPeriodSelect');
            const rawPeriod = periodSelect ? (periodSelect.value || '7d') : '7d';
            const periodMap = { '24h': 'today', '7d': 'week', '30d': 'month', 'all': 'all' };
            const period = periodMap[rawPeriod] || 'all';
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/performance?period=${encodeURIComponent(period)}`);
            const data = await response.json();
            if (data.success) {
                this.displayAIPerformanceMetrics(data.metrics || {});
            } else {
                this.displayAIPerformanceMetrics({});
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки метрик AI:', error);
            this.displayAIPerformanceMetrics({});
        }
    }

    /**
     * Отображает метрики производительности AI
     */
    displayAIPerformanceMetrics(metrics) {
        const winRateEl = document.getElementById('aiOverallWinRate');
        const pnlEl = document.getElementById('aiOverallPnL');
        const decisionsEl = document.getElementById('aiOverallDecisions');
        const topSymbolsEl = document.getElementById('aiTopSymbols');

        let overall = metrics?.overall || {};
        
        // Если метрики пустые, используем данные из статистики как fallback
        if ((!overall.total_ai_decisions || overall.total_ai_decisions === 0) && this._lastAIStats) {
            const stats = this._lastAIStats;
            if (stats.total && stats.total > 0) {
                overall = {
                    total_ai_decisions: stats.total,
                    successful_decisions: stats.successful || 0,
                    failed_decisions: stats.failed || 0,
                    win_rate: stats.win_rate ? (stats.win_rate / 100) : 0,
                    win_rate_percent: stats.win_rate || 0,
                    total_pnl: stats.total_pnl,
                    avg_pnl: stats.avg_pnl
                };
            }
        }
        
        // Вычисляем Win Rate
        let winRate = overall.win_rate_percent;
        if (winRate === undefined || winRate === null) {
            const rawWinRate = overall.win_rate;
            if (rawWinRate !== undefined && rawWinRate !== null) {
                winRate = rawWinRate <= 1 ? rawWinRate * 100 : rawWinRate;
            } else {
                // Пробуем вычислить из successful/failed
                const successful = overall.successful_decisions;
                const failed = overall.failed_decisions;
                const total = overall.total_ai_decisions ?? overall.total_decisions;
                if (total && total > 0 && successful !== undefined && failed !== undefined) {
                    winRate = (successful / total) * 100;
                } else if (successful !== undefined && failed !== undefined && (successful + failed) > 0) {
                    winRate = (successful / (successful + failed)) * 100;
                }
            }
        }
        
        const formattedWinRate = (winRate !== undefined && winRate !== null && winRate > 0)
            ? `${Number(winRate).toFixed(1)}%`
            : '—';

        if (winRateEl) {
            winRateEl.textContent = formattedWinRate;
        }
        
        if (decisionsEl) {
            let totalDecisions = overall.total_ai_decisions ?? overall.total_decisions ?? null;
            if (totalDecisions === null) {
                const successful = overall.successful_decisions;
                const failed = overall.failed_decisions;
                if (successful !== undefined && successful !== null &&
                    failed !== undefined && failed !== null) {
                    totalDecisions = successful + failed;
                }
            }
            decisionsEl.textContent = `Решений: ${totalDecisions ?? '—'}`;
        }
        
        if (pnlEl) {
            // Приоритет: total_pnl, затем avg_pnl * total_decisions
            let totalPnL = overall.total_pnl;
            if (totalPnL === undefined || totalPnL === null) {
                const avgPnL = overall.avg_pnl;
                const totalDecisions = overall.total_ai_decisions ?? overall.total_decisions;
                if (avgPnL !== undefined && avgPnL !== null && totalDecisions && totalDecisions > 0) {
                    totalPnL = avgPnL * totalDecisions;
                }
            }
            
            pnlEl.textContent = (totalPnL !== undefined && totalPnL !== null)
                ? `Total PnL: ${(totalPnL >= 0 ? '+' : '')}${Number(totalPnL).toFixed(2)} USDT`
                : 'Total PnL: —';
        }

        // Топ монет по win rate / pnl
        if (topSymbolsEl) {
            const bySymbol = metrics.by_symbol || {};
            const entries = Object.entries(bySymbol);
            if (entries.length === 0) {
                topSymbolsEl.innerHTML = '';
            } else {
                const sorted = entries
                    .map(([symbol, m]) => ({ symbol, ...m }))
                    .sort((a, b) => (b.win_rate ?? 0) - (a.win_rate ?? 0))
                    .slice(0, 5);
                topSymbolsEl.innerHTML = `
                    <div style="border-top:1px dashed var(--border-color); margin-top:8px; padding-top:8px;">
                        <div style="font-weight:500; margin-bottom:6px;">Топ монет (AI):</div>
                        ${sorted.map(item => `
                            <div style="display:flex; justify-content:space-between; font-size:12px; margin:2px 0;">
                                <span>${item.symbol}</span>
                                <span>${(item.win_rate*100 || 0).toFixed(1)}% · ${(item.total_pnl >= 0 ? '+' : '')}${Number(item.total_pnl||0).toFixed(2)} USDT</span>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
        }
    }

    buildAIComparisonSummary(aiStats = {}, scriptStats = {}, comparison = {}) {
        const aiTotal = aiStats.total || 0;
        const scriptTotal = scriptStats.total || 0;
        if (!aiTotal && !scriptTotal) {
            return 'Недостаточно данных для сравнения';
        }
        if (!aiTotal) {
            return 'Скриптовые правила пока лидируют (AI ещё не открыл сделок)';
        }
        if (!scriptTotal) {
            return 'AI уже торгует, для скриптовых правил нет сделок';
        }

        const winDiff = Number(comparison.win_rate_diff || 0);
        const avgPnlDiff = Number(comparison.avg_pnl_diff || 0);
        const totalPnlDiff = Number(comparison.total_pnl_diff || 0);

        let leaderText = 'AI и скрипты показывают одинаковый результат';
        if (winDiff > 0) {
            leaderText = `🤖 AI опережает скрипты на ${winDiff.toFixed(1)}% по win rate`;
        } else if (winDiff < 0) {
            leaderText = `📜 Скриптовые правила пока впереди на ${Math.abs(winDiff).toFixed(1)}% по win rate`;
        }

        const parts = [];
        if (avgPnlDiff !== 0) {
            parts.push(`средний PnL ${avgPnlDiff >= 0 ? '+' : ''}${avgPnlDiff.toFixed(2)} USDT`);
        }
        if (totalPnlDiff !== 0) {
            parts.push(`общий PnL ${totalPnlDiff >= 0 ? '+' : ''}${totalPnlDiff.toFixed(2)} USDT`);
        }
        
        const pnlText = parts.length > 0 ? `, ${parts.join(', ')}` : '';

        return `${leaderText}${pnlText}.`;
    }
    
    /**
     * Отображает решения AI
     */
    displayAIDecisions(decisions) {
        const container = document.getElementById('aiDecisionsList');
        if (!container) return;
        
        if (decisions.length === 0) {
            container.innerHTML = `
                <div class="empty-history-state">
                    <div class="empty-icon">🤖</div>
                    <p>Решения AI не найдены</p>
                    <small>Решения AI будут отображаться здесь</small>
                </div>
            `;
            return;
        }
        
        const html = decisions.map(decision => {
            const status = decision.status || 'PENDING';
            const statusClass = status === 'SUCCESS' ? 'success' : status === 'FAILED' ? 'failed' : 'pending';
            const statusIcon = status === 'SUCCESS' ? '✅' : status === 'FAILED' ? '❌' : '⏳';
            
            return `
            <div class="history-item ai-decision-item ${statusClass}">
                <div class="history-item-header">
                    <span class="ai-decision-symbol">${decision.symbol || 'N/A'}</span>
                    <span class="ai-decision-status">${statusIcon} ${status}</span>
                    <span class="history-timestamp">${this.formatTimestamp(decision.timestamp)}</span>
                </div>
                <div class="history-item-content">
                    <div class="ai-decision-details">
                        <div>Направление: <strong>${decision.direction || 'N/A'}</strong></div>
                        <div>RSI: ${decision.rsi?.toFixed(2) || 'N/A'}</div>
                        <div>Тренд: ${decision.trend || 'N/A'}</div>
                        <div>Цена: ${decision.price?.toFixed(4) || 'N/A'}</div>
                        ${decision.ai_confidence ? `<div>Уверенность AI: <strong>${(decision.ai_confidence * 100).toFixed(0)}%</strong></div>` : ''}
                        ${decision.pnl !== undefined ? `<div class="trade-pnl ${decision.pnl >= 0 ? 'profit' : 'loss'}">PnL: ${decision.pnl.toFixed(2)} USDT</div>` : ''}
                        ${decision.roi !== undefined ? `<div class="trade-roi ${decision.roi >= 0 ? 'profit' : 'loss'}">ROI: ${decision.roi.toFixed(2)}%</div>` : ''}
                    </div>
                </div>
            </div>
        `;
        }).join('');
        
        container.innerHTML = html;
    }
    /**
     * Загружает действия ботов
     */
    async loadBotActions(filters) {
        try {
            const params = new URLSearchParams();
            if (filters.symbol && filters.symbol !== 'all') params.append('symbol', filters.symbol);
            if (filters.action_type && filters.action_type !== 'all') params.append('action_type', filters.action_type);
            if (filters.period && filters.period !== 'all') params.append('period', filters.period);
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
            if (filters.period && filters.period !== 'all') params.append('period', filters.period);
            params.append('limit', filters.limit);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/trades?${params}`);
            const data = await response.json();
            
            if (data.success) {
                let trades = data.trades || [];
                
                // Фильтруем по источнику решения
                if (filters.decision_source && filters.decision_source !== 'all') {
                    trades = trades.filter(t => t.decision_source === filters.decision_source);
                }
                
                // Фильтруем по результату
                if (filters.result && filters.result !== 'all') {
                    if (filters.result === 'successful') {
                        trades = trades.filter(t => t.is_successful === true || (t.pnl !== null && t.pnl > 0));
                    } else if (filters.result === 'failed') {
                        trades = trades.filter(t => t.is_successful === false || (t.pnl !== null && t.pnl <= 0));
                    }
                }
                
                this.displayBotTrades(trades);
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
            if (filters.period && filters.period !== 'all') params.append('period', filters.period);
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
    async loadHistoryStatistics(filters = {}) {
        try {
            const params = new URLSearchParams();
            const symbol = filters?.symbol;
            const period = filters?.period;

            if (symbol && symbol !== 'all') params.append('symbol', symbol);
            if (period && period !== 'all') params.append('period', period);
            
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
    }

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
     */
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
    
    // ==========================================
    // МЕТОДЫ ДЛЯ РАБОТЫ С ЛИМИТНЫМИ ОРДЕРАМИ
    // ==========================================
    
    initializeLimitOrdersUI() {
        try {
            // ✅ Защита от повторной инициализации
            const toggleEl = document.getElementById('limitOrdersEntryEnabled');
            if (!toggleEl) {
                console.warn('[BotsManager] ⚠️ Элемент limitOrdersEntryEnabled не найден');
                return;
            }
            
            // Проверяем, не инициализирован ли уже обработчик
            if (toggleEl.hasAttribute('data-limit-orders-ui-initialized')) {
                return; // Уже инициализирован
            }
            toggleEl.setAttribute('data-limit-orders-ui-initialized', 'true');
            
            const configDiv = document.getElementById('limitOrdersConfig');
            const positionSizeEl = document.getElementById('defaultPositionSize');
            const positionModeEl = document.getElementById('defaultPositionMode');
            
            // Безопасная проверка - если элементов нет, просто выходим
            if (!configDiv) {
                console.warn('[BotsManager] ⚠️ Элемент limitOrdersConfig не найден');
                return;
            }
            
            // Обработчик переключателя
            const updateUIState = (isEnabled) => {
                configDiv.style.display = isEnabled ? 'block' : 'none';
                
                // Деактивируем настройку "Размер позиции" при включении лимитных ордеров
                if (positionSizeEl) {
                    positionSizeEl.disabled = isEnabled;
                    positionSizeEl.style.opacity = isEnabled ? '0.5' : '1';
                    positionSizeEl.style.cursor = isEnabled ? 'not-allowed' : 'text';
                }
                if (positionModeEl) {
                    positionModeEl.disabled = isEnabled;
                    positionModeEl.style.opacity = isEnabled ? '0.5' : '1';
                    positionModeEl.style.cursor = isEnabled ? 'not-allowed' : 'pointer';
                }
                
                // Деактивируем кнопку "По умолчанию" когда toggle выключен
                const resetBtn = document.getElementById('resetLimitOrdersBtn');
                if (resetBtn) {
                    resetBtn.disabled = !isEnabled;
                    resetBtn.style.opacity = isEnabled ? '1' : '0.5';
                    resetBtn.style.cursor = isEnabled ? 'pointer' : 'not-allowed';
                }
            };
            
            toggleEl.addEventListener('change', () => {
                // ✅ Пропускаем обработку, если это программное изменение (при загрузке конфигурации)
                if (this.isProgrammaticChange) {
                    return;
                }
                
                const isEnabled = toggleEl.checked;
                updateUIState(isEnabled);
                
                if (isEnabled && document.getElementById('limitOrdersList').children.length === 0) {
                    // Добавляем первую пару полей
                    try {
                        this.addLimitOrderRow();
                    } catch (e) {
                        console.error('[BotsManager] ❌ Ошибка добавления строки:', e);
                    }
                }
            });
            
            // ✅ Инициализируем состояние при загрузке БЕЗ триггера события change
            // Просто обновляем UI визуально, не меняя значение toggle
            const currentChecked = toggleEl.checked;
            updateUIState(currentChecked);
            
            // ✅ Обработчик кнопки добавления - используем делегирование событий для надежности
            // Это работает даже если кнопка находится в скрытом контейнере или добавляется динамически
            const setupAddButtonHandler = () => {
                const addBtn = document.getElementById('addLimitOrderBtn');
                if (addBtn) {
                    // Проверяем, не добавлен ли уже обработчик
                    if (addBtn.hasAttribute('data-handler-attached')) {
                        console.log('[BotsManager] ℹ️ Обработчик кнопки уже установлен');
                        return;
                    }
                    
                    // Добавляем новый обработчик
                    addBtn.addEventListener('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        try {
                            console.log('[BotsManager] ➕ Клик по кнопке добавления ордера');
                            this.addLimitOrderRow();
                            // ✅ Триггерим автосохранение при добавлении строки
                            if (!this.isProgrammaticChange) {
                                this.updateFloatingSaveButtonVisibility();
                            }
                        } catch (error) {
                            console.error('[BotsManager] ❌ Ошибка добавления строки лимитного ордера:', error);
                            console.error('[BotsManager] Stack trace:', error.stack);
                        }
                    });
                    addBtn.setAttribute('data-handler-attached', 'true');
                    console.log('[BotsManager] ✅ Обработчик кнопки добавления ордера установлен');
                } else {
                    console.warn('[BotsManager] ⚠️ Кнопка addLimitOrderBtn не найдена, попытка повторной инициализации через 100мс');
                    // Пробуем еще раз через небольшую задержку (на случай, если элемент еще не загружен)
                    setTimeout(setupAddButtonHandler, 100);
                }
            };
            
            // Пытаемся установить обработчик сразу
            setupAddButtonHandler();
            
            // ✅ Дополнительно: делегирование событий на родительском контейнере для надежности
            // Это работает даже если кнопка находится в скрытом контейнере
            if (configDiv) {
                configDiv.addEventListener('click', (e) => {
                    // Проверяем, был ли клик по кнопке добавления
                    if (e.target && (e.target.id === 'addLimitOrderBtn' || e.target.closest('#addLimitOrderBtn'))) {
                        e.preventDefault();
                        e.stopPropagation();
                        try {
                            console.log('[BotsManager] ➕ Клик по кнопке добавления ордера (через делегирование)');
                            this.addLimitOrderRow();
                            // ✅ Триггерим автосохранение при добавлении строки
                            if (!this.isProgrammaticChange) {
                                this.updateFloatingSaveButtonVisibility();
                            }
                        } catch (error) {
                            console.error('[BotsManager] ❌ Ошибка добавления строки лимитного ордера (делегирование):', error);
                            console.error('[BotsManager] Stack trace:', error.stack);
                        }
                    }
                });
                console.log('[BotsManager] ✅ Делегирование событий для кнопки добавления установлено');
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка инициализации UI лимитных ордеров:', error);
        }
    }
    
    addLimitOrderRow(percent = 0, margin = 0) {
        console.log('[BotsManager] ➕ addLimitOrderRow вызван с параметрами:', { percent, margin });
        const listEl = document.getElementById('limitOrdersList');
        if (!listEl) {
            console.error('[BotsManager] ❌ Элемент limitOrdersList не найден!');
            return;
        }
        console.log('[BotsManager] ✅ Элемент limitOrdersList найден, текущее количество строк:', listEl.children.length);
        
        const row = document.createElement('div');
        row.className = 'limit-order-row';
        row.style.cssText = 'display: flex; gap: 10px; align-items: center; padding: 10px; background: #2a2a2a; border-radius: 5px;';
        
        row.innerHTML = `
            <div style="flex: 1;">
                <label style="display: block; margin-bottom: 5px; color: #fff;">% от входа:</label>
                <input type="number" class="limit-order-percent" value="${percent}" step="0.1" min="0" max="100" 
                       style="width: 100%; padding: 5px; background: #1a1a1a; color: #fff; border: 1px solid #404040; border-radius: 3px;">
            </div>
            <div style="flex: 1;">
                <label style="display: block; margin-bottom: 5px; color: #fff;">Сумма (USDT): <small style="color: #ffc107; font-size: 11px;">⚠️ Минимум 5 USDT</small></label>
                <input type="number" class="limit-order-margin" value="${margin}" step="0.1" min="5" 
                       placeholder="Минимум 5 USDT"
                       style="width: 100%; padding: 5px; background: #1a1a1a; color: #fff; border: 1px solid #404040; border-radius: 3px;">
                <small class="limit-order-margin-error" style="display: none; color: #dc3545; font-size: 11px; margin-top: 3px;">⚠️ Минимум 5 USDT (требование биржи Bybit)</small>
            </div>
            <button type="button" class="remove-limit-order-btn" style="padding: 10px 15px; background: #dc3545; color: #fff; border: none; border-radius: 3px; cursor: pointer; margin-top: 20px;">
                ➖
            </button>
        `;
        
        // Обработчик удаления
        row.querySelector('.remove-limit-order-btn').addEventListener('click', () => {
            const listEl = document.getElementById('limitOrdersList');
            // Не удаляем, если это последняя строка - оставляем хотя бы одну
            if (listEl && listEl.children.length > 1) {
                row.remove();
                // ✅ Триггерим автосохранение при удалении строки
                if (!this.isProgrammaticChange) {
                    this.updateFloatingSaveButtonVisibility();
                }
            } else {
                // Если это последняя строка, просто очищаем значения
                row.querySelector('.limit-order-percent').value = 0;
                row.querySelector('.limit-order-margin').value = 0;
                // ✅ Триггерим автосохранение при очистке значений последней строки
                if (!this.isProgrammaticChange) {
                    this.updateFloatingSaveButtonVisibility();
                }
            }
        });
        
        listEl.appendChild(row);
        console.log('[BotsManager] ✅ Строка добавлена в DOM, новое количество строк:', listEl.children.length);
        
        // ✅ ДОБАВЛЯЕМ АВТОСОХРАНЕНИЕ ДЛЯ ДИНАМИЧЕСКИХ ПОЛЕЙ
        // Находим новые поля и добавляем обработчики автосохранения
        const percentInput = row.querySelector('.limit-order-percent');
        const marginInput = row.querySelector('.limit-order-margin');
        
        if (percentInput && !percentInput.hasAttribute('data-autosave-initialized')) {
            percentInput.setAttribute('data-autosave-initialized', 'true');
            percentInput.addEventListener('blur', () => {
                if (!this.isProgrammaticChange) {
                    this.updateFloatingSaveButtonVisibility();
                }
            });
        }
        
        if (marginInput && !marginInput.hasAttribute('data-autosave-initialized')) {
            marginInput.setAttribute('data-autosave-initialized', 'true');
            const errorMsg = row.querySelector('.limit-order-margin-error');
            
            // Валидация при вводе (только подсветка, без автосохранения)
            marginInput.addEventListener('input', () => {
                const value = parseFloat(marginInput.value) || 0;
                if (value > 0 && value < 5) {
                    marginInput.style.borderColor = '#dc3545';
                    if (errorMsg) errorMsg.style.display = 'block';
                } else {
                    marginInput.style.borderColor = '#404040';
                    if (errorMsg) errorMsg.style.display = 'none';
                }
            });
            
            marginInput.addEventListener('blur', () => {
                const value = parseFloat(marginInput.value) || 0;
                if (value > 0 && value < 5) {
                    marginInput.value = 5;
                    marginInput.style.borderColor = '#404040';
                    if (errorMsg) errorMsg.style.display = 'none';
                    this.showNotification('⚠️ Сумма лимитного ордера увеличена до минимума 5 USDT (требование биржи Bybit)', 'warning');
                }
                if (!this.isProgrammaticChange) {
                    this.updateFloatingSaveButtonVisibility();
                }
            });
        }
    }
    
    async saveLimitOrdersSettings() {
        try {
            const enabled = document.getElementById('limitOrdersEntryEnabled').checked;
            const rows = document.querySelectorAll('.limit-order-row');
            
            const percentSteps = [];
            const marginAmounts = [];
            
            // ✅ ВАЛИДАЦИЯ: Проверяем что все суммы >= 5 USDT (кроме рыночного ордера с percent_step = 0)
            const validationErrors = [];
            rows.forEach((row, index) => {
                const percent = parseFloat(row.querySelector('.limit-order-percent').value) || 0;
                const margin = parseFloat(row.querySelector('.limit-order-margin').value) || 0;
                
                // Для лимитных ордеров (percent > 0) проверяем минимум 5 USDT
                if (percent > 0 && margin > 0 && margin < 5) {
                    validationErrors.push(`Ордер #${index + 1} (${percent}%): сумма ${margin} USDT меньше минимума 5 USDT`);
                    // Подсвечиваем поле с ошибкой
                    const marginInput = row.querySelector('.limit-order-margin');
                    if (marginInput) {
                        marginInput.style.borderColor = '#dc3545';
                        const errorMsg = row.querySelector('.limit-order-margin-error');
                        if (errorMsg) errorMsg.style.display = 'block';
                    }
                }
                
                percentSteps.push(percent);
                marginAmounts.push(margin);
            });
            
            // Если есть ошибки валидации - показываем их и не сохраняем
            if (validationErrors.length > 0) {
                const errorText = `❌ Ошибка валидации:\n${validationErrors.join('\n')}\n\n⚠️ Минимум 5 USDT на ордер (требование биржи Bybit)`;
                this.showNotification(errorText, 'error');
                console.error('[BotsManager] ❌ Ошибки валидации лимитных ордеров:', validationErrors);
                return; // Не сохраняем, если есть ошибки
            }
            
            // Если включен режим, но нет ордеров - выключаем режим
            const finalEnabled = enabled && percentSteps.length > 0 && marginAmounts.some(m => m > 0);
            
            const config = {
                limit_orders_entry_enabled: finalEnabled,
                limit_orders_percent_steps: percentSteps,
                limit_orders_margin_amounts: marginAmounts
            };
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                this.showNotification('✅ Настройки набора позиций сохранены', 'success');
                await this.loadConfigurationData();
            } else {
                throw new Error('Ошибка сохранения');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сохранения настроек лимитных ордеров:', error);
            this.showNotification('❌ Ошибка сохранения настроек', 'error');
        }
    }
    
    resetLimitOrdersToDefault() {
        try {
            // Проверяем, включен ли режим лимитных ордеров
            const toggleEl = document.getElementById('limitOrdersEntryEnabled');
            if (!toggleEl || !toggleEl.checked) {
                this.showNotification('⚠️ Сначала включите режим набора позиций лимитными ордерами', 'warning');
                return;
            }
            
            // Дефолтные значения из bot_config.py (минимум 5 USDT на ордер - требование биржи Bybit)
            const defaultPercentSteps = [0, 0.5, 1, 1.5, 2];
            const defaultMarginAmounts = [5, 5, 5, 5, 5];
            
            // НЕ меняем состояние toggle - он должен оставаться включенным!
            
            // ✅ Устанавливаем флаг программного изменения, чтобы не триггерить автосохранение при добавлении строк
            this.isProgrammaticChange = true;
            
            // Очищаем список ордеров
            const limitOrdersList = document.getElementById('limitOrdersList');
            if (limitOrdersList) {
                limitOrdersList.innerHTML = '';
                
                // Добавляем дефолтные ордера
                defaultPercentSteps.forEach((percent, index) => {
                    this.addLimitOrderRow(percent, defaultMarginAmounts[index]);
                });
            }
            
            // ✅ Сбрасываем флаг и триггерим автосохранение после завершения сброса
            this.isProgrammaticChange = false;
            this.updateFloatingSaveButtonVisibility();
            
            this.showNotification('✅ Настройки сброшены к значениям по умолчанию', 'success');
            console.log('[BotsManager] ✅ Лимитные ордера сброшены к значениям по умолчанию');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка сброса лимитных ордеров:', error);
            this.showNotification('❌ Ошибка сброса: ' + error.message, 'error');
            // ✅ Сбрасываем флаг в случае ошибки
            this.isProgrammaticChange = false;
        }
    }
    
    // ==========================================
    // УПРАВЛЕНИЕ ТАЙМФРЕЙМОМ СИСТЕМЫ
    // ==========================================
    
    /**
     * Загружает текущий таймфрейм системы
     */
    async loadTimeframe() {
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/timeframe`);
            const data = await response.json();
            
            if (data.success) {
                // Сохраняем таймфрейм в переменную класса
                this.currentTimeframe = data.timeframe;
                
                const timeframeSelect = document.getElementById('systemTimeframe');
                if (timeframeSelect) {
                    timeframeSelect.value = data.timeframe;
                    console.log('[BotsManager] ✅ Текущий таймфрейм загружен:', data.timeframe);
                }
                return data.timeframe;
            } else {
                console.error('[BotsManager] ❌ Ошибка загрузки таймфрейма:', data.error);
                this.currentTimeframe = '6h'; // Дефолтное значение
                return '6h';
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка запроса таймфрейма:', error);
            this.currentTimeframe = '6h'; // Дефолтное значение
            return '6h';
        }
    }
    
    /**
     * Применяет новый таймфрейм системы
     */
    async applyTimeframe() {
        const timeframeSelect = document.getElementById('systemTimeframe');
        const applyBtn = document.getElementById('applyTimeframeBtn');
        const statusDiv = document.getElementById('timeframeStatus');
        
        if (!timeframeSelect || !applyBtn) {
            console.error('[BotsManager] ❌ Элементы управления таймфреймом не найдены');
            return;
        }
        
        const newTimeframe = timeframeSelect.value;
        const oldTimeframe = applyBtn.dataset.currentTimeframe || '6h';
        
        if (newTimeframe === oldTimeframe) {
            this.showNotification('ℹ️ Таймфрейм не изменился', 'info');
            return;
        }
        
        // Показываем статус
        if (statusDiv) {
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '<div style="color: #ffa500;">⏳ Переключение таймфрейма... Сохранение данных...</div>';
        }
        
        applyBtn.disabled = true;
        applyBtn.innerHTML = '<span>⏳ Применение...</span>';
        
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/timeframe`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ timeframe: newTimeframe })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Обновляем текущий таймфрейм в переменной класса
                this.currentTimeframe = newTimeframe;
                
                // Обновляем текущий таймфрейм
                applyBtn.dataset.currentTimeframe = newTimeframe;
                
                // Показываем успешный статус
                if (statusDiv) {
                    statusDiv.innerHTML = `<div style="color: #4CAF50;">✅ Таймфрейм изменен: ${oldTimeframe} → ${newTimeframe}</div>`;
                }
                
                this.showNotification(`✅ Таймфрейм изменен: ${oldTimeframe} → ${newTimeframe}. Данные сохранены, начинается перезагрузка RSI...`, 'success');
                
                // Обновляем все упоминания таймфрейма в интерфейсе
                this.updateTimeframeInUI(newTimeframe);
                
                // Перезагружаем RSI данные через небольшую задержку
                setTimeout(async () => {
                    if (statusDiv) {
                        statusDiv.innerHTML += '<div style="color: #2196F3; margin-top: 5px;">🔄 Перезагрузка RSI данных...</div>';
                    }
                    
                    // Триггерим обновление RSI данных с принудительной перезагрузкой
                    // Очищаем кэш и перезагружаем данные
                    this.coinsRsiData = [];
                    
                    // Запрашиваем полное обновление RSI на сервере (не refresh-rsi/all — символ "all" не поддерживается API биржи)
                    try {
                        const refreshResponse = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/refresh-rsi-all`, {
                            method: 'POST'
                        });
                        if (refreshResponse.ok) {
                            console.log('[BotsManager] ✅ Запрошено полное обновление RSI на сервере');
                        }
                    } catch (refreshError) {
                        console.warn('[BotsManager] ⚠️ Не удалось запросить обновление RSI:', refreshError);
                    }
                    
                    // Перезагружаем данные через небольшую задержку
                    setTimeout(() => {
                        this.loadCoinsRsiData(true);
                    }, 2000);
                    
                    // Через еще немного времени скрываем статус
                    setTimeout(() => {
                        if (statusDiv) {
                            statusDiv.style.display = 'none';
                        }
                    }, 5000);
                }, 500);
                
                console.log('[BotsManager] ✅ Таймфрейм успешно изменен:', data);
            } else {
                throw new Error(data.error || 'Неизвестная ошибка');
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка изменения таймфрейма:', error);
            this.showNotification('❌ Ошибка изменения таймфрейма: ' + error.message, 'error');
            
            if (statusDiv) {
                statusDiv.innerHTML = `<div style="color: #f44336;">❌ Ошибка: ${error.message}</div>`;
            }
        } finally {
            applyBtn.disabled = false;
            applyBtn.innerHTML = '<span>✅ Применить таймфрейм</span>';
        }
    }
    
    /**
     * Обновляет все упоминания таймфрейма в интерфейсе
     */
    updateTimeframeInUI(timeframe) {
        // Обновляем отображение текущего таймфрейма в заголовке списка монет
        const timeframeDisplay = document.getElementById('currentTimeframeDisplay');
        if (timeframeDisplay) {
            timeframeDisplay.textContent = timeframe.toUpperCase();
        }
        
        // ✅ КРИТИЧНО: Обновляем весь заголовок "Монеты (RSI XH)" с учетом перевода
        const coinsHeader = document.querySelector('h3[data-translate="coins_rsi_6h"]');
        if (coinsHeader) {
            const currentLang = document.documentElement.lang || 'ru';
            const translationKey = 'coins_rsi_6h';
            if (typeof TRANSLATIONS !== 'undefined' && TRANSLATIONS[currentLang] && TRANSLATIONS[currentLang][translationKey]) {
                // Используем перевод, но заменяем таймфрейм
                let translatedText = TRANSLATIONS[currentLang][translationKey];
                // Заменяем 6H на текущий таймфрейм в переводе
                translatedText = translatedText.replace(/6[hH]/gi, timeframe.toUpperCase());
                // Обновляем заголовок, сохраняя структуру с span
                const timeframeSpan = coinsHeader.querySelector('#currentTimeframeDisplay');
                if (timeframeSpan) {
                    // Обновляем только текст до и после span
                    const parts = translatedText.split(/6[hH]/i);
                    if (parts.length >= 2) {
                        coinsHeader.innerHTML = `${parts[0]}<span id="currentTimeframeDisplay">${timeframe.toUpperCase()}</span>${parts.slice(1).join('')}`;
                    } else {
                        // Если формат не совпадает, просто обновляем span
                        timeframeSpan.textContent = timeframe.toUpperCase();
                    }
                } else {
                    // Если span нет, обновляем весь текст
                    coinsHeader.textContent = translatedText.replace(/6[hH]/gi, timeframe.toUpperCase());
                }
            } else {
                // Если переводов нет, просто обновляем span
                if (timeframeDisplay) {
                    timeframeDisplay.textContent = timeframe.toUpperCase();
                }
            }
        }
        
        // Обновляем отображение таймфрейма в деталях монеты
        const selectedCoinTimeframeDisplay = document.getElementById('selectedCoinTimeframeDisplay');
        if (selectedCoinTimeframeDisplay) {
            selectedCoinTimeframeDisplay.textContent = timeframe.toUpperCase();
        }
        
        // Обновляем select с таймфреймом
        const timeframeSelect = document.getElementById('systemTimeframe');
        if (timeframeSelect) {
            timeframeSelect.value = timeframe;
        }
        
        // Обновляем кнопку применения
        const applyBtn = document.getElementById('applyTimeframeBtn');
        if (applyBtn) {
            applyBtn.dataset.currentTimeframe = timeframe;
        }
        
        // Если есть выбранная монета, обновляем её информацию
        if (this.selectedCoin) {
            this.updateCoinInfo(this.selectedCoin);
        }
        
        // Обновляем заголовки и описания с упоминанием таймфрейма
        const timeframeElements = document.querySelectorAll('[data-timeframe-placeholder]');
        timeframeElements.forEach(el => {
            const placeholder = el.getAttribute('data-timeframe-placeholder');
            if (placeholder === '6h' || placeholder === '6H') {
                // Обновляем только текст, не трогая структуру HTML
                const textNodes = this.getTextNodes(el);
                textNodes.forEach(node => {
                    if (node.textContent.includes('6H') || node.textContent.includes('6h')) {
                        node.textContent = node.textContent.replace(/6[hH]/g, timeframe.toUpperCase());
                    }
                });
            }
        });
        
        // Обновляем заголовки с RSI (дополнительная проверка)
        const rsiHeaders = document.querySelectorAll('h3');
        rsiHeaders.forEach(header => {
            // Пропускаем заголовок, который уже обновлен выше
            if (header === coinsHeader) return;
            
            if (header.textContent.includes('RSI 6H') || header.textContent.includes('RSI 6h')) {
                header.textContent = header.textContent.replace(/RSI 6[hH]/g, `RSI ${timeframe.toUpperCase()}`);
            }
        });
        
        // Обновляем описания в help текстах
        const helpTexts = document.querySelectorAll('.config-help, small');
        helpTexts.forEach(el => {
            if (el.textContent.includes('6H') || el.textContent.includes('6h')) {
                // Заменяем только в контексте таймфрейма, не везде
                el.textContent = el.textContent.replace(/(\d+)\s*(свечей|свечи|свеча)\s*=\s*(\d+)\s*(часов|дней|дня|день)\s*на\s*6[hH]/g, 
                    (match, candles, candlesWord, hours, hoursWord) => {
                        // Пересчитываем для нового таймфрейма
                        const timeframeHours = {
                            '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60, '30m': 30/60,
                            '1h': 1, '2h': 2, '4h': 4, '6h': 6, '8h': 8, '12h': 12, '1d': 24
                        };
                        const hoursPerCandle = timeframeHours[timeframe] || 6;
                        const totalHours = parseInt(candles) * hoursPerCandle;
                        const days = Math.floor(totalHours / 24);
                        
                        if (days > 0) {
                            return `${candles} ${candlesWord} = ${days} ${days === 1 ? 'день' : days < 5 ? 'дня' : 'дней'} на ${timeframe.toUpperCase()}`;
                        } else {
                            return `${candles} ${candlesWord} = ${totalHours} ${totalHours === 1 ? 'час' : totalHours < 5 ? 'часа' : 'часов'} на ${timeframe.toUpperCase()}`;
                        }
                    });
                
                // Обновляем упоминания таймфрейма в тексте
                el.textContent = el.textContent.replace(/на\s+6[hH]\s+таймфрейме/g, `на ${timeframe.toUpperCase()} таймфрейме`);
                el.textContent = el.textContent.replace(/\(6H\)/g, `(${timeframe.toUpperCase()})`);
            }
        });
        
        // Обновляем метки в таблицах и списках
        document.querySelectorAll('.label, .label-text').forEach(el => {
            if (el.textContent.includes('6H') || el.textContent.includes('6h')) {
                el.textContent = el.textContent.replace(/6[hH]/g, timeframe.toUpperCase());
            }
        });
        
        console.log('[BotsManager] ✅ Интерфейс обновлен для таймфрейма:', timeframe);
    }
    
    /**
     * Получает все текстовые узлы из элемента (рекурсивно)
     */
    getTextNodes(element) {
        const textNodes = [];
        const walker = document.createTreeWalker(
            element,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        let node;
        while (node = walker.nextNode()) {
            textNodes.push(node);
        }
        
        return textNodes;
    }
    
    /**
     * Инициализирует обработчики для управления таймфреймом
     */
    initTimeframeControls() {
        const applyBtn = document.getElementById('applyTimeframeBtn');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => {
                this.applyTimeframe();
            });
            console.log('[BotsManager] ✅ Обработчик кнопки применения таймфрейма установлен');
        }
        
        // Загружаем текущий таймфрейм при инициализации
        this.loadTimeframe().then(timeframe => {
            // currentTimeframe уже установлен в loadTimeframe()
            if (applyBtn) {
                applyBtn.dataset.currentTimeframe = timeframe;
            }
            this.updateTimeframeInUI(timeframe);
        });
    }
}

// Экспортируем класс глобально сразу после определения
window.BotsManager = BotsManager;

// Глобальная функция для включения бота для текущей монеты (используется в HTML onclick)
window.enableBotForCurrentCoin = function(direction) {
    if (window.botsManager && window.botsManager.selectedCoin) {
        window.botsManager.createBot(direction || null);
    } else {
        console.error('[enableBotForCurrentCoin] BotsManager не инициализирован или монета не выбрана');
        if (window.showToast) {
            window.showToast('Выберите монету для создания бота', 'warning');
        }
    }
};

// BotsManager инициализируется в app.js, не здесь
// Version: 2025-10-21 03:47:29