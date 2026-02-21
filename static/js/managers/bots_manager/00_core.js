/**
 * BotsManager - ядро (constructor, init, логирование)
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
        
        // Единый интервал обновления UI и мониторинга ботов (2 сек — более частое обновление для реального времени)
        this.refreshInterval = 2000;
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
        
        // URL для API: используем тот же origin (app.py проксирует на bots.py), чтобы избежать
        // CORS и блокировок файрвола при прямом обращении к порту 5001 из браузера.
        this.BOTS_SERVICE_URL = window.location.origin || 'http://127.0.0.1:5000';
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
            
            // Сначала загружаем конфиг (в т.ч. position_sync_interval для интервала обновления списка монет слева)
            await this.loadConfigurationData();
            // Запускаем периодическое обновление с интервалом из конфига
            this.startPeriodicUpdate();
            // Повторная загрузка конфига через 2 сек (для актуализации после инициализации сервиса)
            setTimeout(() => this.loadConfigurationData(), 2000);
            
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
}
