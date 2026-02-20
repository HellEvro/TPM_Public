/**
 * BotsManager - 01_interface
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
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
    },
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
    },
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
    },
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
    });
})();
