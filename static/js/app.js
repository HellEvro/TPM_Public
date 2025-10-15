class App {
    constructor() {
        Logger.info('APP', 'Constructor called');
        try {
            Logger.info('APP', 'Starting initialization...');
            this.isUpdating = false;
            this.menuInitialized = false;
            this.currentTab = null;
            
            this._logState = () => {
                Logger.debug('APP', 'Current state:', {
                    currentTab: this.currentTab,
                    menuInitialized: this.menuInitialized,
                    isUpdating: this.isUpdating
                });
            };
            
            Logger.info('APP', 'Initializing managers');
            console.log('[APP] Creating StatisticsManager...');
            this.statisticsManager = new StatisticsManager();
            console.log('[APP] ✅ StatisticsManager created');
            
            console.log('[APP] Creating PositionsManager...');
            this.positionsManager = new PositionsManager();
            console.log('[APP] ✅ PositionsManager created');
            
            console.log('[APP] Creating ExchangeManager...');
            this.exchangeManager = new ExchangeManager();
            console.log('[APP] ✅ ExchangeManager created');
            
            console.log('[APP] Creating PositionCloser...');
            this.positionCloser = new PositionCloser(this.exchangeManager);
            console.log('[APP] ✅ PositionCloser created');
            
            console.log('[APP] Creating BotsManager...');
            try {
                this.botsManager = new BotsManager();
                window.botsManager = this.botsManager; // Делаем доступным глобально
                console.log('[APP] ✅ BotsManager created');
            } catch (error) {
                console.error('[APP] ❌ BotsManager не определен! Проверьте порядок загрузки скриптов.', error);
                console.log('[APP] Доступные глобальные объекты:', Object.keys(window).filter(key => key.includes('Manager') || key.includes('Manager')));
                console.log('[APP] Все загруженные скрипты:', Array.from(document.scripts).map(s => s.src));
                
                // Создаем заглушку BotsManager
                this.botsManager = this.createBotsManagerStub();
                console.log('[APP] ⚠️ BotsManager создан как заглушка');
            }
            
            // Принудительная загрузка конфигурации для страницы ботов
            if (window.location.pathname === '/bots') {
                console.log('[APP] 🔧 Обнаружена страница ботов, загружаем конфигурацию...');
                setTimeout(() => {
                    console.log('[APP] 📋 Принудительная загрузка конфигурации...');
                    this.botsManager.loadConfigurationData();
                }, 1500);
            }
            
            this.currentPage = 1;
            this.allClosedPnlData = [];

            

            
            // Ждем загрузки DOM для остальной инициализации
            if (document.readyState === 'loading') {
                console.log('[APP] DOM still loading, adding DOMContentLoaded listener');
                document.addEventListener('DOMContentLoaded', () => {
                    console.log('[APP] DOMContentLoaded fired, calling initDOM()');
                    this.initDOM();
                });
            } else {
                console.log('[APP] DOM already loaded, calling initDOM() directly');
                this.initDOM();
            }
            
            this.initializeGlobalSearch();
            
        } catch (e) {
            Logger.error('APP', 'Error in constructor:', e);
            NotificationManager.error('Error initializing application');
        }
    }

    initDOM() {
        console.log('[APP] InitDOM started');
        try {
            // Сначала восстанавливаем последнюю активную страницу и название
            console.log('[APP] Calling initializeLastActivePage');
            this.initializeLastActivePage();
            
            // Затем инициализируем компоненты и запускаем обновления
            console.log('[APP] Initializing components');
            this.initializeApp();

            this.initializeControls();
            
            console.log('[APP] InitDOM completed');
        } catch (e) {
            console.error('[APP] Error in initDOM:', e);
        }
    }

    initializeLastActivePage() {
        console.log('[MENU] initializeLastActivePage started');
        try {
            const lastActivePage = localStorage.getItem('lastActivePage') || 'positions';
            const savedMenuText = localStorage.getItem('lastActivePageText');
            const menuTitle = document.querySelector('.menu-title');
            console.log('[MENU] Menu title element:', menuTitle);
            console.log('[MENU] Restoring page:', lastActivePage, 'with text:', savedMenuText);
            
            if (menuTitle && !this.menuInitialized) {
                // Устанавливаем атрибут data-translate для правильного перевода
                menuTitle.setAttribute('data-translate', lastActivePage);
                
                // Используем сохраненный текст или получаем перевод
                let translatedText = savedMenuText;
                if (!translatedText && window.languageUtils) {
                    translatedText = languageUtils.translate(lastActivePage);
                }
                
                // Fallback на базовые названия если перевод не доступен
                if (!translatedText) {
                    const fallbackNames = {
                        'positions': 'Позиции',
                        'bots': 'Боты', 
                        'closedPnl': 'Закрытые PNL'
                    };
                    translatedText = fallbackNames[lastActivePage] || lastActivePage;
                }
                
                console.log('[MENU] Setting menu title to:', translatedText);
                menuTitle.textContent = translatedText;
                this.menuInitialized = true;
            }
            
            // Показываем нужную вкладку
            this.showTab(lastActivePage, false);
            
        } catch (e) {
            console.error('[MENU] Error in initializeLastActivePage:', e);
        }
    }

    showTab(tabName, saveState = true) {
        console.log('[MENU] showTab called with:', { tabName, saveState });
        try {
            if (this.currentTab === tabName) {
                console.log('[MENU] Already on this tab:', tabName);
                this._logState();
                return;
            }
            
            const menuTitle = document.querySelector('.menu-title');
            
            if (saveState && menuTitle) {
                // Безопасное получение перевода с fallback
                let translatedText;
                if (window.languageUtils && typeof languageUtils.translate === 'function') {
                    translatedText = languageUtils.translate(tabName);
                } else {
                    // Fallback на базовые названия
                    const fallbackNames = {
                        'positions': 'Позиции',
                        'bots': 'Боты', 
                        'closedPnl': 'Закрытые PNL'
                    };
                    translatedText = fallbackNames[tabName] || tabName;
                }
                
                console.log('[MENU] Saving to localStorage:', {
                    lastActivePage: tabName,
                    lastActivePageText: translatedText
                });
                
                localStorage.setItem('lastActivePage', tabName);
                localStorage.setItem('lastActivePageText', translatedText);
                
                requestAnimationFrame(() => {
                    console.log('[MENU] Force updating menu title to:', translatedText);
                    menuTitle.textContent = translatedText;
                    this._logState();
                });
            }
            
            // Обновляем текущую вкладку
            this.currentTab = tabName;
            console.log('[MENU] Tab changed to:', tabName);
            this._logState();
            
            // Обновляем видимость контейнеров
            const positionsContainer = document.querySelector('.positions-container');
            const statsContainer = document.querySelector('.stats-container');
            const closedPnlContainer = document.getElementById('closedPnlContainer');

            
            // Скрываем все контейнеры сначала
            if (positionsContainer) positionsContainer.style.display = 'none';
            if (statsContainer) statsContainer.style.display = 'none';
            if (closedPnlContainer) closedPnlContainer.style.display = 'none';

            
            const botsContainer = document.getElementById('botsContainer');
            if (botsContainer) botsContainer.style.display = 'none';
            
            // Показываем нужные контейнеры
            if (tabName === 'positions') {
                if (positionsContainer) positionsContainer.style.display = 'block';
                if (statsContainer) statsContainer.style.display = 'block';
                document.querySelector('.main-container').style.display = 'flex';
                
                // Загружаем данные позиций при переключении на вкладку
                setTimeout(() => {
                    if (this.positionsManager) {
                        this.positionsManager.updateData();
                    }
                }, 100);


            } else if (tabName === 'bots') {
                console.log('[BOTS] Showing bots tab');
                if (botsContainer) {
                    console.log('[BOTS] Setting botsContainer display to block');
                    botsContainer.style.display = 'block';
                } else {
                    console.error('[BOTS] botsContainer not found');
                }
                document.querySelector('.main-container').style.display = 'none';
                
                // Инициализируем менеджер ботов
                if (this.botsManager) {
                    console.log('[BOTS] Initializing bots manager');
                    this.botsManager.init();
                } else {
                    console.error('[BOTS] Bots manager not initialized yet');
                }
                
                // Проверяем, открыт ли таб истории внутри страницы ботов
                setTimeout(() => {
                    const historyTab = document.getElementById('historyTab');
                    if (historyTab && historyTab.style.display !== 'none') {
                        console.log('[BOTS] History tab is visible, initializing...');
                        if (this.botsManager && typeof this.botsManager.initializeHistoryTab === 'function') {
                            this.botsManager.initializeHistoryTab();
                        }
                    }
                }, 100);
            } else if (tabName === 'closedPnl') {
                if (closedPnlContainer) closedPnlContainer.style.display = 'block';
                document.querySelector('.main-container').style.display = 'none';
                // Загружаем данные для закрытых позиций
                this.updateClosedPnl(true);
            }
            
            // Обновляем активный пункт меню
            document.querySelectorAll('.menu-item').forEach(item => {
                item.classList.remove('active');
            });
            document.getElementById(`${tabName}MenuItem`).classList.add('active');
            
            // Закрываем меню
            document.getElementById('menuDropdown').classList.remove('active');
            
        } catch (error) {
            console.error('[MENU] Error in showTab:', error);
        }
    }

    initializeControls() {
        try {
            // Инициализация чекбокса "Снизить нагрузку"
            const reduceLoadCheckbox = document.getElementById('reduceLoadCheckbox');
            if (reduceLoadCheckbox) {
                const savedState = localStorage.getItem('reduceLoad') === 'true';
                reduceLoadCheckbox.checked = savedState;
                reduceLoadCheckbox.addEventListener('change', (e) => {
                    localStorage.setItem('reduceLoad', e.target.checked);
                    this.positionsManager.setReduceLoad(e.target.checked);
                });
                this.positionsManager.setReduceLoad(savedState);
            }

            // Инициализация кнопки смены темы
            const themeButton = document.querySelector('.control-item[onclick="toggleTheme()"]');
            if (themeButton) {
                themeButton.onclick = () => {
                    const body = document.body;
                    const currentTheme = body.getAttribute('data-theme');
                    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                    
                    if (newTheme === 'dark') {
                        body.removeAttribute('data-theme');
                    } else {
                        body.setAttribute('data-theme', 'light');
                    }
                    
                    localStorage.setItem('theme', newTheme);
                    
                    // Обновляем графики при смене темы
                    if (this.statisticsManager) {
                        this.statisticsManager.initializeChart();
                    }
                };
            }
        } catch (e) {
            console.error('Error in initializeControls:', e);
        }
    }

    initializeApp() {
        try {
            console.log('Starting app initialization...');
            // Инициализация темы
            this.initializeTheme();
            
            // Инициализация языка
            updateInterface();
            
            // Запуск обновления данных
            this.startDataUpdates();
            initializeSortSelects();
            
            // Принудительно загружаем данные позиций при старте
            setTimeout(() => {
                console.log('[APP] Initial positions data load');
                if (this.positionsManager) {
                    this.positionsManager.updateData();
                }
            }, 1000);
        } catch (e) {
            console.error('Error in initializeApp:', e);
        }
    }

    initializeTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.setAttribute('data-theme', 'light');
        } else {
            document.body.removeAttribute('data-theme');
        }
    }

    startDataUpdates() {
        console.log('Starting data updates...');
        let isUpdating = false;

        const update = async () => {
            if (isUpdating) {
                console.log('Update already in progress, skipping...');
                return;
            }

            try {
                isUpdating = true;
                await this.updateData();
            } catch (error) {
                console.error('Error in update cycle:', error);
            } finally {
                isUpdating = false;
            }
        };

        // Первоначальная загрузка
        update().then(() => {
            // Запускаем регулярное обновление
            setInterval(update, UPDATE_INTERVAL);
            console.log(`Data updates started with interval ${UPDATE_INTERVAL}ms`);
        });
    }

    async updateData() {
        try {
            // console.log('[MENU] updateData called');
            if (this.isUpdating) {
                // console.log('[UPDATE] Update already in progress, skipping...');
                this._logState();
                return;
            }

            this.isUpdating = true;
            // console.log('[UPDATE] Starting data update...');
            this._logState();
            
            if (this.currentTab === 'positions') {
                const data = await this.positionsManager.updateData();
                this.updateLastUpdateTime();
                return data;
            } else {
                // console.log('[UPDATE] Skipping data update - not on positions tab');
                this._logState();
                return null;
            }
        } catch (error) {
            console.error('[UPDATE] Error updating data:', error);
            this.showErrorNotification('Ошибка обновления данных');
            return null;
        } finally {
            this.isUpdating = false;
            this._logState();
        }
    }

    updateLastUpdateTime() {
        const updateTimeElement = document.getElementById('update-time');
        if (updateTimeElement) {
            const now = new Date();
            updateTimeElement.textContent = now.toLocaleTimeString();
        }
    }

    async updateClosedPnl(resetPage = false) {
        try {
            if (resetPage) {
                this.currentPage = 1;
            }

            const sortSelect = document.getElementById('sortSelect');
            const sortBy = sortSelect ? sortSelect.value : 'time';

            const response = await fetch(`/api/closed_pnl?sort=${sortBy}`);
            const data = await response.json();

            if (data.success) {
                // Обновляем данные о балансе и PNL
                if (data.wallet_data) {
                    document.getElementById('totalBalance').textContent = 
                        `${data.wallet_data.total_balance.toFixed(2)} USDT`;
                    document.getElementById('availableBalance').textContent = 
                        `${data.wallet_data.available_balance.toFixed(2)} USDT`;
                    document.getElementById('realizedPnL').textContent = 
                        `${data.wallet_data.realized_pnl.toFixed(2)} USDT`;
                }

                // Обновляем таблицу закрытых позиций
                this.allClosedPnlData = data.closed_pnl;
                this.updateClosedPnlTable(this.allClosedPnlData, false);
            } else {
                console.error('Failed to get closed PNL data:', data.error);
                this.showErrorNotification('Ошибка при получении данных о закрытых позициях');
            }
        } catch (error) {
            console.error('Error updating closed PNL:', error);
            this.showErrorNotification('Ошибка при обновлении данных');
        }
    }

    updateClosedPnlTable(data = null, resetPage = false) {
        // Используем переданные данные или существующие
        const displayData = data || this.allClosedPnlData;
        if (!displayData || displayData.length === 0) {
            const tableBody = document.getElementById('closedPnlTable');
            if (tableBody) {
                tableBody.innerHTML = '<tr><td colspan="6" class="no-data">Нет данных</td></tr>';
            }
            return;
        }
        
        // Получаем текущий поисковый запрос
        const searchQuery = document.getElementById('tickerSearch')?.value.toUpperCase() || '';
        
        // Фильтруем данные по поисковому запросу
        const filteredData = searchQuery ? 
            displayData.filter(pnl => pnl.symbol.includes(searchQuery)) : 
            displayData;
        
        // Получаем текущую страницу и размер страницы
        const pageSize = parseInt(localStorage.getItem('pageSize') || '10');
        const currentPage = resetPage ? 1 : (this.currentPage || 1);
        this.currentPage = currentPage;
        
        // Вычисляем диапазон для текущей страницы
        const start = (currentPage - 1) * pageSize;
        const end = start + pageSize;
        const pageData = filteredData.slice(start, end);
        
        // Генерируем HTML таблицы
        const tableHtml = pageData.map(pnl => {
            const isProfit = parseFloat(pnl.closed_pnl) >= 0;
            const pnlValue = parseFloat(pnl.closed_pnl);
            return `
                <tr>
                    <td class="ticker-cell">
                        <span class="ticker">${pnl.symbol}</span>
                        <a href="${createTickerLink(pnl.symbol, pnl.exchange)}" 
                           target="_blank" 
                           class="external-link"
                           title="Открыть на бирже">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                                <polyline points="15 3 21 3 21 9"></polyline>
                                <line x1="10" y1="14" x2="21" y2="3"></line>
                            </svg>
                        </a>
                    </td>
                    <td>${pnl.qty}</td>
                    <td>${parseFloat(pnl.entry_price).toFixed(5)}</td>
                    <td>${parseFloat(pnl.exit_price).toFixed(5)}</td>
                    <td class="${isProfit ? 'positive-pnl' : 'negative-pnl'}">
                        ${isProfit ? '+' : ''}${pnlValue.toFixed(2)} USDT
                    </td>
                    <td>${pnl.close_time}</td>
                </tr>
            `;
        }).join('');
        
        // Обновляем таблицу
        const tableBody = document.getElementById('closedPnlTable');
        if (tableBody) {
            tableBody.innerHTML = tableHtml || '<tr><td colspan="6" class="no-data">Нет данных</td></tr>';
        }
        
        // Обновляем элементы пагинации
        this.updatePaginationControls(filteredData.length, pageSize, currentPage);
    }

    updatePaginationControls(totalItems, pageSize, currentPage) {
        const totalPages = Math.ceil(totalItems / pageSize);
        
        // Обновляем информацию о странице
        const pageInfo = domUtils.getElement('pageInfo');
        if (pageInfo) {
            pageInfo.textContent = `Страница ${currentPage} из ${totalPages}`;
        }
        
        // Обновляем состояние кнопок
        const prevButton = document.querySelector('.pagination-btn:first-child');
        const nextButton = document.querySelector('.pagination-btn:last-child');
        
        if (prevButton) {
            prevButton.disabled = currentPage === 1;
        }
        if (nextButton) {
            nextButton.disabled = currentPage === totalPages;
        }
    }

    prevPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.updateClosedPnlTable(this.allClosedPnlData, false);
        }
    }

    nextPage() {
        const pageSize = parseInt(storageUtils.get('pageSize', DEFAULTS.PAGE_SIZE));
        const totalPages = Math.ceil((this.allClosedPnlData?.length || 0) / pageSize);
        
        if (this.currentPage < totalPages) {
            this.currentPage++;
            this.updateClosedPnlTable(this.allClosedPnlData, false);
        }
    }

    changePageSize(newSize) {
        storageUtils.set('pageSize', parseInt(newSize));
        this.currentPage = 1;
        this.updateClosedPnlTable(this.allClosedPnlData, false);
    }

    showErrorNotification(message) {
        NotificationManager.error(message);
    }

    showSuccessNotification(message) {
        NotificationManager.success(message);
    }

    toggleTheme() {
        const body = document.body;
        const currentTheme = body.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        if (newTheme === 'dark') {
            body.removeAttribute('data-theme');
        } else {
            body.setAttribute('data-theme', 'light');
        }
        
        localStorage.setItem('theme', newTheme);
        
        // Обновляем графики при смене темы
        if (this.statisticsManager) {
            this.statisticsManager.initializeChart();
        }
    }







    initializeGlobalSearch() {
        const searchInput = document.getElementById('tickerSearch');
        const clearButton = document.getElementById('clearSearch');
        
        if (searchInput) {
            // Очищаем поле при инициализации
            searchInput.value = '';
            if (clearButton) {
                clearButton.style.display = 'none';
            }

            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.toUpperCase();
                
                // Показываем/скрываем кнопку очистки
                if (clearButton) {
                    clearButton.style.display = query ? 'block' : 'none';
                }
                
                // Фильтрация в зависимости от екущей вкладки
                if (this.currentTab === 'positions') {
                    // Фильтруем позиции
                    document.querySelectorAll('.position').forEach(position => {
                        const symbol = position.getAttribute('data-symbol');
                        if (symbol) {
                            position.style.display = symbol.includes(query) ? '' : 'none';
                        }
                    });
                    
                    // Обновляем счетчики позиций
                    this.updatePositionCounts();
                    

                    
                } else if (this.currentTab === 'closedPnl') {
                    // Обновляем таблицу закрытых позиций с сохраненными данными
                    this.updateClosedPnlTable(this.allClosedPnlData, true);
                }
            });
            
            // Обработчик для кнопки очистки
            if (clearButton) {
                clearButton.addEventListener('click', () => {
                    searchInput.value = '';
                    clearButton.style.display = 'none';
                    
                    // Вызываем событие input для обновления фильтрации
                    const inputEvent = new Event('input');
                    searchInput.dispatchEvent(inputEvent);
                });
            }
        }
    }

    // Вспомогательный метод для обновления счетчиков позиций
    updatePositionCounts() {
        const containers = ['high-profitable', 'profitable', 'losing'];
        
        containers.forEach(type => {
            const container = document.getElementById(`${type}-positions`);
            const countElement = document.querySelector(`#${type}-positions-header .position-count`);
            
            if (container && countElement) {
                const visiblePositions = container.querySelectorAll('.position[style=""]').length;
                countElement.textContent = `(${visiblePositions})`;
            }
        });
    }


    

    

    // Создание заглушки BotsManager при ошибке загрузки
    createBotsManagerStub() {
        return {
            serviceOnline: false,
            updateData: () => {
                console.log('[BOTS] BotsManager not available, checking service status...');
                this.checkBotsServiceDirectly();
            },
            init: () => {
                console.log('[BOTS] BotsManager not available, showing service error...');
                this.showBotsServiceError();
            },
            checkBotsServiceDirectly: () => {
                console.log('[BOTS] Direct service check...');
                return fetch('http://127.0.0.1:5001/api/status')
                    .then(response => response.json())
                    .then(data => {
                        console.log('[BOTS] Service is online:', data);
                        if (data.status === 'online') {
                            this.hideBotsServiceError();
                            return true;
                        }
                        return false;
                    })
                    .catch(error => {
                        console.error('[BOTS] Service check failed:', error);
                        return false;
                    });
            },
            showBotsServiceError: () => {
                const botsContainer = document.getElementById('botsContainer');
                if (botsContainer) {
                    botsContainer.innerHTML = `
                        <div class="service-error">
                            <h3>🔧 Сервис ботов недоступен</h3>
                            <p>Для работы с ботами необходимо запустить сервис:</p>
                            <code>python bots.py</code>
                            <br><br>
                            <button onclick="window.app.checkBotsServiceDirectly()" class="btn btn-primary">
                                Проверить сервис
                            </button>
                        </div>
                    `;
                }
            },
            hideBotsServiceError: () => {
                const botsContainer = document.getElementById('botsContainer');
                if (botsContainer && botsContainer.querySelector('.service-error')) {
                    botsContainer.innerHTML = '';
                }
            }
        };
    }

    // Методы для работы с сервисом ботов
    async checkBotsServiceDirectly() {
        console.log('[BOTS] Direct service check...');
        try {
            const response = await fetch('http://127.0.0.1:5001/api/status');
            const data = await response.json();
            console.log('[BOTS] Service is online:', data);
            if (data.status === 'online') {
                this.hideBotsServiceError();
                return true;
            }
            return false;
        } catch (error) {
            console.error('[BOTS] Service check failed:', error);
            return false;
        }
    }

    showBotsServiceError() {
        const botsContainer = document.getElementById('botsContainer');
        if (botsContainer) {
            botsContainer.innerHTML = `
                <div class="service-error">
                    <h3>🔧 Сервис ботов недоступен</h3>
                    <p>Для работы с ботами необходимо запустить сервис:</p>
                    <code>python bots.py</code>
                    <br><br>
                    <button onclick="window.app.checkBotsServiceDirectly()" class="btn btn-primary">
                        Проверить сервис
                    </button>
                </div>
            `;
        }
    }

    hideBotsServiceError() {
        const botsContainer = document.getElementById('botsContainer');
        if (botsContainer && botsContainer.querySelector('.service-error')) {
            botsContainer.innerHTML = '';
        }
    }
}

// Глобальная функция для обновления позиций
function updatePositions() {
    if (window.app && window.app.positionsManager) {
        window.app.positionsManager.updateData();
    }
}

// В начале файла добавим функции для работы с localStorage
function saveFilterState(containerId, value) {
    localStorage.setItem(`sort_${containerId}`, value);
}

function loadFilterState(containerId) {
    return localStorage.getItem(`sort_${containerId}`) || 'pnl_desc'; // значение по умолчанию
}

// Обновляем функцию инициализации сортировки
function initializeSortSelects() {
    const sortSelects = {
        'sort-high-profitable-positions': '#high-profitable-positions',
        'sort-profitable-positions': '#profitable-positions',
        'sort-losing-positions': '#losing-positions'
    };

    Object.entries(sortSelects).forEach(([selectId, containerId]) => {
        const select = document.getElementById(selectId);
        if (select) {
            // Загруаем сохраненное значение
            const savedValue = loadFilterState(containerId);
            select.value = savedValue;

            select.addEventListener('change', function() {
                // Сохраняем новое значение
                saveFilterState(containerId, this.value);
                updatePositions();
            });
        }
    });

    // Инициализация сортировки для закрытых позиций
    const closedPnlSort = document.getElementById('sortSelect');
    if (closedPnlSort) {
        const savedValue = loadFilterState('closedPnl');
        closedPnlSort.value = savedValue;
        
        closedPnlSort.addEventListener('change', function() {
            saveFilterState('closedPnl', this.value);
            updateClosedPnl();
        });
    }
}

// Инициализация приожения при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    console.log('[INIT] DOMContentLoaded event fired');
    console.log('[INIT] Creating App instance');
    try {
        window.app = new App();
        console.log('[INIT] ✅ App instance created successfully');
    } catch (error) {
        console.error('[INIT] ❌ Error creating App instance:', error);
        console.error('[INIT] Stack trace:', error.stack);
    }
}); 

class ClosedPnlManager {
    constructor() {
        this.data = [];
        this.currentPage = 1;
        this.pageSize = parseInt(storageUtils.get('pageSize', DEFAULTS.PAGE_SIZE));
    }

    async loadData(sortBy) {
        try {
            const data = await apiUtils.fetchData(API_ENDPOINTS.GET_CLOSED_PNL, { sort: sortBy });
            if (data?.closed_pnl) {
                this.data = data.closed_pnl;
                return true;
            }
            return false;
        } catch (error) {
            console.error("Error loading closed PNL data:", error);
            return false;
        }
    }

    getCurrentPageData() {
        const start = (this.currentPage - 1) * this.pageSize;
        const end = start + this.pageSize;
        return this.data.slice(start, end);
    }
} 

// Добавляем в начало файла app.js
function toggleMenu() {
    const dropdown = document.getElementById('menuDropdown');
    dropdown.classList.toggle('active');
}

// Добавляем обработчик клика вне меню для его закрытия
document.addEventListener('click', (e) => {
    const menu = document.querySelector('.burger-menu');
    const dropdown = document.getElementById('menuDropdown');
    if (!menu.contains(e.target) && dropdown.classList.contains('active')) {
        dropdown.classList.remove('active');
    }
}); 