/**
 * BotsManager - 08_filter_ui
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
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
    },
            renderFilters() {
        this.renderWhitelist();
        this.renderBlacklist();
    },
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
    },
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
    },
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
    },
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
    },
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
    },
            async removeFromWhitelist(symbol) {
        try {
            const whitelist = (this.filtersData?.whitelist || []).filter(s => s !== symbol);
            await this.updateFilters({ whitelist });
            this.showNotification(`✅ ${symbol} удалена из белого списка`, 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка удаления из белого списка:', error);
            this.showNotification('❌ Ошибка удаления из белого списка', 'error');
        }
    },
            async removeFromBlacklist(symbol) {
        try {
            const blacklist = (this.filtersData?.blacklist || []).filter(s => s !== symbol);
            await this.updateFilters({ blacklist });
            this.showNotification(`✅ ${symbol} удалена из черного списка`, 'success');
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка удаления из черного списка:', error);
            this.showNotification('❌ Ошибка удаления из черного списка', 'error');
        }
    },
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
    },
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
    },
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
    },
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
    },
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
    },
            validateCoinSymbol(symbol) {
        // Проверяем что монета есть в списке доступных пар
        return this.coinsRsiData && this.coinsRsiData.some(coin => coin.symbol === symbol);
    },
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
    },
            showNotification(message, type = 'info') {
        // Простое уведомление в консоли, можно заменить на toast
        console.log(`[${type.toUpperCase()}] ${message}`);
    }

    // ==================== ИСТОРИЯ БОТОВ ====================

    /**
     * Инициализирует вкладку истории ботов
     */,
            showFilterControls(symbol) {
        const filterSection = document.getElementById('filterControlsSection');
        if (filterSection && symbol) {
            filterSection.style.display = 'block';
        }
    },
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
    },
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
    },
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
    },
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
    },
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
    },
            getFoundCoins(searchTerm) {
        if (!this.coinsRsiData || !searchTerm) return [];

        const term = searchTerm.toLowerCase();
        return this.coinsRsiData.filter(coin => 
            coin.symbol.toLowerCase().includes(term) ||
            coin.symbol.toLowerCase().startsWith(term)
        );
    },
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
    },
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
    },
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
    },
            searchCoins(searchTerm) {
        if (!this.coinsRsiData || !searchTerm) return [];

        const term = searchTerm.toLowerCase();
        return this.coinsRsiData.filter(coin => 
            coin.symbol.toLowerCase().includes(term)
        ).slice(0, 50); // Ограничиваем 50 результатами
    },
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
    },
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
    },
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
    },
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
    },
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
    },
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
    });
})();
