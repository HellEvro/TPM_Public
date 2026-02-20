class PositionsManager {
    constructor() {
        this.pnlThreshold = parseFloat(storageUtils.get('pnl_threshold', DEFAULTS.PNL_THRESHOLD));
        this.lastData = null;
        this.sortSettings = {
            'high-profitable-positions': storageUtils.get('sort_#high-profitable-positions', 'pnl_desc'),
            'profitable-positions': storageUtils.get('sort_#profitable-positions', 'pnl_desc'),
            'losing-positions': storageUtils.get('sort_#losing-positions', 'pnl_desc')
        };
        this.initializeFilters();
        this.initializeSorting();
        this.chartCache = new Map();  // Кэш для миниграфиков
        this.rsiCache = new Map();  // Кэш для значений RSI
        this.isUpdatingData = false;  // Защита от одновременного запуска (ERR_INSUFFICIENT_RESOURCES)
        this.lastUpdateAllDataStart = 0;  // Время последнего старта обновления (throttle)
        this.minUpdateAllDataInterval = 10000;  // Минимум 10 сек между запусками
        this.chartUpdateInterval = 5 * 60 * 1000;  // 5 минут (не используется)
        // ✅ КРИТИЧНО: Загружаем интервал обновления из конфига (в секундах, конвертируем в мс)
        this.updateInterval = 60 * 1000;  // Дефолт до загрузки конфига (10 мин); затем из «Синхронизация позиций»
        this.updateIntervalTimer = null;  // Таймер обновления
        this.currentTheme = document.body.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
        this.initializeThemeListener();
        this.initializeDataUpdater();  // Загрузит интервал из конфига и запустит обновление
        this.previousRoi = new Map();  // Добавляем хранилище для предыдущих значений ROI
        this.reduceLoad = storageUtils.get('reduceLoad', false);
        this.initializeLoadSettings();
        this.clearedSymbols = new Set();  // Добавляем множество для хранения очищенных символов
        this.initializeClearButton();
        this.searchQuery = '';
        this.initializeSearch();
    }

    initializeFilters() {
        const filterInput = domUtils.getElement(DOM_IDS.PNL_FILTER_INPUT);
        if (filterInput) {
            filterInput.value = this.pnlThreshold;
            this.updateBlockHeader('PROFITABLE', 0);
            filterInput.addEventListener('input', (e) => {
                const newValue = parseFloat(e.target.value) || DEFAULTS.PNL_THRESHOLD;
                this.pnlThreshold = newValue;
                storageUtils.set('pnl_threshold', newValue);
                this.updateHighProfitableLabel();
                if (this.lastData) {
                    this.updateData();
                }
            });
        }
    }

    initializeSorting() {
        for (const blockType of Object.keys(POSITION_BLOCKS)) {
            const blockConfig = POSITION_BLOCKS[blockType];
            const sortSelect = domUtils.getElement(blockConfig.sortId);
            
            if (sortSelect) {
                // Загружаем и устанавливаем сохраненное значение
                const savedValue = storageUtils.get(`sort_#${blockConfig.id}`, 'pnl_desc');
                sortSelect.value = savedValue;
                this.sortSettings[blockConfig.id] = savedValue;
                
                sortSelect.addEventListener('change', (e) => {
                    console.log(`Sorting ${blockConfig.id} by ${e.target.value}`);
                    this.sortSettings[blockConfig.id] = e.target.value;
                    storageUtils.set(`sort_#${blockConfig.id}`, e.target.value);
                    this.updatePositionsDisplay();
                });
            } else {
                console.warn(`Sort select not found for ${blockConfig.sortId}`);
            }
        }
    }

    initializeThemeListener() {
        // Следим за изменением темы
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.attributeName === 'data-theme') {
                    const newTheme = document.body.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
                    if (this.currentTheme !== newTheme) {
                        this.currentTheme = newTheme;
                        this.updateAllCharts();
                    }
                }
            });
        });

        observer.observe(document.body, {
            attributes: true,
            attributeFilter: ['data-theme']
        });
    }

    async loadChartUpdateInterval() {
        // Загружает интервал обновления миниграфиков из SystemConfig (bot_config.py)
        try {
            const response = await fetch('/api/bots/system-config');
            if (response.ok) {
                const data = await response.json();
                // Интервал RSI/миниграфиков = только из конфига «Синхронизация позиций»
                const intervalSec = data.config?.position_sync_interval ?? data.config?.mini_chart_update_interval;
                if (data.success && data.config && intervalSec !== undefined) {
                    const newInterval = (Number(intervalSec) || 600) * 1000;
                    if (newInterval !== this.updateInterval) {
                        this.updateInterval = newInterval;
                        console.log(`[PositionsManager] 📊 Интервал RSI/миниграфиков (＝ Синхронизация позиций): ${intervalSec} сек`);
                        
                        // Перезапускаем таймер с новым интервалом
                        if (this.updateIntervalTimer) {
                            clearInterval(this.updateIntervalTimer);
                        }
                        this.updateIntervalTimer = setInterval(() => this.updateAllData(), this.updateInterval);
                        console.log(`[PositionsManager] 🔄 Таймер обновления перезапущен с интервалом: ${this.updateInterval / 1000} сек`);
                    }
                } else {
                    console.warn('[PositionsManager] ⚠️ position_sync_interval не найден в SystemConfig, используем дефолт:', this.updateInterval / 1000, 'сек');
                }
            } else {
                console.warn('[PositionsManager] ⚠️ Не удалось загрузить SystemConfig, используем дефолт:', this.updateInterval / 1000, 'сек');
            }
        } catch (error) {
            console.error('[PositionsManager] ❌ Ошибка загрузки интервала обновления миниграфиков:', error);
            console.warn('[PositionsManager] ⚠️ Используем дефолт:', this.updateInterval / 1000, 'сек');
        }
    }

    initializeDataUpdater() {
        // Запускаем обновление с дефолтным интервалом сразу
        this.updateIntervalTimer = setInterval(() => this.updateAllData(), this.updateInterval);
        console.log(`[PositionsManager] 🔄 Обновление миниграфиков запущено с дефолтным интервалом: ${this.updateInterval / 1000} сек`);
        
        // Загружаем интервал из конфига асинхронно (обновит таймер если значение отличается)
        this.loadChartUpdateInterval();
    }

    initializeLoadSettings() {
        const checkbox = document.getElementById('reduceLoadCheckbox');
        if (checkbox) {
            checkbox.checked = this.reduceLoad;
            checkbox.addEventListener('change', (e) => {
                this.reduceLoad = e.target.checked;
                storageUtils.set('reduceLoad', this.reduceLoad);
                // Если включили снижение нагрузки - очищаем кэши
                if (this.reduceLoad) {
                    this.chartCache.clear();
                    this.rsiCache.clear();
                }
                if (this.lastData) {
                    this.updatePositionsDisplay();
                }
            });
        }
    }

    initializeClearButton() {
        const clearButton = document.getElementById('clear-rapid-growth');
        if (clearButton) {
            clearButton.addEventListener('click', () => {
                // Сохраняем текущие символы в множество очищенных
                const currentPositions = document.querySelectorAll('.rapid-growth-item .ticker');
                currentPositions.forEach(pos => {
                    const symbol = pos.textContent.trim();
                    this.clearedSymbols.add(symbol);
                });
                
                // Очищаем отображение
                const container = document.getElementById('rapid-growth-positions');
                if (container) {
                    container.innerHTML = '';
                }
                
                // Скрываем контейнер
                const mainContainer = document.getElementById('rapid-growth-container');
                if (mainContainer) {
                    mainContainer.style.display = 'none';
                }
            });
        }
    }

    initializeSearch() {
        const searchInput = document.getElementById('tickerSearch');
        const clearButton = document.getElementById('clearSearch');

        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.searchQuery = e.target.value.toUpperCase();
                this.updatePositionsDisplay();
            });
        }

        if (clearButton) {
            clearButton.addEventListener('click', () => {
                if (searchInput) {
                    searchInput.value = '';
                    this.searchQuery = '';
                    this.updatePositionsDisplay();
                }
            });
        }
    }


    async updateTickerData(symbol) {
        if (this.reduceLoad) return; // Не обновляем данные если включено снижение нагрузки
        try {
            console.log(`Updating data for ${symbol}...`);
            
            // Загружаем миниграфик
            const [chartResponse] = await Promise.allSettled([
                fetch(`/get_symbol_chart/${symbol}?theme=${this.currentTheme}`)
            ]);

            // Обрабатываем ответ графика
            if (chartResponse.status === 'fulfilled' && chartResponse.value.ok) {
                try {
                    const chartData = await chartResponse.value.json();
                    if (chartData.success && chartData.chart) {
                        const cacheKey = `${symbol}_${this.currentTheme}`;
                        this.chartCache.set(cacheKey, chartData.chart);
                        
                        // Сохраняем значение RSI если оно есть
                        if (chartData.current_rsi !== undefined && chartData.current_rsi !== null) {
                            this.rsiCache.set(symbol, chartData.current_rsi);
                        }
                        
                        document.querySelectorAll(`.mini-chart[data-symbol="${symbol}"]`)
                            .forEach(elem => {
                                if (elem) {
                                    elem.src = `data:image/png;base64,${chartData.chart}`;
                                }
                            });
                        
                        // Обновляем отображение значения RSI
                        document.querySelectorAll(`.rsi-value[data-symbol="${symbol}"]`)
                            .forEach(elem => {
                                if (elem && chartData.current_rsi !== undefined && chartData.current_rsi !== null) {
                                    // Ищем span с числом внутри (последний span)
                                    const valueSpan = elem.querySelector('span:last-child');
                                    if (valueSpan) {
                                        valueSpan.textContent = chartData.current_rsi.toFixed(2);
                                    } else {
                                        // Если структура не найдена, обновляем весь элемент
                                        elem.innerHTML = `<span style="font-size: 11px; font-weight: 400; opacity: 0.7;">RSI</span><span style="font-size: 11px; font-weight: 400;">${chartData.current_rsi.toFixed(2)}</span>`;
                                        elem.style.display = 'flex';
                                        elem.style.alignItems = 'center';
                                        elem.style.gap = '4px';
                                    }
                                }
                            });
                        console.log(`Chart updated for ${symbol}`);
                    } else {
                        console.warn(`Chart data not available for ${symbol}:`, chartData.error || 'Unknown error');
                    }
                } catch (e) {
                    console.error(`Error parsing chart JSON for ${symbol}:`, e);
                }
            } else {
                const errorMsg = chartResponse.status === 'rejected' 
                    ? chartResponse.reason?.message 
                    : `HTTP ${chartResponse.value?.status || 'unknown'}`;
                console.warn(`Failed to fetch chart for ${symbol}: ${errorMsg}`);
            }


            console.log(`Data update completed for ${symbol}`);
        } catch (error) {
            console.error(`Error updating data for ${symbol}:`, error);
        }
    }

    async updateAllData() {
        if (this.reduceLoad) return; // Не обновляем данные если включено снижение нагрузки
        if (!this.lastData) return;
        if (this.isUpdatingData) {
            console.log('[PositionsManager] updateAllData skipped: previous run still in progress');
            return;
        }
        const now = Date.now();
        if (now - this.lastUpdateAllDataStart < this.minUpdateAllDataInterval) {
            console.log('[PositionsManager] updateAllData skipped: throttle (min interval)', this.minUpdateAllDataInterval / 1000, 's');
            return;
        }
        this.lastUpdateAllDataStart = now;
        this.isUpdatingData = true;
        try {
            const symbols = new Set();
            // Собираем все уникальные символы с проверкой на undefined
            Object.values(this.lastData).forEach(positions => {
                if (Array.isArray(positions)) {
                    positions.forEach(pos => {
                        if (pos && pos.symbol && pos.symbol !== 'undefined') {
                            symbols.add(pos.symbol);
                        }
                    });
                }
            });

            console.log(`Starting data update for ${symbols.size} symbols:`, [...symbols]);

            // Обновляем данные для каждого символа последовательно (с паузой между запросами)
            for (const symbol of symbols) {
                await this.updateTickerData(symbol);
                // Небольшая пауза между запросами, чтобы не перегружать браузер/сервер
                if (symbols.size > 1) {
                    await new Promise(r => setTimeout(r, 150));
                }
            }

            console.log('All data updates completed');
        } finally {
            this.isUpdatingData = false;
        }
    }

    async updateAllCharts() {
        if (this.isUpdatingData) {
            console.log('[PositionsManager] updateAllCharts skipped: updateAllData in progress');
            return;
        }
        if (!this.lastData) return;
        const symbols = new Set();

        // Правильно получаем все позиции из разных категорий
        const allPositions = [
            ...(this.lastData.high_profitable || []),
            ...(this.lastData.profitable || []),
            ...(this.lastData.losing || [])
        ];

        // Собираем уникальные символы с проверкой на undefined
        allPositions.forEach(pos => {
            if (pos && pos.symbol && pos.symbol !== 'undefined') {
                symbols.add(pos.symbol);
            }
        });

        console.log(`Updating charts for ${symbols.size} symbols:`, [...symbols]);

        this.isUpdatingData = true;
        try {
            for (const symbol of symbols) {
                await this.updateTickerData(symbol);
                if (symbols.size > 1) {
                    await new Promise(r => setTimeout(r, 150));
                }
            }
        } finally {
            this.isUpdatingData = false;
        }
    }

    updatePositionsDisplay() {
        if (!this.lastData) {
            console.warn('No data to display');
            return;
        }

        for (const blockType of Object.keys(POSITION_BLOCKS)) {
            const blockConfig = POSITION_BLOCKS[blockType];
            const blockId = blockConfig.id;
            const sortValue = this.sortSettings[blockId];
            
            let positions = [];
            if (blockType === 'HIGH_PROFITABLE') {
                positions = this.lastData.high_profitable || [];
            } else if (blockType === 'PROFITABLE') {
                positions = this.lastData.profitable || [];
            } else {
                positions = this.lastData.losing || [];
            }

            const isLosing = blockType === 'LOSING';
            positions = this.sortPositions(positions, sortValue, isLosing);

            const html = this.generatePositionsHtml(positions, blockId);
            domUtils.setInnerHTML(blockId, html);
            
            this.updateBlockHeader(blockType, positions.length);
        }
    }

    async updateData() {
        try {
            console.log('PositionsManager: Fetching positions data...');
            const data = await apiUtils.fetchData(API_ENDPOINTS.GET_POSITIONS, {
                pnl_threshold: this.pnlThreshold
            });
            
            if (!data) {
                console.warn('PositionsManager: No positions data received');
                return;
            }

            console.log('PositionsManager: Received positions data:', data);
            console.log('Full position data:', {
                high_profitable: data.high_profitable?.[0],
                profitable: data.profitable?.[0],
                losing: data.losing?.[0]
            });
            
            // Проверяем структуру данных
            if (!data.high_profitable || !data.profitable || !data.losing) {
                console.error('PositionsManager: Invalid data structure:', data);
                return;
            }

            this.lastData = data;

            // Баланс, остаток, PnL (из wallet_data и stats)
            if (data.wallet_data) {
                const w = data.wallet_data;
                const balanceEl = document.getElementById('positions-balance');
                const availableEl = document.getElementById('positions-available');
                const pnlEl = document.getElementById('positions-account-pnl');
                if (balanceEl) balanceEl.textContent = '$' + (parseFloat(w.total_balance) || 0).toFixed(2);
                if (availableEl) availableEl.textContent = '$' + (parseFloat(w.available_balance) || 0).toFixed(2);
                const pnlVal = (data.stats && data.stats.total_pnl != null) ? data.stats.total_pnl : (parseFloat(w.realized_pnl) || 0);
                if (pnlEl) {
                    pnlEl.textContent = (pnlVal >= 0 ? '$' : '$-') + Math.abs(pnlVal).toFixed(2);
                    pnlEl.className = 'stats-value ' + (pnlVal >= 0 ? 'positive' : 'negative');
                }
            }
            
            // ВЖНО: Сначала обновяем статистику
            if (window.app?.statisticsManager) {
                console.log('PositionsManager: Updating statistics...');
                window.app.statisticsManager.updateStats(data.stats);
            }
            
            // Затем обнов отображение позиций
            this.updatePositionsDisplay();
            this.updateRapidGrowthPositions(data.rapid_growth);
            
            // Обновляем время последнего обновления
            const updateTimeElement = domUtils.getElement(DOM_IDS.UPDATE_TIME);
            if (updateTimeElement) {
                updateTimeElement.textContent = formatUtils.formatLastUpdate();
            }
            
            // В последнюю очередь обновляем данные (только если не включено снижение нагрузки)
            if (!this.reduceLoad) {
                if (this.chartCache.size === 0) {
                    console.log('PositionsManager: Initializing data for the first time');
                    setTimeout(() => this.updateAllData(), 0);
                } else {
                    // Если кэш уже есть, но нужно обновить данные, запускаем обновление
                    console.log('PositionsManager: Updating existing data');
                    setTimeout(() => this.updateAllData(), 0);
                }
            }
            
            return data;
        } catch (error) {
            console.error("PositionsManager: Error updating positions:", error);
        }
    }

    generatePositionsHtml(positions, blockType) {
        if (!positions || positions.length === 0) {
            return `<div class="no-positions">${languageUtils.translate('noPositions')}</div>`;
        }

        // Фильтруем позиции по поисковому запросу
        const filteredPositions = this.searchQuery ? 
            positions.filter(pos => pos.symbol.includes(this.searchQuery)) : 
            positions;

        if (filteredPositions.length === 0) {
            return `<div class="no-positions">${languageUtils.translate('noMatches')}</div>`;
        }

        const currentTheme = document.body.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
        
        return filteredPositions.map(pos => {
            // console.log('Generating HTML for position:', pos);
            const cacheKey = `${pos.symbol}_${currentTheme}`;
            
            // Определяем цвет ROI в зависимости от знака (выше/ниже нуля)
            // Зеленый если ROI > 0 (прибыль), красный если ROI < 0 (убыток)
            const roiDirection = pos.roi >= 0 ? 'roi-positive' : 'roi-negative';
            
            // Сохраняем текущее значение ROI для следующего сравнения (если понадобится в будущем)
            this.previousRoi.set(pos.symbol, pos.roi);

            // Логируем размер позиции
            const positionSize = pos.qty || pos.quantity || pos.size || 0;
            // console.log('Position size:', {
            //     symbol: pos.symbol,
            //     qty: pos.qty,
            //     quantity: pos.quantity,
            //     size: pos.size,
            //     finalSize: positionSize
            // });
            
            const isVirtual = !!pos.is_virtual;
            const virtualBadge = isVirtual ? `<span class="position-badge-virtual" style="background:#9c27b0;color:white;padding:2px 6px;border-radius:8px;font-size:10px;font-weight:600;margin-left:4px;">${languageUtils.translate('fullai_virtual_position') || 'Виртуальная'}</span>` : '';
            const positionFooter = `
                <div class="position-footer">
                    <span class="${pos.high_roi ? CSS_CLASSES.HIGH_ROI : ''} ${roiDirection}">
                        ROI: ${formatUtils.formatNumber(pos.roi)}%
                    </span>
                    <div class="position-actions">
                        <span class="position-side ${pos.side.toLowerCase()}">${pos.side}</span>
                        ${!isVirtual ? `<button class="close-positions-btn single-close" 
                                data-column="${blockType.replace('-positions', '')}"
                                data-symbol="${pos.symbol}"
                                data-side="${pos.side}"
                                data-size="${positionSize}">✕</button>` : ''}
                    </div>
                </div>
            `;

            const pnlDisplay = isVirtual
                ? `${formatUtils.formatNumber(pos.unrealized_pnl_percent != null ? pos.unrealized_pnl_percent : pos.roi)}%`
                : `${formatUtils.formatUsdt(pos.pnl)} USDT <span class="max-value">(Max: ${formatUtils.formatUsdt(pos.pnl > 0 ? pos.max_profit : pos.max_loss)})</span>`;

            return `
                <div class="position ${blockType.includes('profitable') ? 'profitable' : 'losing'} ${isVirtual ? 'position-virtual' : ''}" 
                     data-symbol="${pos.symbol}"
                     data-size="${positionSize}"
                     data-side="${pos.side}"
                     data-pnl="${isVirtual ? (pos.unrealized_pnl_percent != null ? pos.unrealized_pnl_percent : pos.roi) : pos.pnl}"
                     data-column="${blockType}"
                     data-is-virtual="${isVirtual}">
                    <div class="position-header">
                        <div class="ticker">
                            <a href="${createTickerLink(pos.symbol, window.app?.exchangeManager?.getSelectedExchange())}" 
                               target="_blank">${pos.symbol}</a>${virtualBadge}
                        </div>
                        ${!this.reduceLoad ? `
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <img class="mini-chart" 
                                 data-symbol="${pos.symbol}" 
                                 src="${this.chartCache.has(cacheKey) ? 
                                     `data:image/png;base64,${this.chartCache.get(cacheKey)}` : 
                                     'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'}"
                                 alt="Chart ${pos.symbol}"
                                 style="background-color: ${currentTheme === 'dark' ? '#2d2d2d' : '#ffffff'}"
                            />
                            <span class="rsi-value" 
                                  data-symbol="${pos.symbol}"
                                  style="display: flex; align-items: center; gap: 4px; font-weight: 400; color: ${currentTheme === 'dark' ? '#ffffff' : '#000000'}; font-family: 'Arial', sans-serif;">
                                <span style="font-size: 11px; font-weight: 400; opacity: 0.7;">RSI</span>
                                <span style="font-size: 11px; font-weight: 400;">${this.rsiCache.has(pos.symbol) ? this.rsiCache.get(pos.symbol).toFixed(2) : '-'}</span>
                            </span>
                        </div>
                        ` : ''}
                    </div>
                    <div class="${!isVirtual && pos.pnl > 1000 ? CSS_CLASSES.HIGH_PNL : ''}">
                        ${pnlDisplay}
                    </div>
                    ${positionFooter}
                </div>
            `;
        }).join('');
    }

    sortPositions(positions, sortBy, isLosing = false) {
        if (!positions || !Array.isArray(positions)) {
            console.warn('Invalid positions array:', positions);
            return [];
        }
        const sortPnl = (pos) => pos.is_virtual ? (pos.unrealized_pnl_percent != null ? pos.unrealized_pnl_percent : pos.roi || 0) : (pos.pnl || 0);

        return [...positions].sort((a, b) => {
            const aPnl = sortPnl(a);
            const bPnl = sortPnl(b);
            switch(sortBy) {
                case 'pnl_desc':
                    return isLosing ? aPnl - bPnl : bPnl - aPnl;
                case 'pnl_asc':
                    return isLosing ? bPnl - aPnl : aPnl - bPnl;
                case 'roi_desc':
                    return (b.roi ?? bPnl) - (a.roi ?? aPnl);
                case 'roi_asc':
                    return (a.roi ?? aPnl) - (b.roi ?? bPnl);
                case 'alphabet_asc':
                    return a.symbol.localeCompare(b.symbol);
                case 'alphabet_desc':
                    return b.symbol.localeCompare(a.symbol);
                default:
                    console.warn('Unknown sort type:', sortBy);
                    return 0;
            }
        });
    }

    updateRapidGrowthPositions(rapidGrowth) {
        if (!rapidGrowth || rapidGrowth.length === 0) {
            domUtils.setDisplay(DOM_IDS.RAPID_GROWTH_CONTAINER, 'none');
            return;
        }

        // Фильтруем позиции по поисковому запросу
        const filteredPositions = rapidGrowth.filter(pos => 
            !this.clearedSymbols.has(pos.symbol) && 
            (!this.searchQuery || pos.symbol.includes(this.searchQuery))
        );

        if (filteredPositions.length === 0) {
            domUtils.setDisplay(DOM_IDS.RAPID_GROWTH_CONTAINER, 'none');
            return;
        }

        const sortedPositions = filteredPositions.sort((a, b) => b.growth_ratio - a.growth_ratio);
        const html = sortedPositions.map(pos => `
            <div class="rapid-growth-item">
                <a href="${createTickerLink(pos.symbol, window.app?.exchangeManager?.getSelectedExchange())}" 
                   target="_blank" 
                   class="ticker">
                    ${pos.symbol}
                </a>
                <div class="growth-ratio">
                    x${formatUtils.formatNumber(pos.growth_ratio, 1)} (${formatUtils.formatUsdt(pos.current_pnl)} USDT)
                    <div style="font-size: 0.8em; color: #888;">
                        ${languageUtils.translate('from')} ${formatUtils.formatUsdt(pos.start_pnl)} ${languageUtils.translate('to')} ${formatUtils.formatUsdt(pos.current_pnl)} USDT
                    </div>
                </div>
            </div>
        `).join('');

        domUtils.setInnerHTML(DOM_IDS.RAPID_GROWTH_POSITIONS, html);
        domUtils.setDisplay(DOM_IDS.RAPID_GROWTH_CONTAINER, 'block');
    }

    updateBlockHeader(blockType, count) {
        const blockConfig = POSITION_BLOCKS[blockType];
        const headerElement = document.getElementById(`${blockConfig.id}-header`);
        
        if (headerElement) {
            const countText = this.searchQuery ? ` (найдено: ${count})` : ` (${count})`;
            const countElement = headerElement.querySelector('.position-count');
            
            if (countElement) {
                countElement.textContent = countText;
            }

            // Обновляем текст для profitable и losing заголовков
            if (blockType === 'PROFITABLE') {
                const titleContainer = headerElement.querySelector('.title-container');
                if (titleContainer) {
                    titleContainer.innerHTML = `PnL < ${this.pnlThreshold} USDT <span class="position-count">${countText}</span>`;
                }
            } else if (blockType === 'LOSING') {
                const titleContainer = headerElement.querySelector('.title-container');
                if (titleContainer) {
                    titleContainer.innerHTML = `${languageUtils.translate('losingPositions')} <span class="position-count">${countText}</span>`;
                }
            }
        }
    }

    updateHighProfitableLabel() {
        const label = domUtils.getElement('total-high-profitable-label');
        if (label) {
            label.textContent = `Всего прибыльных сделок >${this.pnlThreshold}:`;
        }
    }

    setReduceLoad(value) {
        this.reduceLoad = value;
        if (this.lastData) {
            if (!value) {  // Если выключили режим снижения нагрузки
                // Сначала обновляем отображение
                this.updatePositionsDisplay();
                // Затем запускаем загрузку данных
                this.updateAllData();
            } else {  // Если включили режим снижения нагрузки
                // Очищаем кэши
                this.chartCache.clear();
                this.rsiCache.clear();
                // Обновляем отображение
                this.updatePositionsDisplay();
            }
        }
    }
} 

function getExchangeLink(symbol, exchange = 'bybit') {
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
        case 'kucoin':
            return `https://www.kucoin.com/futures/trade/${cleanSymbol}USDTM`;
        default:
            return `https://www.bybit.com/trade/usdt/${cleanSymbol}USDT`; // По умолчанию Bybit
    }
}

function createTickerLink(symbol, exchange) {
    try {
        // Получаем текущую биржу из exchangeManager
        let currentExchange = 'bybit'; // значение по умолчанию
        
        // Проверяем наличие exchangeManager и его метода
        const exchangeManager = window.app?.exchangeManager;
        if (exchangeManager && typeof exchangeManager.getSelectedExchange === 'function') {
            currentExchange = exchangeManager.getSelectedExchange();
            // console.log('Exchange from manager:', currentExchange); // Для отладки
        } else {
            console.warn('ExchangeManager or getSelectedExchange not available');
        }
        
        // Используем переданную биржу или полученную из менеджера
        const selectedExchange = exchange || currentExchange;
        // console.log('Using exchange:', selectedExchange); // Для отладки
        
        return getExchangeLink(symbol, selectedExchange);
    } catch (error) {
        console.warn('Error in createTickerLink:', error);
        return getExchangeLink(symbol, 'bybit');
    }
} 