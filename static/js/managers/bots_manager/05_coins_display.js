/**
 * BotsManager - 05_coins_display
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
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
    },
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
    },
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
    },
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
    },
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
    },
            generateAntiPumpFilterInfo(coin) {
        return this.generateExitScamFilterInfo(coin);
    },
            getRsiZoneClass(rsi) {
        if (rsi <= this.rsiLongThreshold) return 'buy-zone';
        if (rsi >= this.rsiShortThreshold) return 'sell-zone';
        return '';
    },
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
    },
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
            case 'kucoin':
                return `https://www.kucoin.com/futures/trade/${cleanSymbol}USDTM`;
            default:
                return `https://www.bybit.com/trade/usdt/${cleanSymbol}USDT`; // По умолчанию Bybit
        }
    },
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
     */,
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
    },
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
    },
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
    },
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
    },
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
    },
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
    },
            getRsiZone(rsi) {
        if (rsi === '-' || rsi === null || rsi === undefined) return 'NEUTRAL';
        if (rsi <= 30) return 'OVERSOLD';
        if (rsi >= 70) return 'OVERBOUGHT';
        return 'NEUTRAL';
    },
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
    },
            updateStatusIcon(iconId, statusValue) {
        const iconElement = document.getElementById(iconId);
        if (iconElement && statusValue) {
            const icon = this.getStatusIcon('zone', statusValue); // Используем зону как базовую
            iconElement.textContent = icon;
            iconElement.style.display = 'inline';
        } else if (iconElement) {
            iconElement.style.display = 'none';
        }
    },
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
    },
            getStatusIcon(statusType, statusValue) {
        const iconMap = {
            'OVERSOLD': '🔴',
            'OVERBOUGHT': '🟢',
            'NEUTRAL': '🟡',
            'UP': '📈',
            'DOWN': '📉'
        };
        
        return iconMap[statusValue] || '';
    },
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
    },
            filterCoins(searchTerm) {
        const items = document.querySelectorAll('.coin-item');
        const term = searchTerm.toLowerCase();
        
        items.forEach(item => {
            const symbol = item.dataset.symbol.toLowerCase();
            const visible = symbol.includes(term);
            item.style.display = visible ? 'block' : 'none';
        });
    },
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
    },
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
    });
})();
