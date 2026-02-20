/**
 * BotsManager - 16_timeframe
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
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
     */,
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
     */,
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
     */,
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
     */,
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
    });
})();
