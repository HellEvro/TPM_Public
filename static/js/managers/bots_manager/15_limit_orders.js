/**
 * BotsManager - 15_limit_orders
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
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
    },
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
    },
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
            
            const entryCancelEl = document.getElementById('limitOrderEntryCancelSeconds');
            const exitCancelEl = document.getElementById('limitOrderExitCancelSeconds');
            const config = {
                limit_orders_entry_enabled: finalEnabled,
                limit_orders_percent_steps: percentSteps,
                limit_orders_margin_amounts: marginAmounts,
                limit_order_entry_cancel_seconds: entryCancelEl ? Math.max(0, parseInt(entryCancelEl.value, 10) || 0) : 10,
                limit_order_exit_cancel_seconds: exitCancelEl ? Math.max(0, parseInt(exitCancelEl.value, 10) || 0) : 10
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
    },
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
    });
})();
