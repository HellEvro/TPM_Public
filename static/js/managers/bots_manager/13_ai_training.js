/**
 * BotsManager - 13_ai_training
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
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
     */,
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
     */,
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
     */,
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
     */,
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
     */,
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
    },
            getAITrainingStatusMeta(status) {
        const normalized = (status || 'SUCCESS').toUpperCase();
        const meta = {
            'SUCCESS': { icon: '✅', className: 'success' },
            'FAILED': { icon: '❌', className: 'failed' },
            'SKIPPED': { icon: '⏸️', className: 'skipped' }
        };
        return meta[normalized] || meta.SUCCESS;
    },
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
     */,
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
     */,
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
     */,
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
    },
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
     */,
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
     */,
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
     */,
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
     */,
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
     */,
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
    });
})();
