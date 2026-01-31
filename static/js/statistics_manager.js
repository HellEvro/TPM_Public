class StatisticsManager {
    constructor() {
        Logger.info('STATS', 'Initializing StatisticsManager');
        
        // Получаем stateManager (через window для надежности)
        const stateManager = window.stateManager;
        if (!stateManager) {
            Logger.error('STATS', 'StateManager not found! Make sure state_manager.js is loaded before statistics_manager.js');
            throw new Error('StateManager is not initialized. Check that state_manager.js is included in index.html');
        }
        
        this.stateManager = stateManager; // Сохраняем ссылку для использования в методах
        
        // Подписываемся на изменения состояния
        this.unsubscribers = [
            stateManager.subscribe('app.theme', this.handleThemeChange.bind(this)),
            stateManager.subscribe('positions.data', this.handlePositionsUpdate.bind(this))
        ];

        // Инициализируем состояние для PnL графика
        const state = stateManager.getState('statistics');
        this.chartData = {
            labels: state?.chartData?.labels || [],
            values: state?.chartData?.values || []
        };
        
        this.chartId = `chart_${Math.random().toString(36).substr(2, 9)}`;
        this.chart = null;
        this.isFirstUpdate = true;
        this.lastChartUpdate = 0;

        // Сохраняем начальное состояние
        stateManager.setState('statistics.chartData', this.chartData);
        stateManager.setState('statistics.isLoading', false);

        // Инициализируем график только если мы на странице позиций
        if (document.querySelector('.positions-container')) {
            requestAnimationFrame(() => this.initializeChart());
        }
    }

    handlePositionsUpdate(data) {
        if (data?.stats) {
            this.updateStats(data.stats);
        }
    }

    handleThemeChange(newTheme) {
        Logger.debug('STATS', `Theme changed to: ${newTheme}`);
        this.updateChartTheme(newTheme);
    }

    updateStats(stats) {
        try {
            Logger.debug('STATS', 'Updating statistics:', stats);
            this.stateManager.setState('statistics.isUpdating', true);

            // Обновляем значения статистики
            this.updateStatValues(stats);
            
            // Обновляем график PnL
            if (this.isFirstUpdate) {
                Logger.info('STATS', 'First update, initializing chart with PnL:', stats.total_pnl);
                const timeLabel = formatUtils.formatTime(new Date());
                this.updateChart(stats.total_pnl, timeLabel);
                this.isFirstUpdate = false;
                this.lastChartUpdate = Date.now();
            } else {
                const currentTime = Date.now();
                if (currentTime - this.lastChartUpdate >= CHART_UPDATE_INTERVAL) {
                    Logger.debug('STATS', 'Updating chart with PnL:', stats.total_pnl);
                    const timeLabel = formatUtils.formatTime(new Date());
                    this.updateChart(stats.total_pnl, timeLabel);
                    this.lastChartUpdate = currentTime;
                }
            }

            this.stateManager.setState('statistics.lastUpdate', new Date().toISOString());
            this.stateManager.setState('statistics.data', stats);

        } catch (error) {
            Logger.error('STATS', 'Error updating statistics:', error);
            this.stateManager.setState('statistics.error', error.message);
        } finally {
            this.stateManager.setState('statistics.isUpdating', false);
        }
    }

    updateStatValues(stats) {
        const updates = {
            'total-pnl': { value: stats.total_pnl, useSign: true },
            'total-profit': { value: stats.total_profit },
            'total-loss': { value: stats.total_loss },
            'total-trades': { value: stats.total_trades },
            'total-high-profitable': { value: stats.high_profitable_count },
            'total-all-profitable': { value: stats.profitable_count },
            'total-losing': { value: stats.losing_count }
        };

        Object.entries(updates).forEach(([elementId, { value, useSign }]) => {
            this.updateStatValue(elementId, value, useSign);
        });
        
        // Обновляем TOP-3
        if (stats.top_profitable) {
            this.updateTopPositions('top-profitable', stats.top_profitable, true);
        }
        if (stats.top_losing) {
            this.updateTopPositions('top-losing', stats.top_losing, false);
        }
    }

    updateStatValue(elementId, value, useSign = false) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const isTradeCount = [
            'total-trades',
            'total-high-profitable',
            'total-all-profitable',
            'total-losing'
        ].includes(elementId);

        element.textContent = isTradeCount ? 
            Math.round(value) : 
            `${formatUtils.formatUsdt(value)} USDT`;

        if (useSign) {
            element.className = `stats-value ${value >= 0 ? 'positive' : 'negative'}`;
        }
    }

    updateTopPositions(elementId, positions, isProfit = true) {
        if (!positions || !Array.isArray(positions)) {
            Logger.warn(`STATS', Invalid positions data for ${elementId}`);
            return;
        }

        Logger.debug(`STATS', Updating ${elementId} with ${positions.length} positions`);
        
        const html = positions.map(pos => {
            const pnlValue = isProfit ? pos.pnl : -Math.abs(pos.pnl);
            return `
                <div class="stats-value ${isProfit ? CSS_CLASSES.POSITIVE : CSS_CLASSES.NEGATIVE}">
                    <a href="${createTickerLink(pos.symbol, window.app?.exchangeManager?.getSelectedExchange())}" 
                       target="_blank" 
                       class="ticker">
                        ${pos.symbol}
                    </a>
                    <span style="margin-left: 10px;">
                        ${formatUtils.formatUsdt(pnlValue)} USDT
                    </span>
                </div>
            `;
        }).join('');
        
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = html;
            Logger.debug(`STATS', ${elementId} updated successfully`);
        }
    }

    updateChart(totalPnl, timeLabel) {
        if (!this.chart) {
            Logger.warn('STATS', 'Chart not initialized');
            return;
        }

        try {
            this.chartData.labels.push(timeLabel);
            this.chartData.values.push(totalPnl);

            if (this.chartData.labels.length > CHART_CONFIG.MAX_DATA_POINTS) {
                this.chartData.labels.shift();
                this.chartData.values.shift();
            }

            this.chart.data.labels = this.chartData.labels;
            this.chart.data.datasets[0].data = this.chartData.values;
            this.updateChartColors(totalPnl);
            this.updateChartTheme(this.stateManager.getState('app.theme'));

            this.stateManager.setState('statistics.chartData', this.chartData);

            requestAnimationFrame(() => this.chart.update());
            
            Logger.debug('STATS', 'Chart updated. Points:', this.chartData.values.length, 'Last value:', totalPnl);
        } catch (error) {
            Logger.error('STATS', 'Error updating chart:', error);
        }
    }

    updateChartColors(totalPnl) {
        const theme = this.stateManager.getState('app.theme') || 'dark';
        const themeKey = theme === 'light' ? 'LIGHT' : 'DARK';
        const colors = totalPnl >= 0 ? 
            {
                BORDER: CHART_THEMES[themeKey].UPTREND,
                BACKGROUND: this.hexToRgba(CHART_THEMES[themeKey].UPTREND, 0.2)
            } : 
            {
                BORDER: CHART_THEMES[themeKey].DOWNTREND,
                BACKGROUND: this.hexToRgba(CHART_THEMES[themeKey].DOWNTREND, 0.2)
            };
        
        if (this.chart && this.chart.data.datasets[0]) {
            this.chart.data.datasets[0].borderColor = colors.BORDER;
            this.chart.data.datasets[0].backgroundColor = colors.BACKGROUND;
        }
    }

    updateChartTheme(theme) {
        if (!this.chart) return;

        try {
            const themeKey = theme === 'light' ? 'LIGHT' : 'DARK';
            const totalPnl = this.chartData.values[this.chartData.values.length - 1] || 0;

            this.chart.data.datasets[0].borderColor = totalPnl >= 0 ? CHART_THEMES[themeKey].UPTREND : CHART_THEMES[themeKey].DOWNTREND;
            this.chart.data.datasets[0].backgroundColor = this.hexToRgba(
                totalPnl >= 0 ? CHART_THEMES[themeKey].UPTREND : CHART_THEMES[themeKey].DOWNTREND,
                0.2
            );

            // Обновляем цвет сетки
            const gridColor = theme === 'light' ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.1)';
            if (this.chart.options.scales && this.chart.options.scales.y) {
                this.chart.options.scales.y.grid.color = gridColor;
            }

            this.chart.update();
        } catch (error) {
            Logger.error('STATS', 'Error updating chart theme:', error);
        }
    }

    destroy() {
        // Отписываемся от всех подписок
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        
        // Уничтожаем график
        this.destroyChart();
        
        // Очищаем данные
        this.chartData = { labels: [], values: [] };
        if (this.stateManager) {
            this.stateManager.setState('statistics.chartData', this.chartData);
        }
    }

    destroyChart() {
        Logger.info('STATS', 'Destroying chart');
        if (this.chart) {
            try {
                this.chart.destroy();
                this.chart = null;
                
                if (Chart.getChart(this.chartId)) {
                    Chart.getChart(this.chartId).destroy();
                }
            } catch (error) {
                Logger.error('STATS', 'Error destroying chart:', error);
            }
        }
    }

    initializeChart() {
        try {
            Logger.info('STATS', 'Initializing PnL chart');
            this.destroyChart();

            const ctx = document.getElementById('pnlChart');
            if (!ctx) {
                Logger.warn('STATS', 'Chart canvas not found');
                return;
            }

            const theme = this.stateManager.getState('app.theme') || 'dark';
            const themeKey = theme === 'light' ? 'LIGHT' : 'DARK';
            const gridColor = theme === 'light' ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.1)';

            this.chart = new Chart(ctx, {
                id: this.chartId,
                type: 'line',
                data: {
                    labels: this.chartData.labels,
                    datasets: [{
                        label: 'Total P&L',
                        data: this.chartData.values,
                        borderColor: CHART_THEMES[themeKey].UPTREND,
                        backgroundColor: this.hexToRgba(CHART_THEMES[themeKey].UPTREND, 0.2),
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            grid: {
                                color: gridColor
                            }
                        },
                        x: {
                            grid: {
                                display: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });

            Logger.info('STATS', 'PnL chart initialized successfully');
        } catch (error) {
            Logger.error('STATS', 'Error initializing chart:', error);
            NotificationManager.error('Error initializing statistics chart');
        }
    }

    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
}

// Экспортируем класс
window.StatisticsManager = StatisticsManager;
