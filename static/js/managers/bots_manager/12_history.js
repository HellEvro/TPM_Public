/**
 * BotsManager - 12_history
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            initializeHistoryTab() {
        console.log('[BotsManager] 📊 Инициализация вкладки истории ботов...');,
            initializeAnalyticsTab() {
        const runBtn = document.getElementById('analyticsRunBtn');
        if (runBtn && !runBtn.hasAttribute('data-analytics-bound')) {
            runBtn.setAttribute('data-analytics-bound', 'true');
            runBtn.addEventListener('click', () => this.runTradingAnalytics());
        }
        const syncBtn = document.getElementById('analyticsSyncExchangeBtn');
        if (syncBtn && !syncBtn.hasAttribute('data-sync-bound')) {
            syncBtn.setAttribute('data-sync-bound', 'true');
            syncBtn.addEventListener('click', () => this.syncTradesFromExchange());
        }
        const rsiAuditBtn = document.getElementById('rsiAuditRunBtn');
        if (rsiAuditBtn && !rsiAuditBtn.hasAttribute('data-rsi-audit-bound')) {
            rsiAuditBtn.setAttribute('data-rsi-audit-bound', 'true');
            rsiAuditBtn.addEventListener('click', () => this.runRsiAudit());
        }
        const fullaiBtn = document.getElementById('fullaiAnalyticsRunBtn');
        if (fullaiBtn && !fullaiBtn.hasAttribute('data-fullai-bound')) {
            fullaiBtn.setAttribute('data-fullai-bound', 'true');
            fullaiBtn.addEventListener('click', () => this.loadFullaiAnalytics());
        }
        const aiReanalyzeBtn = document.getElementById('aiReanalyzeBtn');
        if (aiReanalyzeBtn && !aiReanalyzeBtn.hasAttribute('data-ai-reanalyze-bound')) {
            aiReanalyzeBtn.setAttribute('data-ai-reanalyze-bound', 'true');
            aiReanalyzeBtn.addEventListener('click', () => this.runAiReanalyze());
        }
        const subtabBtns = document.querySelectorAll('.analytics-subtab-btn');
        const subtabPanels = document.querySelectorAll('.analytics-subtab-content');
        if (subtabBtns.length && !document.getElementById('analyticsTab').hasAttribute('data-subtabs-bound')) {
            document.getElementById('analyticsTab').setAttribute('data-subtabs-bound', 'true');
            subtabBtns.forEach(btn => {
                btn.addEventListener('click', () => {
                    const id = btn.getAttribute('data-analytics-subtab');
                    subtabBtns.forEach(b => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
                    subtabPanels.forEach(p => {
                        const on = p.getAttribute('data-analytics-subtab') === id;
                        p.classList.toggle('active', on);
                        p.hidden = !on;
                    });
                    btn.classList.add('active');
                    btn.setAttribute('aria-selected', 'true');
                    if (id === 'fullai') this.loadFullaiAnalytics();
                    if (id === 'rsi') this.runRsiAudit();
                });
            });
        }
    }

    /**
     * Загрузка и отображение аналитики FullAI (события и сводка из data/fullai_analytics.db)
     */,
            async loadFullaiAnalytics() {
        const loadingEl = document.getElementById('fullaiAnalyticsLoading');
        const summaryEl = document.getElementById('fullaiAnalyticsSummary');
        const eventsEl = document.getElementById('fullaiAnalyticsEvents');
        const periodHours = parseInt(document.getElementById('fullaiAnalyticsPeriod')?.value, 10) || 168;
        const symbol = (document.getElementById('fullaiAnalyticsSymbol')?.value || '').trim().toUpperCase() || undefined;
        const from_ts = (Date.now() / 1000) - periodHours * 3600;
        const to_ts = Date.now() / 1000;
        if (loadingEl) loadingEl.style.display = 'flex';
        if (summaryEl) summaryEl.innerHTML = '';
        if (eventsEl) eventsEl.innerHTML = '';
        try {
            const params = new URLSearchParams({ from_ts: String(from_ts), to_ts: String(to_ts), limit: '300' });
            if (symbol) params.set('symbol', symbol);
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/analytics/fullai?${params}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            if (!data.success) throw new Error(data.error || 'Нет данных');
            this.renderFullaiAnalytics(data.summary || {}, data.events || [], summaryEl, eventsEl, {
                db_path: data.db_path,
                total_events: data.total_events,
                bot_trades_stats: data.bot_trades_stats || null,
                closed_trades: data.closed_trades || []
            });
        } catch (err) {
            if (summaryEl) summaryEl.innerHTML = `<div class="analytics-error">❌ ${(err && err.message) || String(err)}</div>`;
            if (eventsEl) eventsEl.innerHTML = '';
            console.error('[BotsManager] Ошибка аналитики FullAI:', err);
        } finally {
            if (loadingEl) loadingEl.style.display = 'none';
        }
    },
            renderFullaiAnalytics(summary, events, summaryEl, eventsEl, meta) {
        if (!summaryEl) return;
        const botStats = (meta && meta.bot_trades_stats) || null;
        const totalInDb = (meta && meta.total_events) != null ? meta.total_events : null;
        const dbPath = (meta && meta.db_path) || '';
        const s = summary;
        // Реальные сделки: используем bots_data.db (истинный источник), если есть — иначе fullai_analytics
        const realClose = (botStats != null) ? (botStats.total || 0) : (s.real_close || 0);
        const realWins = (botStats != null) ? (botStats.wins || 0) : (s.real_wins || 0);
        const realLosses = (botStats != null) ? (botStats.losses || 0) : (s.real_losses || 0);
        const winRate = (botStats != null && botStats.win_rate_pct != null) ? String(botStats.win_rate_pct) : (s.real_total > 0 ? ((s.real_wins / s.real_total) * 100).toFixed(1) : '—');
        const virtualRate = s.virtual_total > 0 ? ((s.virtual_ok / s.virtual_total) * 100).toFixed(1) : '—';
        let html = '';
        if (botStats && (botStats.total > 0 || botStats.total_pnl_usdt !== 0)) {
            const wr = botStats.win_rate_pct != null ? botStats.win_rate_pct + '%' : '—';
            const pnlClass = (botStats.total_pnl_usdt || 0) >= 0 ? 'positive' : 'negative';
            const pnlStr = (botStats.total_pnl_usdt != null ? (botStats.total_pnl_usdt >= 0 ? '+' : '') + botStats.total_pnl_usdt : '—') + ' USDT';
            html += '<div class="fullai-bot-trades-block" style="margin-bottom:1rem;padding:0.75rem;background:var(--bg-secondary, #1a1a2e);border-radius:8px;border:1px solid var(--border, #333);">';
            html += '<strong>По сделкам бота (bots_data.db)</strong> — совпадает с монитором «Закрытые PNL»:<br>';
            html += '<span>Сделок: ' + botStats.total + '</span> · <span class="positive">В плюс: ' + (botStats.wins || 0) + '</span> · <span class="negative">В минус: ' + (botStats.losses || 0) + '</span> · Win rate: ' + wr + ' · Суммарный PnL: <span class="' + pnlClass + '">' + pnlStr + '</span></div>';
        }
        let cards = '<div class="fullai-cards">';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Реальные входы</span><span class="fullai-card-value">' + (s.real_open || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Виртуальные входы</span><span class="fullai-card-value">' + (s.virtual_open || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Реальные закрытия</span><span class="fullai-card-value">' + realClose + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Реальные в плюс</span><span class="fullai-card-value positive">' + realWins + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Реальные в минус</span><span class="fullai-card-value negative">' + realLosses + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Win rate (реал.)</span><span class="fullai-card-value">' + winRate + '%</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Вирт. закрытий удачных</span><span class="fullai-card-value">' + (s.virtual_ok || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Вирт. закрытий неудачных</span><span class="fullai-card-value">' + (s.virtual_fail || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Успешность вирт.</span><span class="fullai-card-value">' + virtualRate + '%</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Блокировок входа</span><span class="fullai-card-value">' + (s.blocked || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Отказов ИИ</span><span class="fullai-card-value">' + (s.refused || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Смен параметров</span><span class="fullai-card-value">' + (s.params_change || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Раундов → реал.</span><span class="fullai-card-value">' + (s.round_success || 0) + '</span></div>';
        cards += '<div class="fullai-card"><span class="fullai-card-label">Решений держать</span><span class="fullai-card-value">' + (s.exit_hold || 0) + '</span></div>';
        cards += '</div>';
        html += '<p class="fullai-events-note" style="font-size:0.85rem;color:var(--text-muted,#888);margin-top:0.25rem;">Карточки «Реальные закрытия/в плюс/в минус/Win rate» — из bots_data.db (история ботов). Остальные карточки — события FullAI (записываются только при включённом FullAI).</p>';
        summaryEl.innerHTML = html + cards;

        let closedTradesHtml = '';
        const closedTrades = (meta && meta.closed_trades) || [];,
            async runRsiAudit() {
        const loadingEl = document.getElementById('rsiAuditLoading');
        const resultEl = document.getElementById('rsiAuditResult');
        const limitEl = document.getElementById('rsiAuditLimit');
        const limit = (limitEl && parseInt(limitEl.value, 10)) || 500;
        if (loadingEl) loadingEl.style.display = 'flex';
        if (resultEl) resultEl.innerHTML = '';
        try {
            const response = await fetch(this.BOTS_SERVICE_URL + '/api/bots/analytics/rsi-audit?limit=' + Math.min(2000, Math.max(50, limit)));
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            if (!data.success || !data.report) throw new Error(data.error || 'Нет данных отчёта');
            this.renderRsiAuditReport(data.report, resultEl);
        } catch (err) {
            if (resultEl) resultEl.innerHTML = '<div class="analytics-error">❌ ' + ((err && err.message) || String(err)) + '</div>';
            console.error('[BotsManager] Ошибка аудита RSI:', err);
        } finally {
            if (loadingEl) loadingEl.style.display = 'none';
        }
    }

    /**
     * Рендер отчёта аудита RSI: сводка, конфиг, таблица сделок (ошибочные входы/выходы подсвечены)
     */,
            renderRsiAuditReport(report, container) {
        if (!container) return;
        const cfg = report.config || {};
        const tf = report.timeframe || '1m';
        const sum = report.summary || {};
        const trades = report.trades || [];
        let html = '<div class="rsi-audit-report">';
        html += '<div class="rsi-audit-summary">';
        html += '<h4>Сводка</h4>';
        html += `<p><strong>Всего сделок:</strong> ${sum.total || 0}</p>`;
        html += '<p><strong>Вход:</strong> ';
        html += `✅ по порогу: ${sum.entry_ok || 0} · `;
        html += `<span class="rsi-audit-error">❌ ошибочных (вне порога): ${sum.entry_error || 0}</span> · `;
        html += `без RSI: ${sum.entry_no_rsi || 0}</p>`;
        html += '<p><strong>Выход:</strong> ';
        html += `✅ по порогу: ${sum.exit_ok || 0} · `;
        html += `<span class="rsi-audit-error">❌ вне порога: ${sum.exit_error || 0}</span> · `;
        html += `без RSI: ${sum.exit_no_rsi || 0}</p>`;
        html += '</div>';
        html += '<div class="rsi-audit-config">';
        html += '<h4>Текущий конфиг (эталон)</h4>';
        html += `<p>Таймфрейм: <strong>${tf}</strong> · LONG: RSI ≤ ${cfg.rsi_long_threshold ?? 29} · SHORT: RSI ≥ ${cfg.rsi_short_threshold ?? 71}</p>`;
        html += `<p>Выход LONG: RSI ≥ ${cfg.rsi_exit_long_with_trend ?? 65} (по тренду) / ${cfg.rsi_exit_long_against_trend ?? 60} (против) · Выход SHORT: RSI ≤ ${cfg.rsi_exit_short_with_trend ?? 35} / ${cfg.rsi_exit_short_against_trend ?? 40}</p>`;
        html += '</div>';
        html += '<div class="rsi-audit-table-wrap"><h4>Сделки</h4><table class="rsi-audit-table"><thead><tr>';
        html += '<th>Символ</th><th>Направление</th><th>Вход (время)</th><th>RSI входа</th><th>Порог входа</th><th>Вход</th>';
        html += '<th>Выход (время)</th><th>RSI выхода</th><th>Порог выхода</th><th>Выход</th><th>PnL</th></tr></thead><tbody>';
        trades.forEach((t, i) => {
            const entryStatus = t.entry_rsi == null ? '—' : (t.entry_ok ? '✅ OK' : '<span class="rsi-audit-error">❌ Ошибка</span>');
            const exitStatus = t.exit_rsi == null ? '—' : (t.exit_ok ? '✅ OK' : '<span class="rsi-audit-error">❌ Ошибка</span>');
            const rowClass = (t.entry_error || t.exit_error) ? 'rsi-audit-row-error' : '';
            html += `<tr class="${rowClass}">`;
            html += `<td>${t.symbol || ''}</td><td>${t.direction || ''}</td>`;
            html += `<td>${t.entry_time_iso || ''}</td><td>${t.entry_rsi != null ? t.entry_rsi : '—'}</td><td>${t.entry_threshold != null ? t.entry_threshold : ''}</td><td>${entryStatus}</td>`;
            html += `<td>${t.exit_time_iso || ''}</td><td>${t.exit_rsi != null ? t.exit_rsi : '—'}</td><td>${t.exit_threshold != null ? t.exit_threshold : ''}</td><td>${exitStatus}</td>`;
            html += `<td>${t.pnl != null ? Number(t.pnl).toFixed(4) : ''}</td>`;
            html += '</tr>';
        });
        html += '</tbody></table></div>';
        html += `<div class="rsi-audit-meta">Отчёт: ${report.generated_at || ''}</div>`;
        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Синхронизирует bot_trades_history с данными биржи (обновляет цены и PnL в БД)
     */,
            async syncTradesFromExchange() {
        const syncBtn = document.getElementById('analyticsSyncExchangeBtn');
        const origText = syncBtn ? syncBtn.textContent : '';
        if (syncBtn) syncBtn.disabled = true;
        try {
            const response = await fetch(this.BOTS_SERVICE_URL + '/api/bots/analytics/sync-from-exchange', { method: 'POST' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            const msg = data.updated != null ? ('Обновлено ' + data.updated + ' из ' + (data.matched || 0) + ' совпавших') : (data.message || 'Готово');
            alert('Синхронизация с биржей: ' + msg);
            if (data.updated > 0) this.runTradingAnalytics();
        } catch (err) {
            alert('Ошибка синхронизации: ' + ((err && err.message) || String(err)));
        } finally {,
            async runAiReanalyze() {
        const btn = document.getElementById('aiReanalyzeBtn');
        const resultEl = document.getElementById('aiReanalyzeResult');
        const origText = btn ? btn.textContent : '';,
            async runTradingAnalytics() {
        const loadingEl = document.getElementById('analyticsLoading');
        const resultEl = document.getElementById('analyticsResult');
        const includeExchange = document.getElementById('analyticsIncludeExchange') && document.getElementById('analyticsIncludeExchange').checked;
        if (loadingEl) loadingEl.style.display = 'flex';
        if (resultEl) resultEl.innerHTML = '';
        try {
            const params = new URLSearchParams({ limit: '10000', include_exchange: includeExchange ? '1' : '0' });
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/analytics?${params}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            if (!data.success || !data.report) throw new Error(data.error || 'Нет данных отчёта');
            this.renderAnalyticsReport(data.report, resultEl);
        } catch (err) {
            if (resultEl) resultEl.innerHTML = '<div class="analytics-error">❌ ' + ((err && err.message) || String(err)) + '</div>';
            console.error('[BotsManager] Ошибка аналитики:', err);
        } finally {
            if (loadingEl) loadingEl.style.display = 'none';
        }
    }

    /**
     * Формирует HTML отчёта аналитики с переключаемыми категориями и вставляет в контейнер
     */,
            renderAnalyticsReport(report, container) {
        if (!container) return;
        const s = report.summary || {};
        const bot = report.bot_analytics || {};
        const categories = [
            { id: 'summary', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_summary')) || 'Сводка' },
            { id: 'bots', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_bots')) || 'Сделки ботов' },
            { id: 'trades_table', label: 'Таблица сделок' },
            { id: 'by_symbol', label: 'По символам' },
            { id: 'by_bot', label: 'По ботам' },
            { id: 'by_decision_source', label: 'По источникам решений' },
            { id: 'reasons', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_reasons')) || 'Причины закрытия' },
            { id: 'unsuccessful_coins', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_unsuccessful_coins')) || 'Неудачные монеты' },
            { id: 'unsuccessful_settings', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_unsuccessful_settings')) || 'Неудачные настройки' },
            { id: 'successful_coins', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_successful_coins')) || 'Удачные монеты' },
            { id: 'successful_settings', label: (window.languageUtils && window.languageUtils.translate('analytics_cat_successful_settings')) || 'Удачные настройки' }
        ];
        let tabsHtml = '<div class="analytics-category-tabs">';
        categories.forEach((cat, i) => {
            tabsHtml += `<button type="button" class="analytics-cat-btn ${i === 0 ? 'active' : ''}" data-category="${cat.id}">${cat.label}</button>`;
        });
        tabsHtml += '</div>';

        let bodyHtml = '<div class="analytics-report">';
        const exchangeCount = s.exchange_trades_count ?? 0;
        const botCountRaw = s.bot_trades_count ?? 0;
        const botCountUnique = (bot.total_trades != null ? bot.total_trades : botCountRaw);
        const onlyBots = s.reconciliation_only_bots ?? 0;
        let summaryNote = '';,
            initializeHistoryFilters() {
        // Фильтр по боту
        const botFilter = document.getElementById('historyBotFilter');
        if (botFilter && !botFilter.hasAttribute('data-listener-bound')) {
            botFilter.addEventListener('change', () => this.loadHistoryData(this.currentHistoryTab));
            botFilter.setAttribute('data-listener-bound', 'true');
        }

        // Фильтр по типу действия
        const actionFilter = document.getElementById('historyActionFilter');
        if (actionFilter && !actionFilter.hasAttribute('data-listener-bound')) {
            actionFilter.addEventListener('change', () => this.loadHistoryData(this.currentHistoryTab));
            actionFilter.setAttribute('data-listener-bound', 'true');
        }

        // Фильтр по периоду
        const dateFilter = document.getElementById('historyDateFilter');
        if (dateFilter && !dateFilter.hasAttribute('data-listener-bound')) {
            dateFilter.addEventListener('change', () => this.loadHistoryData(this.currentHistoryTab));
            dateFilter.setAttribute('data-listener-bound', 'true');
        }

        // Кнопки фильтров
        const applyBtn = document.getElementById('applyHistoryFilters');
        if (applyBtn && !applyBtn.hasAttribute('data-listener-bound')) {
            applyBtn.addEventListener('click', () => this.loadHistoryData(this.currentHistoryTab));
            applyBtn.setAttribute('data-listener-bound', 'true');
        }

        const clearBtn = document.getElementById('clearHistoryFilters');
        if (clearBtn && !clearBtn.hasAttribute('data-listener-bound')) {
            clearBtn.addEventListener('click', () => this.clearHistoryFilters());
            clearBtn.setAttribute('data-listener-bound', 'true');
        }

        const exportBtn = document.getElementById('exportHistoryBtn');
        if (exportBtn && !exportBtn.hasAttribute('data-listener-bound')) {
            exportBtn.addEventListener('click', () => this.exportHistoryData());
            exportBtn.setAttribute('data-listener-bound', 'true');
        }
    }

    /**
     * Инициализирует подвкладки истории
     */,
            initializeHistorySubTabs() {
        const tabButtons = document.querySelectorAll('.history-tab-btn');
        const tabContents = document.querySelectorAll('.history-tab-content');

        tabButtons.forEach(button => {
            if (button.hasAttribute('data-listener-bound')) {
                return;
            }

            button.addEventListener('click', () => {
                const tabName = button.dataset.historyTab;
                
                // Убираем активный класс со всех кнопок и контента
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Добавляем активный класс к выбранной кнопке и контенту
                button.classList.add('active');
                const targetContent = document.getElementById(`${tabName}History`);,
            initializeHistoryActionButtons() {
        // Кнопка обновления
        const refreshBtn = document.getElementById('refreshHistoryBtn');
        if (refreshBtn && !refreshBtn.hasAttribute('data-listener-bound')) {
            refreshBtn.addEventListener('click', () => this.loadHistoryData(this.currentHistoryTab));
            refreshBtn.setAttribute('data-listener-bound', 'true');
        }

        // Кнопка создания демо-данных
        const demoBtn = document.getElementById('createDemoDataBtn');
        if (demoBtn && !demoBtn.hasAttribute('data-listener-bound')) {
            demoBtn.addEventListener('click', () => this.createDemoHistoryData());
            demoBtn.setAttribute('data-listener-bound', 'true');
        }

        // Кнопка очистки истории
        const clearBtn = document.getElementById('clearHistoryBtn');
        if (clearBtn && !clearBtn.hasAttribute('data-listener-bound')) {
            clearBtn.addEventListener('click', () => this.clearAllHistory());
            clearBtn.setAttribute('data-listener-bound', 'true');
        }
    }

    /**
     * Загружает данные истории
     */,
            async loadHistoryData(tabName = null) {
        try {
            const targetTab = tabName || this.currentHistoryTab || 'actions';
            this.currentHistoryTab = targetTab;

            console.log(`[BotsManager] 📊 Загрузка данных истории: ${targetTab}`);
            
            // Получаем параметры фильтров
            const filters = this.getHistoryFilters();
            
            // Загружаем данные в зависимости от вкладки,
            getHistoryFilters() {
        const botFilter = document.getElementById('historyBotFilter');
        const actionFilter = document.getElementById('historyActionFilter');
        const dateFilter = document.getElementById('historyDateFilter');
        
        const symbolValue = botFilter ? (botFilter.value || 'all') : 'all';
        const actionValueRaw = actionFilter ? (actionFilter.value || 'all') : 'all';
        const actionValue = actionValueRaw !== 'all' ? actionValueRaw.toUpperCase() : 'all';
        const periodValue = dateFilter ? (dateFilter.value || 'all') : 'all';

        const decisionSourceFilter = document.getElementById('historyDecisionSourceFilter');
        const resultFilter = document.getElementById('historyResultFilter');
        
        return {
            symbol: symbolValue,
            action_type: actionValue,
            trade_type: actionValue,
            period: periodValue,
            decision_source: decisionSourceFilter ? decisionSourceFilter.value : 'all',
            result: resultFilter ? resultFilter.value : 'all',
            limit: 100
        };
    }
    
    /**
     * Загружает AI историю
     */,
            async loadAIHistory() {
        try {
            // Сначала загружаем статистику, чтобы использовать её как fallback для метрик
            await this.loadAIStats();
            // Затем загружаем остальные данные параллельно
            await Promise.all([
                this.loadAIDecisions(),
                this.loadAIOptimizerSummary(),
                this.loadAITrainingHistory(),
                this.loadAIPerformanceMetrics()
            ]);
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки AI истории:', error);
        }
    }
    /**
     * Загружает статистику AI vs скриптовые
     */,
            async loadAIStats() {
        try {
            // Период из селектора
            const periodSelect = document.getElementById('aiPeriodSelect');
            const rawPeriod = periodSelect ? (periodSelect.value || '7d') : '7d';
            const periodMap = { '24h': 'today', '7d': 'week', '30d': 'month', 'all': 'all' };
            const period = periodMap[rawPeriod] || 'all';
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/stats?period=${encodeURIComponent(period)}`);
            const data = await response.json();
    });
})();
