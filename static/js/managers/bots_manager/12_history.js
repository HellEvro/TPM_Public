/**
 * BotsManager - 12_history
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            initializeHistoryTab() {
        console.log('[BotsManager] 📊 Инициализация вкладки истории ботов...');

        if (!this.historyInitialized) {
            // Инициализируем фильтры
            this.initializeHistoryFilters();

            // Инициализируем подвкладки истории
            this.initializeHistorySubTabs();

            // Инициализируем кнопки действий
            this.initializeHistoryActionButtons();

            this.historyInitialized = true;
        }

        // Загружаем данные для текущей подвкладки
        this.loadHistoryData(this.currentHistoryTab);
    }

    /**
     * Инициализирует вкладку «Аналитика»: привязка кнопок и однократная привязка обработчиков
     */,
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
        const eventFilter = document.getElementById('fullaiAnalyticsEventFilter')?.value || 'all';
        const from_ts = (Date.now() / 1000) - periodHours * 3600;
        const to_ts = Date.now() / 1000;
        if (loadingEl) loadingEl.style.display = 'flex';
        if (summaryEl) summaryEl.innerHTML = '';
        if (eventsEl) eventsEl.innerHTML = '';
        try {
            const params = new URLSearchParams({ from_ts: String(from_ts), to_ts: String(to_ts), limit: '300' });
            if (symbol) params.set('symbol', symbol);
            params.set('_', String(Date.now()));
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/analytics/fullai?${params}`, { cache: 'no-store' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            if (!data.success) throw new Error(data.error || 'Нет данных');
            let events = data.events || [];
            if (eventFilter === 'params_and_virtual') {
                events = events.filter(e => ['params_change', 'virtual_open', 'virtual_close', 'round_success'].indexOf(e.event_type) >= 0);
            } else if (eventFilter === 'entries_only') {
                events = events.filter(e => ['real_open', 'virtual_open'].indexOf(e.event_type) >= 0);
            }
            this.renderFullaiAnalytics(data.summary || {}, events, summaryEl, eventsEl, {
                db_path: data.db_path,
                total_events: data.total_events,
                bot_trades_stats: data.bot_trades_stats || null,
                closed_trades: data.closed_trades || [],
                fullai_configs: data.fullai_configs || null
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

        this._renderFullaiConfigsBlock(meta && meta.fullai_configs);

        // Сначала — события входов/выходов (как FullAI входит: реал./вирт.), затем — закрытые сделки
        let closedTradesHtml = '';
        const closedTrades = (meta && meta.closed_trades) || [];
        if (closedTrades.length > 0) {
            closedTradesHtml = '<h4 style="margin-top:1.5rem;">Закрытые сделки (PnL и вывод)</h4><table class="fullai-events-table"><thead><tr><th>Время</th><th>Символ</th><th>Напр.</th><th>Вход</th><th>Выход</th><th>PnL %</th><th>PnL USDT</th><th>Причина</th><th>Вывод</th></tr></thead><tbody>';
            closedTrades.forEach(tr => {
                const pnlUsdt = tr.pnl_usdt != null ? Number(tr.pnl_usdt) : null;
                const roiPct = tr.roi_pct != null ? Number(tr.roi_pct) : null;
                const pnlClass = (roiPct != null ? (roiPct >= 0 ? 'positive' : 'negative') : (pnlUsdt != null ? (pnlUsdt >= 0 ? 'positive' : 'negative') : ''));
                const pnlPctStr = roiPct != null ? ((roiPct >= 0 ? '+' : '') + roiPct.toFixed(2) + '%') : '—';
                const pnlUsdtStr = tr.is_virtual ? '—' : (pnlUsdt != null ? ((pnlUsdt >= 0 ? '+' : '') + pnlUsdt.toFixed(2)) : '—');
                const entryPrice = tr.entry_price != null ? Number(tr.entry_price).toFixed(6) : '—';
                const exitPrice = tr.exit_price != null ? Number(tr.exit_price).toFixed(6) : '—';
                const conclusion = tr.conclusion || (pnlUsdt >= 0 || roiPct >= 0 ? 'Прибыль' : 'Убыток');
                const virtualBadge = tr.is_virtual ? ' <span class="virtual-pnl-badge" style="background:#9c27b0;color:#fff;padding:1px 6px;border-radius:4px;font-size:10px;">Виртуальная</span>' : '';
                closedTradesHtml += '<tr><td>' + (tr.ts_iso || tr.exit_time || '') + '</td><td>' + (tr.symbol || '') + virtualBadge + '</td><td>' + (tr.direction || '') + '</td><td>' + entryPrice + '</td><td>' + exitPrice + '</td><td class="' + pnlClass + '">' + pnlPctStr + '</td><td class="' + pnlClass + '">' + pnlUsdtStr + '</td><td>' + (tr.close_reason || '—') + '</td><td>' + (conclusion || '—') + '</td></tr>';
            });
            closedTradesHtml += '</tbody></table>';
        }

        if (!eventsEl) return;
        const eventLabels = { real_open: 'Вход реал.', virtual_open: 'Вход вирт.', real_close: 'Закрытие реал.', virtual_close: 'Закрытие вирт.', blocked: 'Блок', refused: 'Отказ ИИ', params_change: 'Смена параметров', round_success: 'Раунд → реал.', exit_hold: 'ИИ держать' };
        if (events.length === 0 && closedTrades.length === 0) {
            let hint = 'Нет событий и закрытых сделок за выбранный период.';
            if (totalInDb === 0) {
                hint = 'В БД 0 событий. Путь: ' + (dbPath || 'data/fullai_analytics.db') + '. Перезапустите сервис ботов после включения FullAI. В логах ботов при записи должна появиться строка «FullAI analytics: запись в БД». Если её нет — решения FullAI не доходят до записи (проверьте, что боты запущены и FullAI включён в Конфигурации).';
            } else if (totalInDb != null && totalInDb > 0) {
                hint = 'В БД всего событий: ' + totalInDb + '. За выбранный период — нет (попробуйте увеличить период).';
            }
            eventsEl.innerHTML = '<p class="analytics-placeholder">' + hint + '</p>';
            return;
        }
        if (events.length === 0 && closedTrades.length > 0) {
            eventsEl.innerHTML = '<h4 style="margin-top:0.5rem;">Последние события FullAI (входы/выходы)</h4><p class="analytics-placeholder">Нет событий входов за период. Реальные входы появляются при создании бота FullAI.</p>' + (closedTrades.length ? '<h4 style="margin-top:1.5rem;">Закрытые сделки (PnL)</h4>' : '') + closedTradesHtml;
            return;
        }
        let table = '<h4 style="margin-top:0.5rem;">Последние события FullAI (входы реал./вирт., выходы, блокировки)</h4>';
        table += '<table class="fullai-events-table"><thead><tr><th>Время</th><th>Символ</th><th>Событие</th><th>Направление</th><th>Вход</th><th>Выход</th><th>PnL %</th><th>PnL USDT</th><th>Лимит выхода</th><th>Тип</th><th>Время заявки</th><th>Проскальз.%</th><th>Задержка с</th><th>Детали</th><th>Вывод</th></tr></thead><tbody>';
        events.forEach(ev => {
            const label = eventLabels[ev.event_type] || ev.event_type;
            const dir = ev.direction || '—';
            const ex = ev.extra || {};
            const entryPrice = ex.entry_price != null ? Number(ex.entry_price).toFixed(6) : (ev.event_type === 'real_open' || ev.event_type === 'refused' ? (ex.price != null ? Number(ex.price).toFixed(6) : '—') : '—');
            const exitPrice = ex.exit_price != null ? Number(ex.exit_price).toFixed(6) : '—';
            const limitExit = ex.limit_price_exit != null ? Number(ex.limit_price_exit).toFixed(6) : '—';
            const orderType = ex.order_type_exit || '—';
            const tsPlaced = ex.ts_order_placed_exit != null ? (function() { const d = new Date(ex.ts_order_placed_exit * 1000); return d.toISOString ? d.toISOString().slice(0, 19).replace('T', ' ') : d.toLocaleString(); })() : '—';
            const slippage = ex.slippage_exit_pct != null ? Number(ex.slippage_exit_pct).toFixed(2) + '%' : '—';
            const delay = ex.delay_sec != null ? String(Number(ex.delay_sec).toFixed(1)) : '—';
            const pnlPct = ev.pnl_percent != null ? Number(ev.pnl_percent) : (ex.pnl_percent != null ? Number(ex.pnl_percent) : null);
            const pnlUsdt = ex.pnl_usdt != null ? Number(ex.pnl_usdt) : null;
            const pnlClass = pnlPct != null ? (pnlPct >= 0 ? 'positive' : 'negative') : (pnlUsdt != null ? (pnlUsdt >= 0 ? 'positive' : 'negative') : '');
            const pnlStr = pnlPct != null ? ((pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(2) + '%') : '—';
            const pnlUsdtStr = ev.event_type === 'virtual_close' ? '—' : (pnlUsdt != null ? ((pnlUsdt >= 0 ? '+' : '') + pnlUsdt.toFixed(2)) : '—');
            let details = '—';
            let conclusion = '—';
            if (ev.event_type === 'params_change') {
                details = ev.reason || 'Мутация';
                const parts = [];
                if (ex.new_rsi_long != null) parts.push('RSI L=' + ex.new_rsi_long);
                if (ex.new_rsi_short != null) parts.push('S=' + ex.new_rsi_short);
                if (ex.new_tp != null) parts.push('TP=' + ex.new_tp + '%');
                if (ex.new_sl != null) parts.push('SL=' + ex.new_sl + '%');
                conclusion = parts.length ? parts.join(', ') : '—';
            } else if (ev.event_type === 'virtual_close') {
                const ok = ex.success !== false;
                details = ok ? '✅ Успех' : '❌ Убыток';
                conclusion = pnlStr !== '—' ? (ok ? '✅ ' + pnlStr : '❌ ' + pnlStr) : (ok ? '✅ В плюс' : '❌ В минус');
            } else if (ev.event_type === 'virtual_open') {
                details = ex.entry_price != null ? 'Вход ' + Number(ex.entry_price).toFixed(6) : '—';
                conclusion = '—';
            } else {
                details = ev.reason || (ex.success !== undefined ? (ex.success ? 'успех' : 'убыток') : '') || '—';
                conclusion = pnlPct != null ? (pnlPct >= 0 ? 'Прибыль. ' + (ev.reason || '') : 'Убыток. ' + (ev.reason || '')) : '—';
            }
            const rowClass = ev.event_type === 'params_change' ? 'fullai-event-params' : (ev.event_type === 'virtual_close' ? (ex.success ? 'fullai-event-virt-ok' : 'fullai-event-virt-fail') : '');
            table += '<tr class="' + rowClass + '"><td>' + (ev.ts_iso || '') + '</td><td>' + (ev.symbol || '') + '</td><td>' + label + '</td><td>' + dir + '</td><td>' + entryPrice + '</td><td>' + exitPrice + '</td><td class="' + pnlClass + '">' + pnlStr + '</td><td class="' + pnlClass + '">' + pnlUsdtStr + '</td><td>' + limitExit + '</td><td>' + orderType + '</td><td>' + tsPlaced + '</td><td>' + slippage + '</td><td>' + delay + '</td><td>' + details + '</td><td>' + conclusion + '</td></tr>';
        });
        table += '</tbody></table>';
        eventsEl.innerHTML = table + closedTradesHtml;
    },

    _renderFullaiConfigsBlock(fullaiConfigs) {
        const selectEl = document.getElementById('fullaiConfigCoinSelect');
        const currentEl = document.getElementById('fullaiConfigCurrent');
        const previousEl = document.getElementById('fullaiConfigPrevious');
        if (!selectEl || !currentEl || !previousEl) return;
        const cfg = fullaiConfigs || { global_config: {}, coin_configs: {} };
        const coins = Object.keys(cfg.coin_configs || {}).sort();
        const options = [{ value: '_global', text: 'Глобальный конфиг' }];
        coins.forEach(sym => options.push({ value: sym, text: sym }));
        selectEl.innerHTML = options.map(o => '<option value="' + o.value + '">' + o.text + '</option>').join('');
        const renderSelected = () => {
            const val = selectEl.value;
            let current = null, previous = null, updatedAt = null;
            if (val === '_global') {
                current = cfg.global_config || {};
                previous = null;
            } else {
                const coin = (cfg.coin_configs || {})[val];
                if (coin) {
                    current = coin.current || {};
                    previous = coin.previous || null;
                    updatedAt = coin.updated_at || null;
                }
            }
            currentEl.textContent = Object.keys(current || {}).length ? JSON.stringify(current, null, 2) : '—';
            previousEl.textContent = previous && Object.keys(previous).length ? JSON.stringify(previous, null, 2) : '—';
            currentEl.setAttribute('title', updatedAt && val !== '_global' ? 'Обновлено: ' + updatedAt : '');
        };
        if (!selectEl.hasAttribute('data-fullai-config-bound')) {
            selectEl.setAttribute('data-fullai-config-bound', 'true');
            selectEl.addEventListener('change', renderSelected);
        }
        renderSelected();
    }

    /**
     * Запускает аудит RSI входа/выхода и отображает отчёт
     */,
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
        } finally {
            if (syncBtn) { syncBtn.disabled = false; syncBtn.textContent = origText; }
        }
    }

    /**
     * Запускает ручной анализ ИИ: обновление данных, подход к сделкам и переобучение (в фоне).
     * Показывает изменения в формате «старое → новое».
     */,
            async runAiReanalyze() {
        const btn = document.getElementById('aiReanalyzeBtn');
        const resultEl = document.getElementById('aiReanalyzeResult');
        const origText = btn ? btn.textContent : '';
        if (btn) { btn.disabled = true; btn.textContent = '⏳ Запуск...'; }
        if (resultEl) { resultEl.style.display = 'none'; resultEl.innerHTML = ''; }
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/analytics/ai-reanalyze`, { method: 'POST' });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
            if (!data.success) throw new Error(data.error || 'Не удалось запустить');

            const changes = data.changes || [];
            if (resultEl) {
                resultEl.style.display = 'block';
                if (changes.length > 0) {
                    const paramNames = {
                        take_profit_percent: 'TP%',
                        max_loss_percent: 'SL%',
                        rsi_long_threshold: 'RSI long',
                        rsi_short_threshold: 'RSI short'
                    };
                    const isPercent = (p) => p === 'take_profit_percent' || p === 'max_loss_percent';
                    let html = '<strong>🧠 Изменения ИИ:</strong><ul style="margin: 6px 0 0 16px;">';
                    changes.forEach(c => {
                        const p = paramNames[c.param] || c.param;
                        const suf = isPercent(c.param) ? '%' : '';
                        html += `<li><code>${c.symbol}</code> ${p}: <span style="text-decoration:line-through">${c.old}${suf}</span> → <strong>${c.new}${suf}</strong></li>`;
                    });
                    html += '</ul>';
                    html += '<p style="margin: 8px 0 0; color: var(--text-muted, #666); font-size: 0.85em;">' + (data.message || '') + '</p>';
                    resultEl.innerHTML = html;
                } else {
                    resultEl.innerHTML = '<strong>🧠</strong> ' + (data.message || 'Готово. Изменений параметров нет.');
                }
            } else {
                alert(data.message || 'ИИ анализирует и обновляет данные в фоне.');
            }
        } catch (err) {
            if (resultEl) {
                resultEl.style.display = 'block';
                resultEl.innerHTML = '<span class="analytics-error">❌ ' + ((err && err.message) || String(err)) + '</span>';
            } else {
                alert('Ошибка: ' + ((err && err.message) || String(err)));
            }
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = origText; }
        }
    }

    /**
     * Запускает аналитику торговли и отображает результат во вкладке «Аналитика»
     */,
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
        let summaryNote = '';
        if (botCountRaw > exchangeCount && exchangeCount > 0) {
            summaryNote = '<p class="analytics-summary-note">В БД записей больше, чем биржа вернула по API: у биржи ограничена история (например 2 года или лимит страниц). «Только в БД» — сделки из БД без пары в ответе API (часто старые). В БД учтены закрытия ботов и ручные через интерфейс.</p>';
        }
        const botCountNote = (botCountUnique < botCountRaw) ? ` <small>(уникальных: ${botCountUnique}, всего записей в БД: ${botCountRaw})</small>` : ` <small>(всего записей в БД)</small>`;
        const series = bot.consecutive_series || {};
        const dd = bot.drawdown || {};
        const pfStr = bot.profit_factor != null ? (bot.profit_factor >= 999 ? '∞' : bot.profit_factor.toFixed(2)) : '—';
        var possibleErrorsHtml = '';
        if ((bot.possible_errors_count || 0) > 0) {
            var errs = Array.isArray(bot.possible_errors) ? bot.possible_errors.slice(0, 20) : [];
            possibleErrorsHtml = '<h4>⚠ Возможные ошибки по сделкам</h4><p>Найдено: <strong>' + bot.possible_errors_count + '</strong>.</p>';
            if (errs.length > 0) {
                possibleErrorsHtml += '<div class="analytics-stats-table-wrap"><table class="analytics-stats-table"><thead><tr><th>Символ</th><th>Время</th><th>PnL</th><th>Причина</th></tr></thead><tbody>';
                for (var i = 0; i < errs.length; i++) {
                    var e = errs[i];
                    var ts = e.exit_timestamp ? new Date(e.exit_timestamp * 1000).toISOString().slice(0, 19) : '—';
                    var reason = String(e.close_reason != null ? e.close_reason : '—').slice(0, 30);
                    possibleErrorsHtml += '<tr><td>' + (e.symbol || '—') + '</td><td>' + ts + '</td><td>' + (e.pnl != null ? e.pnl : '—') + '</td><td>' + reason + '</td></tr>';
                }
                possibleErrorsHtml += '</tbody></table></div>';
            }
        }
        bodyHtml += '<div class="analytics-section" data-category="summary">' +
            '<h3>' + categories[0].label + '</h3>' +
            '<h4 style="margin-top:0;">Метрики торговли</h4>' +
            '<p>Сделок: <strong>' + (bot.total_trades != null ? bot.total_trades : botCountUnique) + '</strong> · Прибыльных: <strong>' + (bot.win_count ?? '—') + '</strong> · Убыточных: <strong>' + (bot.loss_count ?? '—') + '</strong> · Нулевых: <strong>' + (bot.neutral_count ?? '—') + '</strong><br>' +
            'Win Rate: <strong>' + (s.bot_win_rate_pct != null ? s.bot_win_rate_pct + '%' : '—') + '</strong> · Суммарный PnL: <strong>' + (s.bot_total_pnl_usdt != null ? s.bot_total_pnl_usdt + ' USDT' : '—') + '</strong> · Profit Factor: <strong>' + pfStr + '</strong></p>' +
            '<p>Средняя прибыль на сделку: <strong>' + (bot.avg_win_usdt != null ? bot.avg_win_usdt + ' USDT' : '—') + '</strong> · Средний убыток: <strong>' + (bot.avg_loss_usdt != null ? bot.avg_loss_usdt + ' USDT' : '—') + '</strong></p>' +
            '<p>Макс. серия побед: <strong>' + (series.max_consecutive_wins ?? '—') + '</strong> · Макс. серия убытков: <strong>' + (series.max_consecutive_losses ?? '—') + '</strong> · Просадка: <strong>' + (dd.max_drawdown_usdt != null ? dd.max_drawdown_usdt + ' USDT' : '—') + (dd.max_drawdown_pct != null ? ' (' + dd.max_drawdown_pct + '%)' : '') + '</strong></p>' +
            possibleErrorsHtml +
            '<h4>Сверка с биржей</h4>' +
            '<p><strong>С биржи (по API):</strong> ' + exchangeCount + ' · <strong>В БД</strong> (закрытия ботов и ручные): <strong>' + botCountUnique + '</strong>' + botCountNote + '<br>' +
            'Совпадений: <strong>' + (s.reconciliation_matched ?? 0) + '</strong> · Только в ответе биржи: <strong>' + (s.reconciliation_only_exchange ?? 0) + '</strong> · ' +
            'Только в БД (нет пары в ответе API): <strong>' + onlyBots + '</strong> · Расхождений PnL: <strong>' + (s.reconciliation_pnl_mismatches ?? 0) + '</strong></p>' +
            summaryNote +
            '<p class="analytics-summary-note" style="margin-top: 6px;">В отчёте учтены только уникальные сделки: дубликаты отброшены по времени закрытия.</p>' +
            '</div>';

        bodyHtml += '<div class="analytics-section" data-category="bots">';
        if (bot.total_trades != null) {
            const series = bot.consecutive_series || {};
            const dd = bot.drawdown || {};
            const pfVal = bot.profit_factor != null ? (bot.profit_factor >= 999 ? '∞' : bot.profit_factor.toFixed(2)) : '—';
            bodyHtml += '<h3>' + (categories[1].label || '') + '</h3><p>Всего сделок: <strong>' + bot.total_trades + '</strong> · Прибыльных: <strong>' + (bot.win_count ?? 0) + '</strong> · Убыточных: <strong>' + (bot.loss_count ?? 0) + '</strong> · Нулевых: <strong>' + (bot.neutral_count ?? 0) + '</strong></p>';
            bodyHtml += '<p>PnL: <strong>' + bot.total_pnl_usdt + ' USDT</strong> · Win Rate: <strong>' + bot.win_rate_pct + '%</strong> · Profit Factor: <strong>' + pfVal + '</strong></p>';
            bodyHtml += '<p>Средняя прибыль: <strong>' + (bot.avg_win_usdt != null ? bot.avg_win_usdt + ' USDT' : '—') + '</strong> · Средний убыток: <strong>' + (bot.avg_loss_usdt != null ? bot.avg_loss_usdt + ' USDT' : '—') + '</strong></p>';
            bodyHtml += '<p>Макс. серия побед: <strong>' + (series.max_consecutive_wins ?? 0) + '</strong> · Макс. серия убытков: <strong>' + (series.max_consecutive_losses ?? 0) + '</strong> · Просадка: <strong>' + (dd.max_drawdown_usdt ?? 0) + ' USDT</strong> (' + (dd.max_drawdown_pct ?? 0) + '%)</p>';
        } else {
            bodyHtml += '<p>Нет данных</p>';
        }
        bodyHtml += '</div>';

        const tradesList = bot.trades || [];
        bodyHtml += '<div class="analytics-section" data-category="trades_table"><h3>Таблица сделок</h3><p>Показано последних <strong>' + tradesList.length + '</strong> сделок (символ, дата выхода, направление, цены, объём, PnL, причина, источник, RSI, тренд).</p>';
        bodyHtml += '<div class="analytics-trades-table-wrap"><table class="analytics-trades-table"><thead><tr>';
        bodyHtml += '<th>Дата выхода</th><th>Символ</th><th>Направление</th><th>Вход</th><th>Выход</th><th>Объём USDT</th><th>PnL</th><th>Причина</th><th>Источник</th><th>RSI</th><th>Тренд</th></tr></thead><tbody>';
        tradesList.slice(-500).reverse().forEach(tr => {
            const pnlClass = (tr.pnl || 0) > 0 ? 'pnl-win' : ((tr.pnl || 0) < 0 ? 'pnl-loss' : '');
            bodyHtml += '<tr>';
            bodyHtml += '<td>' + (tr.exit_time_iso || '').replace('T', ' ').slice(0, 19) + '</td>';
            bodyHtml += '<td>' + (tr.symbol || '') + '</td><td>' + (tr.direction || '') + '</td>';
            bodyHtml += '<td>' + (tr.entry_price != null ? Number(tr.entry_price).toFixed(6) : '—') + '</td><td>' + (tr.exit_price != null ? Number(tr.exit_price).toFixed(6) : '—') + '</td>';
            bodyHtml += '<td>' + (tr.position_size_usdt != null ? Number(tr.position_size_usdt).toFixed(2) : '—') + '</td>';
            bodyHtml += '<td class="' + pnlClass + '">' + (tr.pnl != null ? Number(tr.pnl).toFixed(4) : '—') + '</td>';
            bodyHtml += '<td>' + (tr.close_reason || '—').slice(0, 20) + '</td><td>' + (tr.decision_source || '—').slice(0, 15) + '</td>';
            bodyHtml += '<td>' + (tr.entry_rsi != null ? tr.entry_rsi : '—') + '</td><td>' + (tr.entry_trend || '—') + '</td>';
            bodyHtml += '</tr>';
        });
        bodyHtml += '</tbody></table></div></div>';

        const bySymbol = bot.by_symbol || {};
        bodyHtml += '<div class="analytics-section" data-category="by_symbol"><h3>По символам</h3><p>Сделок, PnL, победы/убытки/нулевые, Win Rate по каждому символу.</p>';
        bodyHtml += '<div class="analytics-stats-table-wrap"><table class="analytics-stats-table"><thead><tr><th>Символ</th><th>Сделок</th><th>PnL USDT</th><th>Победы</th><th>Убытки</th><th>Нулевые</th><th>Win Rate %</th></tr></thead><tbody>';
        Object.entries(bySymbol).sort((a, b) => (b[1].count || 0) - (a[1].count || 0)).forEach(([sym, d]) => {
            const wr = (d.count && d.wins != null) ? ((d.wins / d.count) * 100).toFixed(1) : '—';
            const pnlClass = (d.pnl || 0) >= 0 ? 'pnl-win' : 'pnl-loss';
            bodyHtml += '<tr><td>' + sym + '</td><td>' + (d.count ?? 0) + '</td><td class="' + pnlClass + '">' + (d.pnl || 0).toFixed(2) + '</td><td>' + (d.wins ?? 0) + '</td><td>' + (d.losses ?? 0) + '</td><td>' + (d.neutral ?? 0) + '</td><td>' + wr + '</td></tr>';
        });
        bodyHtml += '</tbody></table></div></div>';

        const byBot = bot.by_bot || {};
        bodyHtml += '<div class="analytics-section" data-category="by_bot"><h3>По ботам</h3><p>Статистика по каждому bot_id.</p>';
        bodyHtml += '<div class="analytics-stats-table-wrap"><table class="analytics-stats-table"><thead><tr><th>Bot ID</th><th>Сделок</th><th>PnL USDT</th><th>Победы</th><th>Убытки</th><th>Нулевые</th><th>Win Rate %</th></tr></thead><tbody>';
        Object.entries(byBot).sort((a, b) => (b[1].count || 0) - (a[1].count || 0)).forEach(([bid, d]) => {
            const wr = (d.count && d.wins != null) ? ((d.wins / d.count) * 100).toFixed(1) : '—';
            const pnlClass = (d.pnl || 0) >= 0 ? 'pnl-win' : 'pnl-loss';
            bodyHtml += '<tr><td>' + bid + '</td><td>' + (d.count ?? 0) + '</td><td class="' + pnlClass + '">' + (d.pnl || 0).toFixed(2) + '</td><td>' + (d.wins ?? 0) + '</td><td>' + (d.losses ?? 0) + '</td><td>' + (d.neutral ?? 0) + '</td><td>' + wr + '</td></tr>';
        });
        bodyHtml += '</tbody></table></div></div>';

        const byDecision = bot.by_decision_source || {};
        bodyHtml += `<div class="analytics-section" data-category="by_decision_source"><h3>По источникам решений</h3><p>Статистика по источнику решения (FullAI, RSI, и т.д.).</p>`;
        bodyHtml += '<div class="analytics-stats-table-wrap"><table class="analytics-stats-table"><thead><tr><th>Источник</th><th>Сделок</th><th>PnL USDT</th><th>Победы</th><th>Убытки</th><th>Нулевые</th><th>Win Rate %</th></tr></thead><tbody>';
        Object.entries(byDecision).sort((a, b) => (b[1].count || 0) - (a[1].count || 0)).forEach(([src, d]) => {
            const wr = (d.count && d.wins != null) ? ((d.wins / d.count) * 100).toFixed(1) : '—';
            const pnlClass = (d.pnl || 0) >= 0 ? 'pnl-win' : 'pnl-loss';
            bodyHtml += `<tr><td>${src}</td><td>${d.count ?? 0}</td><td class="${pnlClass}">${(d.pnl || 0).toFixed(2)}</td><td>${d.wins ?? 0}</td><td>${d.losses ?? 0}</td><td>${d.neutral ?? 0}</td><td>${wr}</td></tr>`;
        });
        bodyHtml += '</tbody></table></div></div>';

        const byReason = bot.by_close_reason || {};
        bodyHtml += `<div class="analytics-section" data-category="reasons"><h3>Причины закрытия</h3>`;
        if (Object.keys(byReason).length) {
            bodyHtml += '<div class="analytics-stats-table-wrap"><table class="analytics-stats-table"><thead><tr><th>Причина</th><th>Сделок</th><th>PnL USDT</th><th>Победы</th><th>Убытки</th><th>Нулевые</th><th>Win Rate %</th></tr></thead><tbody>';
            for (const [reason, d] of Object.entries(byReason)) {
                const wr = (d.count && d.wins != null) ? ((d.wins / d.count) * 100).toFixed(1) : '—';
                const pnlClass = (d.pnl || 0) >= 0 ? 'pnl-win' : 'pnl-loss';
                bodyHtml += `<tr><td>${reason}</td><td>${d.count ?? 0}</td><td class="${pnlClass}">${(d.pnl || 0).toFixed(2)}</td><td>${d.wins ?? 0}</td><td>${d.losses ?? 0}</td><td>${d.neutral ?? 0}</td><td>${wr}</td></tr>`;
            }
            bodyHtml += '</tbody></table></div>';
        } else {
            bodyHtml += '<p>Нет данных</p>';
        }
        bodyHtml += '</div>';

        const uc = bot.unsuccessful_coins || [];
        bodyHtml += `<div class="analytics-section" data-category="unsuccessful_coins"><h3>${categories[7].label}</h3><p>(PnL &lt; 0 или Win Rate &lt; 45%, мин. 3 сделки)</p>`;
        if (uc.length) {
            bodyHtml += '<ul>';
            uc.forEach(c => {
                bodyHtml += `<li><strong>${c.symbol}</strong>: сделок ${c.trades_count}, PnL ${c.pnl_usdt} USDT, Win Rate ${c.win_rate_pct}%, причины: ${(c.reasons || []).join(', ')}</li>`;
            });
            bodyHtml += '</ul>';
        } else {
            bodyHtml += '<p>Нет неудачных монет по критериям</p>';
        }
        bodyHtml += '</div>';

        const us = bot.unsuccessful_settings || [];
        bodyHtml += `<div class="analytics-section" data-category="unsuccessful_settings"><h3>${categories[8].label}</h3>`;
        if (us.length) {
            us.forEach(u => {
                if (!u.bad_rsi_ranges?.length && !u.bad_trends?.length) return;
                bodyHtml += `<p><strong>${u.symbol}</strong></p><ul>`;
                (u.bad_rsi_ranges || []).forEach(r => {
                    bodyHtml += `<li>RSI ${r.rsi_range}: сделок ${r.trades_count}, PnL ${r.pnl_usdt}, Win Rate ${r.win_rate_pct}%</li>`;
                });
                (u.bad_trends || []).forEach(t => {
                    bodyHtml += `<li>Тренд ${t.trend}: сделок ${t.trades_count}, PnL ${t.pnl_usdt}, Win Rate ${t.win_rate_pct}%</li>`;
                });
                bodyHtml += '</ul>';
            });
        } else {
            bodyHtml += '<p>Нет данных</p>';
        }
        bodyHtml += '</div>';

        const sc = bot.successful_coins || [];
        bodyHtml += `<div class="analytics-section" data-category="successful_coins"><h3>${categories[9].label}</h3><p>(PnL &gt; 0 и Win Rate ≥ 55%, мин. 3 сделки)</p>`;
        if (sc.length) {
            bodyHtml += '<ul>';
            sc.forEach(c => {
                bodyHtml += `<li><strong>${c.symbol}</strong>: сделок ${c.trades_count}, PnL ${c.pnl_usdt} USDT, Win Rate ${c.win_rate_pct}%</li>`;
            });
            bodyHtml += '</ul>';
        } else {
            bodyHtml += '<p>Нет удачных монет по критериям</p>';
        }
        bodyHtml += '</div>';

        const ss = bot.successful_settings || [];
        bodyHtml += `<div class="analytics-section" data-category="successful_settings"><h3>${categories[10].label}</h3><p>(Диапазоны RSI и тренды с Win Rate ≥ 55% и PnL &gt; 0)</p>`;
        if (ss.length) {
            ss.forEach(u => {
                if (!u.good_rsi_ranges?.length && !u.good_trends?.length) return;
                bodyHtml += `<p><strong>${u.symbol}</strong></p><ul>`;
                (u.good_rsi_ranges || []).forEach(r => {
                    bodyHtml += `<li>RSI ${r.rsi_range}: сделок ${r.trades_count}, PnL ${r.pnl_usdt}, Win Rate ${r.win_rate_pct}%</li>`;
                });
                (u.good_trends || []).forEach(t => {
                    bodyHtml += `<li>Тренд ${t.trend}: сделок ${t.trades_count}, PnL ${t.pnl_usdt}, Win Rate ${t.win_rate_pct}%</li>`;
                });
                bodyHtml += '</ul>';
            });
        } else {
            bodyHtml += '<p>Нет данных</p>';
        }
        bodyHtml += '</div>';

        bodyHtml += `<div class="analytics-meta">Отчёт сформирован: ${report.generated_at || '—'}</div></div>`;

        container.innerHTML = tabsHtml + '<div class="analytics-report-wrap">' + bodyHtml + '</div>';
        container.querySelectorAll('.analytics-cat-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const cat = btn.dataset.category;
                container.querySelectorAll('.analytics-cat-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                container.querySelectorAll('.analytics-section').forEach(sec => {
                    sec.classList.toggle('active', sec.dataset.category === cat);
                });
            });
        });
        container.querySelectorAll('.analytics-section').forEach(sec => {
            sec.classList.toggle('active', sec.dataset.category === 'summary');
        });
    }

    /**
     * Инициализирует фильтры истории
     */,
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
                const targetContent = document.getElementById(`${tabName}History`);
                if (targetContent) {
                    targetContent.classList.add('active');
                }
                
                // Загружаем данные для выбранной вкладки
                this.currentHistoryTab = tabName;
                this.loadHistoryData(tabName);
            });

            button.setAttribute('data-listener-bound', 'true');
        });
    }

    /**
     * Инициализирует кнопки действий истории
     */,
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
            
            // Загружаем данные в зависимости от вкладки
            switch (targetTab) {
                case 'actions':
                    await this.loadBotActions(filters);
                    break;
                case 'trades':
                    await this.loadBotTrades(filters);
                    break;
                case 'signals':
                    await this.loadBotSignals(filters);
                    break;
                case 'ai':
                    await this.loadAIHistory();
                    break;
            }
            
            // Загружаем статистику (если не AI вкладка)
            if (targetTab !== 'ai') {
                await this.loadHistoryStatistics(filters);
            }
            
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки данных истории:', error);
            this.showNotification(`Ошибка загрузки истории: ${error.message}`, 'error');
        }
    }

    /**
     * Получает параметры фильтров
     */,
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
            
            if (data.success) {
                const aiStats = data.ai || {};
                const scriptStats = data.script || {};
                const comparisonStats = data.comparison || {};
                
                // Сохраняем данные AI для использования в метриках производительности
                this._lastAIStats = aiStats;
                
                // Обновляем UI
                const aiTotalEl = document.getElementById('aiTotalDecisions');
                const aiWinRateEl = document.getElementById('aiWinRate');
                const scriptTotalEl = document.getElementById('scriptTotalDecisions');
                const scriptWinRateEl = document.getElementById('scriptWinRate');
                const comparisonWinRateEl = document.getElementById('comparisonWinRate');
                const comparisonAvgPnlEl = document.getElementById('comparisonAvgPnl');
                const comparisonSummaryEl = document.getElementById('aiComparisonSummary');
                
                const aiTotal = Number(aiStats.total) || 0;
                const aiWinRate = typeof aiStats.win_rate === 'number' ? aiStats.win_rate : 0;
                const aiTotalPnL = Number(aiStats.total_pnl) || 0;
                const aiAvgPnL = Number(aiStats.avg_pnl) || 0;
                const scriptTotal = Number(scriptStats.total) || 0;
                const scriptWinRate = typeof scriptStats.win_rate === 'number' ? scriptStats.win_rate : 0;
                const scriptTotalPnL = Number(scriptStats.total_pnl) || 0;
                const scriptAvgPnL = Number(scriptStats.avg_pnl) || 0;
                
                // Обновляем карточку AI
                if (aiTotalEl) {
                    aiTotalEl.textContent = aiTotal;
                    const aiCard = aiTotalEl.closest('.stat-card');
                    if (aiCard) {
                        aiCard.classList.remove('profit', 'loss', 'neutral');
                        if (aiTotal > 0) {
                            aiCard.classList.add(aiWinRate >= 50 ? 'profit' : 'loss');
                        }
                    }
                }
                if (aiWinRateEl) {
                    aiWinRateEl.innerHTML = `Win Rate: <strong>${aiWinRate.toFixed(1)}%</strong>`;
                    if (aiTotalPnL !== 0) {
                        aiWinRateEl.innerHTML += `<br>Total PnL: <strong class="${aiTotalPnL >= 0 ? 'profit' : 'loss'}">${aiTotalPnL >= 0 ? '+' : ''}${aiTotalPnL.toFixed(2)} USDT</strong>`;
                    }
                }
                
                // Обновляем карточку Скриптовые
                if (scriptTotalEl) {
                    scriptTotalEl.textContent = scriptTotal;
                    const scriptCard = scriptTotalEl.closest('.stat-card');
                    if (scriptCard) {
                        scriptCard.classList.remove('profit', 'loss', 'neutral');
                        if (scriptTotal > 0) {
                            scriptCard.classList.add(scriptWinRate >= 50 ? 'profit' : 'loss');
                        }
                    }
                }
                if (scriptWinRateEl) {
                    scriptWinRateEl.innerHTML = `Win Rate: <strong>${scriptWinRate.toFixed(1)}%</strong>`;
                    if (scriptTotalPnL !== 0) {
                        scriptWinRateEl.innerHTML += `<br>Total PnL: <strong class="${scriptTotalPnL >= 0 ? 'profit' : 'loss'}">${scriptTotalPnL >= 0 ? '+' : ''}${scriptTotalPnL.toFixed(2)} USDT</strong>`;
                    }
                }
                
                const winRateDiff = Number(comparisonStats.win_rate_diff) || 0;
                const avgPnlDiff = Number(comparisonStats.avg_pnl_diff) || 0;
                const totalPnlDiff = Number(comparisonStats.total_pnl_diff) || 0;
                
                // Обновляем карточку Сравнение
                if (comparisonWinRateEl) {
                    const diffIcon = winRateDiff > 0 ? '📈' : winRateDiff < 0 ? '📉' : '➖';
                    comparisonWinRateEl.innerHTML = `${diffIcon} ${winRateDiff >= 0 ? '+' : ''}${winRateDiff.toFixed(1)}%`;
                    comparisonWinRateEl.className = `stat-value ${winRateDiff >= 0 ? 'profit' : winRateDiff < 0 ? 'loss' : 'neutral'}`;
                    
                    const comparisonCard = comparisonWinRateEl.closest('.stat-card');
                    if (comparisonCard) {
                        comparisonCard.classList.remove('profit', 'loss', 'neutral');
                        if (winRateDiff > 0) {
                            comparisonCard.classList.add('profit');
                        } else if (winRateDiff < 0) {
                            comparisonCard.classList.add('loss');
                        } else {
                            comparisonCard.classList.add('neutral');
                        }
                    }
                }
                
                if (comparisonAvgPnlEl) {
                    comparisonAvgPnlEl.innerHTML = `Avg PnL: <strong class="${avgPnlDiff >= 0 ? 'profit' : 'loss'}">${avgPnlDiff >= 0 ? '+' : ''}${avgPnlDiff.toFixed(2)} USDT</strong>`;
                    if (totalPnlDiff !== 0) {
                        comparisonAvgPnlEl.innerHTML += `<br>Total PnL: <strong class="${totalPnlDiff >= 0 ? 'profit' : 'loss'}">${totalPnlDiff >= 0 ? '+' : ''}${totalPnlDiff.toFixed(2)} USDT</strong>`;
                    }
                }

                if (comparisonSummaryEl) {
                    comparisonSummaryEl.textContent = this.buildAIComparisonSummary(aiStats, scriptStats, comparisonStats);
                    comparisonSummaryEl.classList.toggle('profit', winRateDiff > 0);
                    comparisonSummaryEl.classList.toggle('loss', winRateDiff < 0);
                }
            }
        } catch (error) {
            console.error('[BotsManager] ❌ Ошибка загрузки статистики AI:', error);
            const summaryEl = document.getElementById('aiComparisonSummary');
            if (summaryEl) {
                summaryEl.textContent = 'Недостаточно данных для сравнения';
                summaryEl.classList.remove('profit', 'loss');
            }
        }
    }

    /**
     * Навешивает обработчик на селектор периода
     */
    });
})();
