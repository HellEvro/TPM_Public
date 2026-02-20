/**
 * BotsManager - 03_filters
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            initializeManagementButtons() {
        // Кнопки фильтров в блоке управления
        const addToWhitelistBtnMgmt = document.getElementById('addToWhitelistBtnManagement');
        const addToBlacklistBtnMgmt = document.getElementById('addToBlacklistBtnManagement');
        const removeFromFiltersBtnMgmt = document.getElementById('removeFromFiltersBtnManagement');,
            initializeRsiFilters() {
        document.querySelectorAll('.rsi-filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                // ✅ ИСПРАВЛЕНИЕ: Используем currentTarget вместо target
                // currentTarget = сама кнопка, target = может быть вложенный элемент (эмодзи, текст)
                const clickedBtn = e.currentTarget;
                const filter = clickedBtn.dataset.filter;
                
                console.log(`[BotsManager] 🔍 Клик на фильтр: ${filter}`);
                
                // Переключаем активный фильтр
                document.querySelectorAll('.rsi-filter-btn').forEach(b => b.classList.remove('active'));
                clickedBtn.classList.add('active');
                
                // Применяем фильтр
                this.applyRsiFilter(filter);
            });
        });
        
        // Обновляем текст кнопок с текущими значениями из конфига
        this.updateRsiFilterButtons();
    },
            updateRsiFilterButtons() {
        // Обновляем кнопки фильтров с текущими значениями RSI
        const buyFilterBtn = document.querySelector('.rsi-filter-btn[data-filter="buy-zone"]');
        const sellFilterBtn = document.querySelector('.rsi-filter-btn[data-filter="sell-zone"]');,
            initActiveBotsFilters() {
        document.querySelectorAll('.active-bots-filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const clickedBtn = e.currentTarget;
                const filter = clickedBtn.dataset.filter;
                this.activeBotsFilter = filter;
                document.querySelectorAll('.active-bots-filter-btn').forEach(b => b.classList.remove('active'));
                clickedBtn.classList.add('active');
                this.renderActiveBotsDetails();
            });
        });
    },
            getFilteredActiveBotsForDetails() {
        const bots = Array.isArray(this.activeBots) ? this.activeBots : [];
        if (this.activeBotsFilter === 'all') return bots;
        return bots.filter(bot => {
            const pnl = bot.unrealized_pnl_usdt ?? bot.unrealized_pnl ?? 0;
            const pnlVal = Number.parseFloat(pnl) || 0;,
            getVirtualPositionsAsBots() {
        const list = Array.isArray(this.activeVirtualPositions) ? this.activeVirtualPositions : [];
        const filter = this.activeBotsFilter;
        return list
            .filter(v => {
                if (filter === 'long') return (v.direction || '').toUpperCase() === 'LONG';
                if (filter === 'short') return (v.direction || '').toUpperCase() === 'SHORT';
                return true;
            })
            .map((v, i) => {
                const entry = parseFloat(v.entry_price) || 0;
                const current = parseFloat(v.current_price) || 0;
                const isLong = (v.direction || '').toUpperCase() === 'LONG';
                const pnlPct = entry ? (isLong ? (current - entry) / entry : (entry - current) / entry) * 100 : 0;
                const pnlUsdt = 0; // виртуальная позиция без объёма в USDT
                return {
                    symbol: v.symbol,
                    is_virtual: true,
                    _virtualIndex: i,
                    position_side: isLong ? 'Long' : 'Short',
                    status: isLong ? 'virtual_long' : 'virtual_short',
                    entry_price: v.entry_price,
                    current_price: v.current_price,
                    unrealized_pnl_usdt: pnlUsdt,
                    unrealized_pnl: pnlPct,
                    config: {},
                    volume_value: 0,
                    position_size: 0
                };
            });
    },
            updateActiveBotsFilterCounts() {
        const bots = Array.isArray(this.activeBots) ? this.activeBots : [];
        const counts = {
            all: bots.length,
            long: bots.filter(b => b.status === 'in_position_long').length,
            short: bots.filter(b => b.status === 'in_position_short').length,
            profitable: bots.filter(b => ((b.unrealized_pnl_usdt ?? b.unrealized_pnl ?? 0) || 0) >= 0).length,
            loss: bots.filter(b => ((b.unrealized_pnl_usdt ?? b.unrealized_pnl ?? 0) || 0) < 0).length
        };
        const idMap = { all: 'All', long: 'Long', short: 'Short', profitable: 'Profitable', loss: 'Loss' };
        Object.keys(counts).forEach(key => {
            const el = document.getElementById(`activeBotsFilter${idMap[key]}Count`);
            if (el) el.textContent = counts[key];
        });
    },
            updateTrendFilterLabels() {
        // Проверяем, не обновлялись ли уже подписи,
            updateRsiThresholds(config) {
        // Обновляем внутренние пороговые значения RSI
        const oldLongThreshold = this.rsiLongThreshold;
        const oldShortThreshold = this.rsiShortThreshold;
        
        this.rsiLongThreshold = config.rsi_long_threshold || 29;
        this.rsiShortThreshold = config.rsi_short_threshold || 71;
        
        // Сбрасываем флаг обновления подписей при изменении порогов
        this.trendLabelsUpdated = false;
        
        console.log(`[BotsManager] 📊 Обновлены пороги RSI: ${oldLongThreshold}→${this.rsiLongThreshold}, ${oldShortThreshold}→${this.rsiShortThreshold}`);
        
        // Обновляем кнопки фильтров
        this.updateRsiFilterButtons();
        
        // Перепересчитываем классы для существующих монет
        this.refreshCoinsRsiClasses();
        
        // Обновляем счетчики
        this.updateCoinsCounter();
        
        // Если текущий фильтр buy-zone или sell-zone, переприменяем его,
            refreshCoinsRsiClasses() {
        // Перепересчитываем RSI классы для всех монет в списке
        const coinItems = document.querySelectorAll('.coin-item');
        
        coinItems.forEach(item => {
            const symbol = item.dataset.symbol;
            const coinData = this.coinsRsiData.find(c => c.symbol === symbol);
    });
})();
