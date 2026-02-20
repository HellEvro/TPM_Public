/**
 * BotsManager - 07_bot_controls
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            async createBot(manualDirection = null) {
        console.log('[BotsManager] 🚀 Запуск создания бота...');,
            async startBot(symbol) {
        const targetSymbol = symbol || this.selectedCoin?.symbol;,
            async stopBot(symbol) {
        const targetSymbol = symbol || this.selectedCoin?.symbol;,
            async pauseBot(symbol) {
        const targetSymbol = symbol || this.selectedCoin?.symbol;,
            async resumeBot(symbol) {
        const targetSymbol = symbol || this.selectedCoin?.symbol;,
            async deleteBot(symbol) {
        const targetSymbol = symbol || this.selectedCoin?.symbol;,
            async quickLaunchBot(direction) {
        if (!this.selectedCoin) return;
        
        try {
            console.log(`[BotsManager] 🚀 Быстрый запуск ${direction} бота для ${this.selectedCoin.symbol}`);
            await this.createBot(direction);
        } catch (error) {
            console.error(`[BotsManager] ❌ Ошибка быстрого запуска ${direction} бота:`, error);
            this.showNotification('❌ Ошибка соединения при создании бота', 'error');
        }
    },
            updateBotStatusInUI(symbol, status) {
        const botCard = document.querySelector(`[data-symbol="${symbol}"]`);
        if (!botCard) return;

        const statusElement = botCard.querySelector('.bot-status');
        const startButton = botCard.querySelector('.start-bot-btn');
        const stopButton = botCard.querySelector('.stop-bot-btn');
        const deleteButton = botCard.querySelector('.delete-bot-btn');,
            removeBotFromUI(symbol) {
        const botCard = document.querySelector(`[data-symbol="${symbol}"]`);,
            getBotStopButtonHtml(bot) {
        const isRunning = bot.status === 'running' || bot.status === 'idle' || 
                         bot.status === 'in_position_long' || bot.status === 'in_position_short';
        const isStopped = bot.status === 'stopped' || bot.status === 'paused';,
            getBotDeleteButtonHtml(bot) {
        return `<span onclick="event.stopPropagation(); window.app.botsManager.deleteBot('${bot.symbol}')" title="${window.languageUtils.translate('delete_btn')}" class="bot-icon-btn bot-icon-delete">🗑</span>`;
    },
            getBotControlButtonsHtml(bot) {
        return (this.getBotStopButtonHtml(bot) || '') + this.getBotDeleteButtonHtml(bot);
    },
            getBotDetailButtonsHtml(bot) {
        // Бот активен если running, idle, или в позиции
        const isRunning = bot.status === 'running' || bot.status === 'idle' || 
                         bot.status === 'in_position_long' || bot.status === 'in_position_short';
        const isStopped = bot.status === 'stopped' || bot.status === 'paused';
        
        let buttons = [];,
            updateBotStatus(status) {
        const statusText = document.getElementById('botStatusText');
        const statusIndicator = document.getElementById('botStatusIndicator');
        
        // Проверяем есть ли бот для выбранной монеты
        const selectedBot = this.selectedCoin && this.activeBots ? 
                           this.activeBots.find(bot => bot.symbol === this.selectedCoin.symbol) : null;,
            updateBotControlButtons() {
        console.log(`[BotsManager] 🎮 Обновление кнопок управления...`);
        
        const createBtn = document.getElementById('createBotBtn');
        const startBtn = document.getElementById('startBotBtn');
        const stopBtn = document.getElementById('stopBotBtn');
        const pauseBtn = document.getElementById('pauseBotBtn');
        const resumeBtn = document.getElementById('resumeBotBtn');
        
        // Кнопки быстрого запуска
        const quickStartLongBtn = document.getElementById('quickStartLongBtn');
        const quickStartShortBtn = document.getElementById('quickStartShortBtn');
        const quickStopBtn = document.getElementById('quickStopBtn');
        
        // Кнопки ручного запуска в секции настроек
        const manualLaunchLongBtn = document.getElementById('manualLaunchLongBtn');
        const manualLaunchShortBtn = document.getElementById('manualLaunchShortBtn');
        
        // Проверяем есть ли бот для выбранной монеты
        const selectedBot = this.selectedCoin && this.activeBots ? 
                           this.activeBots.find(bot => bot.symbol === this.selectedCoin.symbol) : null;
        
        // Проверяем, есть ли активная позиция
        const hasActivePosition = selectedBot && (
            selectedBot.status === 'in_position_long' || 
            selectedBot.status === 'in_position_short' ||
            selectedBot.status === 'running'
        );
        
        console.log(`[BotsManager] 🔍 Выбранная монета: ${this.selectedCoin?.symbol}`);
        console.log(`[BotsManager] 🤖 Найден бот:`, selectedBot);
        console.log(`[BotsManager] 📊 Есть активная позиция:`, hasActivePosition);,
            updateCoinsListWithBotStatus() {
        this.logDebug('[BotsManager] 💰 Обновление списка монет с пометками о ботах...');
        
        if (!this.activeBots) return;
        
        // Создаем set с символами только активных ботов (не idle/paused) для быстрого поиска
        const activeBotsSymbols = new Set(
            this.activeBots
                .filter(bot => bot.status !== 'idle' && bot.status !== 'paused')
                .map(bot => bot.symbol)
        );
        
        this.logDebug(`[BotsManager] 🤖 Найдено ${activeBotsSymbols.size} активных ботов из ${this.activeBots.length} общих`);
        
        // Обновляем отображение монет в списке
        const coinItems = document.querySelectorAll('.coin-item');
        coinItems.forEach(item => {
            const symbolElement = item.querySelector('.coin-symbol');,
            updateActiveBotsTab() {
        console.log('[BotsManager] 🚀 Обновление вкладки "Боты в работе"...');
        
        // Если мы сейчас на вкладке "Боты в работе", обновляем данные
        const activeTab = document.querySelector('.tab-btn.active');,
                renderActiveBotsDetails() {
        this.logDebug('[BotsManager] 🎨 Отрисовка деталей активных ботов...');
        
        // Обновляем вкладку "Боты в работе"
        const detailsElement = document.getElementById('activeBotsDetailsList');
        
        // Обновляем правую панель на вкладке "Управление"
        const scrollListElement = document.getElementById('activeBotsScrollList');
        const emptyStateElement = document.getElementById('emptyActiveBotsState');
        
        this.logDebug(`[BotsManager] 📊 Количество активных ботов: ${this.activeBots ? this.activeBots.length : 0}`);
        this.logDebug(`[BotsManager] 🔍 Элементы найдены:`, {
            detailsElement: !!detailsElement,
            scrollListElement: !!scrollListElement,
            emptyStateElement: !!emptyStateElement
        });

        const hasActiveBots = this.activeBots && this.activeBots.length > 0;
        
        // Обновляем счётчики фильтров вкладки "Боты в работе"
        this.updateActiveBotsFilterCounts();
        
        // Проверяем, нужно ли полностью перерисовывать HTML
        const existingBots = scrollListElement ? Array.from(scrollListElement.querySelectorAll('.active-bot-item')).map(item => item.dataset.symbol) : [];
        const currentBots = hasActiveBots ? this.activeBots.map(bot => bot.symbol) : [];
        const needsFullRedraw = JSON.stringify(existingBots.sort()) !== JSON.stringify(currentBots.sort());
        const filteredBots = this.getFilteredActiveBotsForDetails();
        const virtualAsBots = this.getVirtualPositionsAsBots();
        const displayListForDetails = filteredBots.concat(virtualAsBots);
        const detailsKey = (b) => b.is_virtual ? `${b.symbol}_v${b._virtualIndex}` : b.symbol;
        const existingDetailsBots = detailsElement ? Array.from(detailsElement.querySelectorAll('.active-bot-item')).map(i => (i.dataset.isVirtual === 'true' ? `${i.dataset.symbol}_v${i.dataset.virtualIndex || 0}` : i.dataset.symbol)).sort() : [];
        const displayKeys = displayListForDetails.map(detailsKey).sort();
        const needsDetailsRedraw = needsFullRedraw || (this.activeBotsFilter !== this._lastActiveBotsFilter) ||
            JSON.stringify(displayKeys) !== JSON.stringify(existingDetailsBots);
        
        console.log(`[DEBUG] Проверка перерисовки:`, { existingBots, currentBots, needsFullRedraw, needsDetailsRedraw });

        // Обновляем правую панель (вкладка "Управление"),
            async loadActiveBotsData() {
        this.logDebug('[BotsManager] 🤖 Загрузка данных активных ботов...');
        
        if (!this.serviceOnline) return;
        
        try {
            // ⚡ УБРАНО: Синхронизация позиций теперь выполняется только автоматически воркерами
            // Вызов sync-positions здесь вызывал race condition с остановкой бота
            // и перезаписывал статус PAUSED обратно на in_position_long/short
            
            // Загружаем и ботов, и конфигурацию автобота параллельно
            const [botsResponse, configResponse] = await Promise.all([
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/list`),
                fetch(`${this.BOTS_SERVICE_URL}/api/bots/auto-bot`)
            ]);
            
            const botsData = await botsResponse.json();
            const configData = await configResponse.json();,
            async updateActiveBotsDetailed() {
        if (!this.serviceOnline) return;
        
        try {
            this.logDebug('[BotsManager] 📊 Обновление детальной информации о ботах...');
            
            // Получаем детальную информацию о всех активных ботах
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/active-detailed`);
    });
})();
