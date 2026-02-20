/**
 * BotsManager - 16_timeframe
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            async loadTimeframe() {
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/timeframe`);
            const data = await response.json();,
            async applyTimeframe() {
        const timeframeSelect = document.getElementById('systemTimeframe');
        const applyBtn = document.getElementById('applyTimeframeBtn');
        const statusDiv = document.getElementById('timeframeStatus');,
            updateTimeframeInUI(timeframe) {
        // Обновляем отображение текущего таймфрейма в заголовке списка монет
        const timeframeDisplay = document.getElementById('currentTimeframeDisplay');,
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
    });
})();
