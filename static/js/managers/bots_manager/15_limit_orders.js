/**
 * BotsManager - 15_limit_orders
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            initializeLimitOrdersUI() {
        try {
            // ✅ Защита от повторной инициализации
            const toggleEl = document.getElementById('limitOrdersEntryEnabled');,
            addLimitOrderRow(percent = 0, margin = 0) {
        console.log('[BotsManager] ➕ addLimitOrderRow вызван с параметрами:', { percent, margin });
        const listEl = document.getElementById('limitOrdersList');,
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
                
                // Для лимитных ордеров (percent > 0) проверяем минимум 5 USDT,
            resetLimitOrdersToDefault() {
        try {
            // Проверяем, включен ли режим лимитных ордеров
            const toggleEl = document.getElementById('limitOrdersEntryEnabled');
    });
})();
