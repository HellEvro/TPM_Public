/**
 * BotsManager - экспорт и глобальные функции
 */
// Экспортируем класс глобально сразу после определения
window.BotsManager = BotsManager;

// Глобальная функция для включения бота для текущей монеты (используется в HTML onclick)
window.enableBotForCurrentCoin = function(direction) {
    if (window.botsManager && window.botsManager.selectedCoin) {
        window.botsManager.createBot(direction || null);
    } else {
        console.error('[enableBotForCurrentCoin] BotsManager не инициализирован или монета не выбрана');
        if (window.showToast) {
            window.showToast('Выберите монету для создания бота', 'warning');
        }
    }
};

// BotsManager инициализируется в app.js, не здесь
// Version: 2025-10-21 03:47:29
