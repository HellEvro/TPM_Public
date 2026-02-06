// UI утилиты
const UIUtils = {
    toggleMenu() {
        Logger.debug('UI', 'Toggling menu');
        const dropdown = document.getElementById('menuDropdown');
        if (dropdown) {
            dropdown.classList.toggle('active');
        }
    },

    toggleTheme() {
        Logger.debug('UI', 'Toggling theme');
        const currentTheme = stateManager.getState('app.theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        stateManager.setState('app.theme', newTheme);
        document.body.setAttribute('data-theme', newTheme);
        
        // Уведомляем о смене темы через EventBus
        eventBus.emit('theme:changed', newTheme);
    },

    async updateInterface() {
        Logger.debug('UI', 'Updating interface');
        try {
            // Получаем все элементы с переводами
            const elements = document.querySelectorAll('[data-translate]');
            
            // Обновляем переводы
            elements.forEach(element => {
                const key = element.getAttribute('data-translate');
                if (typeof languageUtils !== 'undefined' && languageUtils.translate) {
                    const translation = languageUtils.translate(key);
                    if (translation) {
                        element.textContent = translation;
                    }
                }
            });

            // Обновляем плейсхолдеры
            const inputs = document.querySelectorAll('input[data-translate-placeholder]');
            inputs.forEach(input => {
                const key = input.getAttribute('data-translate-placeholder');
                if (typeof languageUtils !== 'undefined' && languageUtils.translate) {
                    const translation = languageUtils.translate(key);
                    if (translation) {
                        input.placeholder = translation;
                    }
                }
            });

            Logger.debug('UI', 'Interface updated successfully');
        } catch (error) {
            Logger.error('UI', 'Error updating interface:', error);
            throw error;
        }
    },

    // Добавим метод для безопасного вызова функций
    safeCall(fn, ...args) {
        try {
            return fn.apply(this, args);
        } catch (error) {
            Logger.error('UI', `Error calling function ${fn.name}:`, error);
            NotificationManager.error('Error in UI operation');
            return null;
        }
    }
};

// Экспортируем функции в глобальную область с безопасным вызовом
window.toggleMenu = () => UIUtils.safeCall(UIUtils.toggleMenu);
window.toggleTheme = () => UIUtils.safeCall(UIUtils.toggleTheme);
// window.toggleLanguage определена в language.js