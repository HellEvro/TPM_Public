// Добавим флаг для блокировки обновления меню
window.isLanguageChanging = false;

async function toggleLanguage() {
    try {
        console.log('🌐 toggleLanguage called');
        
        // Сохраняем текущий текст меню
        const savedMenuText = localStorage.getItem('currentMenuText');
        
        // Меняем язык
        const currentLang = document.documentElement.lang;
        console.log('🌐 Current language:', currentLang);
        const newLang = currentLang === 'ru' ? 'en' : 'ru';
        console.log('🌐 Switching to:', newLang);
        document.documentElement.lang = newLang;
        
        // Сохраняем новый язык на сервере
        console.log('🌐 Saving language to server...');
        const response = await fetch('/api/set_language', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: newLang })
        });
        console.log('🌐 Server response:', response.status);
        
        // Обновляем интерфейс
        console.log('🌐 Updating interface...');
        updateInterface();
        console.log('🌐 Language toggle completed');
        
        // Восстанавливаем текст меню
        if (savedMenuText) {
            const currentPage = document.getElementById('currentPage');
            if (currentPage) {
                currentPage.textContent = savedMenuText;
            }
        }
        
    } catch (error) {
        console.error('Error toggling language:', error);
    }
}

function updateInterface() {
    console.log('updateInterface called');
    
    if (typeof TRANSLATIONS === 'undefined') {
        console.error('TRANSLATIONS not found');
        return;
    }
    
    const currentLang = document.documentElement.lang || 'ru';
    console.log('Current language:', currentLang);
    console.log('TRANSLATIONS.ru keys:', Object.keys(TRANSLATIONS.ru).length);
    console.log('TRANSLATIONS.en keys:', Object.keys(TRANSLATIONS.en).length);
    
    // Обновляем все элементы с data-translate
    let translated = 0, missing = 0;
    document.querySelectorAll('[data-translate]').forEach(element => {
        // Пропускаем элемент с id="currentPage"
        if (element.id === 'currentPage') return;
        
        // ✅ КРИТИЧНО: Пропускаем заголовок с таймфреймом - он обновляется отдельно
        const timeframeDisplay = element.querySelector('#currentTimeframeDisplay');
        if (timeframeDisplay) {
            // Не перезаписываем элемент с таймфреймом - он обновляется через updateTimeframeInUI()
            return;
        }
        
        const key = element.getAttribute('data-translate');
        if (key && TRANSLATIONS[currentLang] && TRANSLATIONS[currentLang][key]) {
            element.textContent = TRANSLATIONS[currentLang][key];
            translated++;
        } else if (key) {
            console.warn(`Missing translation for key: ${key} (lang: ${currentLang})`);
            missing++;
        }
    });
    console.log(`Translated: ${translated}, Missing: ${missing}`);
    
    // Обновляем все элементы с data-translate-placeholder
    document.querySelectorAll('[data-translate-placeholder]').forEach(element => {
        const key = element.getAttribute('data-translate-placeholder');
        if (key && TRANSLATIONS[currentLang] && TRANSLATIONS[currentLang][key]) {
            element.placeholder = TRANSLATIONS[currentLang][key];
        }
    });
    
    // Обновляем placeholder для поиска (для обратной совместимости)
    const searchInput = document.getElementById('tickerSearch');
    if (searchInput && TRANSLATIONS[currentLang]) {
        searchInput.placeholder = TRANSLATIONS[currentLang]['searchPlaceholder'] || 'Поиск тикера...';
    }
    
    // Принудительно обновляем данные аккаунта в BotsManager, если он существует
    if (typeof window.botsManager !== 'undefined' && window.botsManager.loadAccountInfo) {
        window.botsManager.loadAccountInfo();
    }
    
    // ✅ КРИТИЧНО: Обновляем таймфрейм в UI после перевода, чтобы заголовок был правильным
    if (typeof window.botsManager !== 'undefined' && window.botsManager.currentTimeframe) {
        window.botsManager.updateTimeframeInUI(window.botsManager.currentTimeframe);
    } else if (typeof window.botsManager !== 'undefined' && window.botsManager.loadTimeframe) {
        // Если таймфрейм еще не загружен, загружаем его
        window.botsManager.loadTimeframe().then(timeframe => {
            if (timeframe) {
                window.botsManager.updateTimeframeInUI(timeframe);
            }
        });
    }
    
    console.log('Interface updated successfully');
}

// Делаем функции глобально доступными
window.toggleLanguage = toggleLanguage;
window.updateInterface = updateInterface; 