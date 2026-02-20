/**
 * BotsManager - 02_search
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            initializeSearch() {
        const searchInput = document.getElementById('coinSearchInput');
        const clearSearchBtn = document.getElementById('clearSearchBtn');
        
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const searchTerm = e.target.value;
                
                // ✅ DEBOUNCE: Отменяем предыдущий таймер
                if (this.searchDebounceTimer) {
                    clearTimeout(this.searchDebounceTimer);
                }
                
                // ✅ Сразу обновляем кнопку очистки (без задержки)
                this.updateClearButtonVisibility(searchTerm);
                
                // ✅ Фильтрацию делаем с задержкой 150ms
                this.searchDebounceTimer = setTimeout(() => {
                    this.filterCoins(searchTerm);
                    this.updateSmartFilterControls(searchTerm);
                }, 150);
            });
        }
        
        if (clearSearchBtn) {
            clearSearchBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                // ✅ Отменяем любые pending фильтрации
                if (this.searchDebounceTimer) {
                    clearTimeout(this.searchDebounceTimer);
                }
                
                this.clearSearch();
            });
        }
    },
            updateClearButtonVisibility(searchTerm) {
        const clearSearchBtn = document.getElementById('clearSearchBtn');
        if (clearSearchBtn) {
            clearSearchBtn.style.display = searchTerm && searchTerm.length > 0 ? 'flex' : 'none';
        }
    },
            clearSearch() {
        console.log('[BotsManager] 🧹 Очистка поиска...');
        const searchInput = document.getElementById('coinSearchInput');
        if (searchInput) {
            // ✅ Очищаем поле
            searchInput.value = '';
            
            // ✅ Применяем пустой фильтр
            this.filterCoins('');
            this.updateSmartFilterControls('');
            this.updateClearButtonVisibility('');
            
            // ✅ Возвращаем фокус
            searchInput.focus();
            
            console.log('[BotsManager] ✅ Поиск очищен');
        }
    }
    });
})();
