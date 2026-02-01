/**
 * Toast Notification System
 * Красивые всплывающие уведомления справа снизу
 */

class ToastManager {
    constructor() {
        this.container = null;
        this.toasts = new Map();
        this.toastCounter = 0;
        this.init();
    }

    init() {
        // Если контейнер уже существует, не создаем новый
        if (this.container && document.body.contains(this.container)) {
            console.log('[ToastManager] ℹ️ Контейнер уже инициализирован');
            return;
        }
        
        // Создаем контейнер для toast уведомлений
        this.container = document.createElement('div');
        this.container.className = 'toast-container';
        this.container.id = 'toast-container';
        
        // Проверяем, что document.body существует
        if (document.body) {
            document.body.appendChild(this.container);
            console.log('[ToastManager] ✅ Контейнер добавлен в DOM');
        } else {
            // Если body еще не готов, ждем DOMContentLoaded
            console.log('[ToastManager] ⏳ Ожидание DOMContentLoaded...');
            const initContainer = () => {
                if (document.body) {
                    document.body.appendChild(this.container);
                    console.log('[ToastManager] ✅ Контейнер добавлен в DOM (после DOMContentLoaded)');
                } else {
                    console.error('[ToastManager] ❌ document.body все еще не доступен!');
                }
            };
            
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', initContainer);
            } else {
                // DOM уже загружен
                initContainer();
            }
        }
    }

    show(message, type = 'info', duration = 5000) {
        console.log(`[ToastManager] 🔔 Показ уведомления [${type}]:`, message);
        
        // ✅ Проверяем и инициализируем контейнер, если нужно
        if (!this.container) {
            console.warn('[ToastManager] ⚠️ Контейнер не инициализирован, инициализируем...');
            this.init();
        }
        
        // ✅ Проверяем, что контейнер в DOM
        if (!this.container || !document.body.contains(this.container)) {
            console.warn('[ToastManager] ⚠️ Контейнер не в DOM, добавляем...');
            if (document.body) {
                if (!this.container) {
                    this.init();
                }
                if (this.container && !document.body.contains(this.container)) {
                    document.body.appendChild(this.container);
                    console.log('[ToastManager] ✅ Контейнер добавлен в DOM');
                }
            } else {
                console.error('[ToastManager] ❌ document.body не доступен! Пропускаем уведомление.');
                return null; // ❌ НЕ используем alert - просто возвращаем null
            }
        }
        
        console.log('[ToastManager] ✅ Контейнер готов, создаем toast');
        
        const toastId = ++this.toastCounter;
        
        // Создаем элемент toast
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-icon"></div>
            <div class="toast-message">${this.escapeHtml(message)}</div>
            <div class="toast-progress" style="transition-duration: ${duration}ms"></div>
        `;

        // Добавляем toast в контейнер
        this.container.appendChild(toast);
        this.toasts.set(toastId, toast);
        
        // ✅ Принудительно устанавливаем стили контейнера
        this.container.style.position = 'fixed';
        this.container.style.top = '20px';
        this.container.style.right = '20px';
        this.container.style.zIndex = '999999';
        this.container.style.display = 'flex';
        this.container.style.flexDirection = 'column';
        this.container.style.gap = '10px';
        this.container.style.maxWidth = '400px';
        this.container.style.visibility = 'visible';
        this.container.style.opacity = '1';
        this.container.style.pointerEvents = 'none';

        // ✅ Принудительно устанавливаем стили для toast
        toast.style.display = 'block';
        toast.style.visibility = 'visible';
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(0)';
        toast.style.zIndex = '999999';
        toast.style.position = 'relative';

        // Анимация появления - сразу показываем
        requestAnimationFrame(() => {
            toast.classList.add('show');
            // ✅ Дополнительно принудительно устанавливаем стили для видимости
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(0)';
            toast.style.visibility = 'visible';
            toast.style.zIndex = '999999';
            toast.style.display = 'block';
            toast.style.position = 'relative';
            toast.style.pointerEvents = 'auto';
            
            // ✅ Проверяем, что toast действительно виден
            const rect = toast.getBoundingClientRect();
            const isVisible = rect.width > 0 && rect.height > 0 && 
                            window.getComputedStyle(toast).visibility !== 'hidden' &&
                            window.getComputedStyle(toast).display !== 'none';
            
            if (isVisible) {
                console.log('[ToastManager] ✅ Toast показан и виден:', message.substring(0, 50));
            } else {
                console.warn('[ToastManager] ⚠️ Toast создан, но не виден!', {
                    width: rect.width,
                    height: rect.height,
                    visibility: window.getComputedStyle(toast).visibility,
                    display: window.getComputedStyle(toast).display,
                    opacity: window.getComputedStyle(toast).opacity
                });
            }
        });

        // Запускаем прогресс-бар
        setTimeout(() => {
            const progress = toast.querySelector('.toast-progress');
            if (progress) {
                progress.style.transform = 'scaleX(0)';
            }
        }, 50);

        // Автозакрытие
        if (duration > 0) {
            setTimeout(() => {
                this.hide(toastId);
            }, duration);
        }

        // Клик для закрытия
        toast.addEventListener('click', () => {
            this.hide(toastId);
        });

        return toastId;
    }

    hide(toastId) {
        const toast = this.toasts.get(toastId);
        if (!toast) return;

        toast.classList.remove('show');
        toast.classList.add('hide');

        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
            this.toasts.delete(toastId);
        }, 300);
    }

    // Методы для разных типов уведомлений
    success(message, duration = 4000) {
        return this.show(message, 'success', duration);
    }

    error(message, duration = 6000) {
        return this.show(message, 'error', duration);
    }

    warning(message, duration = 5000) {
        return this.show(message, 'warning', duration);
    }

    info(message, duration = 4000) {
        return this.show(message, 'info', duration);
    }

    // Очистить все уведомления
    clear() {
        this.toasts.forEach((toast, id) => {
            this.hide(id);
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Создаем глобальный экземпляр
window.toastManager = new ToastManager();

// ✅ Автоматическая инициализация при загрузке DOM
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        console.log('[ToastManager] 🔧 Автоматическая инициализация при DOMContentLoaded');
        if (window.toastManager) {
            window.toastManager.init();
        }
    });
} else {
    // DOM уже загружен
    console.log('[ToastManager] 🔧 Автоматическая инициализация (DOM уже загружен)');
    if (window.toastManager) {
        window.toastManager.init();
    }
}

// Совместимость с старым API
window.notifications = {
    show: (message, type) => window.toastManager.show(message, type)
};

// Глобальная функция showToast для совместимости (ai_config_manager и др.)
window.showToast = function(message, type = 'info', duration = 4000) {
    if (window.toastManager) {
        if (type === 'success') window.toastManager.success(message, duration);
        else if (type === 'error') window.toastManager.error(message, duration);
        else if (type === 'warning') window.toastManager.warning(message, duration);
        else window.toastManager.info(message, duration);
    }
};

// ✅ Тестовая функция для проверки работы toast (можно вызвать из консоли: testToast())
window.testToast = function() {
    console.log('[ToastManager] 🧪 Тестирование toast уведомлений...');
    if (window.toastManager) {
        window.toastManager.init();
        window.toastManager.success('✅ Тест успешного уведомления');
        setTimeout(() => window.toastManager.info('ℹ️ Тест информационного уведомления'), 500);
        setTimeout(() => window.toastManager.warning('⚠️ Тест предупреждения'), 1000);
        setTimeout(() => window.toastManager.error('❌ Тест ошибки'), 1500);
    } else {
        console.error('[ToastManager] ❌ toastManager не найден!');
    }
};
