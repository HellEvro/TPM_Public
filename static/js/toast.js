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
        // Создаем контейнер для toast уведомлений
        this.container = document.createElement('div');
        this.container.className = 'toast-container';
        
        // Проверяем, что document.body существует
        if (document.body) {
            document.body.appendChild(this.container);
        } else {
            // Если body еще не готов, ждем DOMContentLoaded
            document.addEventListener('DOMContentLoaded', () => {
                if (document.body) {
                    document.body.appendChild(this.container);
                }
            });
        }
    }

    show(message, type = 'info', duration = 5000) {
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

        // Анимация появления
        setTimeout(() => {
            toast.classList.add('show');
        }, 10);

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

// Совместимость с старым API
window.notifications = {
    show: (message, type) => window.toastManager.show(message, type)
};
