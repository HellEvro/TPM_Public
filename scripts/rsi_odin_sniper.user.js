// ==UserScript==
// @name         RSI Ship Sniper — Odin
// @namespace    https://robertsspaceindustries.com/
// @version      1.0.1
// @description  Быстро ловит появление кнопки Add to cart на странице pledge RSI (Odin и другие корабли)
// @author       InfoBot
// @match        https://robertsspaceindustries.com/*/pledge/*
// @match        https://robertsspaceindustries.com/pledge/*
// @run-at       document-idle
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_notification
// ==/UserScript==

(function () {
    'use strict';

    const CONFIG = {
        // Интервал полного обновления страницы (мс)
        reloadIntervalMs: 500,
        // Как часто проверять DOM без перезагрузки (мс) — быстрее ловит появление кнопки
        pollIntervalMs: 50,
        // Перейти в корзину после успешного клика
        goToCartOnSuccess: true,
        cartUrl: 'https://robertsspaceindustries.com/en/store/pledge/cart',
        // Остановить снайпер после успешного добавления
        stopAfterSuccess: true,
        // Звуковой сигнал при успехе
        playSoundOnSuccess: true,
        // Только эта страница (false = любой pledge URL из @match)
        odinOnly: true,
        odinPathPart: '/Standalone-Ships/Odin',
        // Тексты кнопки (EN + возможные варианты)
        buttonTexts: [
            'add to cart',
            'add to basket',
            'в корзину',
        ],
        // Тексты «нет в наличии» — если видны, клик не делаем
        unavailableTexts: [
            'out of stock',
            'sold out',
            'unavailable',
            'нет в наличии',
            'распродано',
        ],
    };

    const STORAGE_KEY = 'rsi_sniper_success_' + location.pathname;

    if (CONFIG.odinOnly && !location.pathname.includes(CONFIG.odinPathPart)) {
        return;
    }

    if (CONFIG.stopAfterSuccess && GM_getValue(STORAGE_KEY, false)) {
        console.info('[RSI Sniper] Уже сработал ранее. Сброс: rsiSniperReset()');
        window.rsiSniperReset = () => {
            GM_setValue(STORAGE_KEY, false);
            location.reload();
        };
        return;
    }

    let stopped = false;
    let snipeLocked = false;
    let attempts = 0;
    let lastReloadAt = 0;
    let pollTimer = null;
    let reloadTimer = null;
    let observer = null;
    let hudEl = null;

    function normalize(text) {
        return (text || '').replace(/\s+/g, ' ').trim().toLowerCase();
    }

    function isUnavailablePage() {
        const bodyText = normalize(document.body ? document.body.innerText : '');
        return CONFIG.unavailableTexts.some((t) => bodyText.includes(t));
    }

    function isVisible(el) {
        if (!el || !el.isConnected) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
            return false;
        }
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    }

    function isDisabled(el) {
        if (!el) return true;
        if (el.disabled) return true;
        if (el.getAttribute('aria-disabled') === 'true') return true;
        if (el.classList.contains('disabled')) return true;
        if (el.hasAttribute('disabled')) return true;
        const style = window.getComputedStyle(el);
        if (style.pointerEvents === 'none') return true;
        if (parseFloat(style.opacity) < 0.5) return true;
        return false;
    }

    function matchesButtonText(el) {
        const text = normalize(el.innerText || el.textContent || el.getAttribute('aria-label') || '');
        return CONFIG.buttonTexts.some((needle) => text.includes(needle));
    }

    function findAddToCartButton() {
        const candidates = [];

        document.querySelectorAll('button, a[role="button"], [role="button"]').forEach((el) => {
            if (matchesButtonText(el)) {
                candidates.push(el);
            }
        });

        // Запасной поиск по submit/input
        document.querySelectorAll('input[type="submit"], input[type="button"]').forEach((el) => {
            const val = normalize(el.value || '');
            if (CONFIG.buttonTexts.some((t) => val.includes(t))) {
                candidates.push(el);
            }
        });

        for (const el of candidates) {
            if (isVisible(el) && !isDisabled(el)) {
                return el;
            }
        }

        return null;
    }

    function forceClick(el) {
        el.scrollIntoView({ block: 'center', inline: 'center' });
        el.focus({ preventScroll: true });

        const events = ['pointerdown', 'mousedown', 'mouseup', 'click'];
        for (const type of events) {
            el.dispatchEvent(new MouseEvent(type, {
                bubbles: true,
                cancelable: true,
                view: window,
            }));
        }

        if (typeof el.click === 'function') {
            el.click();
        }
    }

    function playSuccessSound() {
        if (!CONFIG.playSoundOnSuccess) return;
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'square';
            osc.frequency.value = 880;
            gain.gain.value = 0.08;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            setTimeout(() => {
                osc.stop();
                ctx.close();
            }, 400);
        } catch (e) {
            // ignore
        }
    }

    function notifySuccess() {
        try {
            GM_notification({
                title: 'RSI Sniper',
                text: 'Odin добавлен в корзину! Быстрее оформляй checkout.',
                timeout: 10000,
            });
        } catch (e) {
            // ignore
        }
    }

    function showHud(message, level) {
        if (!hudEl) {
            hudEl = document.createElement('div');
            hudEl.id = 'rsi-sniper-hud';
            hudEl.style.cssText = [
                'position:fixed',
                'top:12px',
                'right:12px',
                'z-index:2147483647',
                'padding:10px 14px',
                'border-radius:8px',
                'font:600 13px/1.35 Segoe UI,system-ui,sans-serif',
                'color:#fff',
                'background:rgba(10,20,40,.92)',
                'border:1px solid rgba(80,160,255,.55)',
                'box-shadow:0 8px 24px rgba(0,0,0,.45)',
                'max-width:320px',
                'pointer-events:none',
            ].join(';');
            document.documentElement.appendChild(hudEl);
        }

        const colors = {
            info: '#58a6ff',
            warn: '#f0ad4e',
            ok: '#3ddc84',
            err: '#ff6b6b',
        };

        hudEl.style.borderColor = colors[level] || colors.info;
        hudEl.textContent = message;
    }

    function stopSniper() {
        if (stopped) return;
        stopped = true;
        if (pollTimer !== null) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
        if (reloadTimer !== null) {
            clearInterval(reloadTimer);
            reloadTimer = null;
        }
        if (observer) {
            observer.disconnect();
            observer = null;
        }
    }

    function markSuccess() {
        stopSniper();
        GM_setValue(STORAGE_KEY, true);
        playSuccessSound();
        notifySuccess();
        showHud('В корзину! Скрипт остановлен.', 'ok');

        if (CONFIG.goToCartOnSuccess) {
            location.replace(CONFIG.cartUrl);
        }
    }

    function trySnipe() {
        if (stopped || snipeLocked) return false;

        attempts += 1;
        const btn = findAddToCartButton();

        if (btn) {
            snipeLocked = true;
            forceClick(btn);
            markSuccess();
            return true;
        }

        const waiting = isUnavailablePage();
        showHud(
            waiting
                ? `Ожидание… попытка #${attempts} (Out of Stock)`
                : `Поиск кнопки… попытка #${attempts}`,
            waiting ? 'warn' : 'info'
        );
        return false;
    }

    function maybeReload() {
        if (stopped) return;
        const now = Date.now();
        if (now - lastReloadAt < CONFIG.reloadIntervalMs) return;
        lastReloadAt = now;
        location.reload();
    }

    // Быстрый опрос DOM — ловит кнопку сразу после рендера SPA
    pollTimer = setInterval(() => {
        if (stopped) return;
        trySnipe();
    }, CONFIG.pollIntervalMs);

    // MutationObserver — реагирует на изменения React/Vue без ожидания reload
    observer = new MutationObserver(() => {
        if (!stopped) trySnipe();
    });

    if (document.body) {
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['disabled', 'aria-disabled', 'class'],
        });
    }

    // Полное обновление страницы каждые N мс
    reloadTimer = setInterval(() => {
        if (stopped) return;
        maybeReload();
    }, Math.max(100, CONFIG.reloadIntervalMs));

    // Первая попытка сразу
    trySnipe();

    window.rsiSniperStop = () => {
        stopSniper();
        showHud('Снайпер остановлен вручную', 'info');
    };

    window.rsiSniperReset = () => {
        GM_setValue(STORAGE_KEY, false);
        location.reload();
    };

    console.info('[RSI Sniper] Активен. rsiSniperStop() — стоп, rsiSniperReset() — сброс.');
})();
