// ==UserScript==
// @name         RSI Ship Sniper — Odin
// @namespace    https://robertsspaceindustries.com/
// @version      1.1.0
// @description  Быстро ловит Add to cart только на странице Odin; останавливается при уходе в корзину
// @author       InfoBot
// @match        https://robertsspaceindustries.com/*/pledge/Standalone-Ships/Odin*
// @match        https://robertsspaceindustries.com/pledge/Standalone-Ships/Odin*
// @exclude      https://robertsspaceindustries.com/*/store/*
// @exclude      https://robertsspaceindustries.com/store/*
// @run-at       document-idle
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_notification
// ==/UserScript==

(function () {
    'use strict';

    const CONFIG = {
        // === Страница покупки (скрипт работает ТОЛЬКО здесь) ===
        targetPathPart: '/Standalone-Ships/Odin',

        reloadIntervalMs: 500,
        pollIntervalMs: 50,
        urlWatchIntervalMs: 100,

        goToCartOnSuccess: true,
        cartUrl: 'https://robertsspaceindustries.com/en/store/pledge/cart',

        stopAfterSuccess: true,
        playSoundOnSuccess: true,

        buttonTexts: [
            'add to cart',
            'add to basket',
            'в корзину',
        ],
        unavailableTexts: [
            'out of stock',
            'sold out',
            'unavailable',
            'нет в наличии',
            'распродано',
        ],
    };

    const STORAGE_SUCCESS = 'rsi_sniper_success_' + CONFIG.targetPathPart;
    const STORAGE_ACTIVE = 'rsi_sniper_active_' + CONFIG.targetPathPart;

    function isTargetPage(loc = location) {
        return loc.pathname.includes(CONFIG.targetPathPart);
    }

    function isCartOrCheckoutPage(loc = location) {
        const path = loc.pathname.toLowerCase();
        return path.includes('/cart') || path.includes('/checkout');
    }

    // Не целевая страница — ничего не делаем (на всякий случай при широком @match)
    if (!isTargetPage()) {
        GM_setValue(STORAGE_ACTIVE, false);
        return;
    }

    if (CONFIG.stopAfterSuccess && GM_getValue(STORAGE_SUCCESS, false)) {
        console.info('[RSI Sniper] Уже сработал. Сброс: rsiSniperReset()');
        window.rsiSniperReset = () => {
            GM_setValue(STORAGE_SUCCESS, false);
            GM_setValue(STORAGE_ACTIVE, false);
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
    let urlWatchTimer = null;
    let observer = null;
    let hudEl = null;
    let watchedUrl = location.pathname + location.search + location.hash;

    GM_setValue(STORAGE_ACTIVE, true);

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
            if (matchesButtonText(el)) candidates.push(el);
        });

        document.querySelectorAll('input[type="submit"], input[type="button"]').forEach((el) => {
            const val = normalize(el.value || '');
            if (CONFIG.buttonTexts.some((t) => val.includes(t))) candidates.push(el);
        });

        for (const el of candidates) {
            if (isVisible(el) && !isDisabled(el)) return el;
        }

        return null;
    }

    function forceClick(el) {
        el.scrollIntoView({ block: 'center', inline: 'center' });
        el.focus({ preventScroll: true });

        for (const type of ['pointerdown', 'mousedown', 'mouseup', 'click']) {
            el.dispatchEvent(new MouseEvent(type, {
                bubbles: true,
                cancelable: true,
                view: window,
            }));
        }

        if (typeof el.click === 'function') el.click();
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
                text: 'Odin в корзине. Оформляй checkout!',
                timeout: 10000,
            });
        } catch (e) {
            // ignore
        }
    }

    function removeHud() {
        if (hudEl && hudEl.parentNode) {
            hudEl.parentNode.removeChild(hudEl);
        }
        hudEl = null;
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

        const colors = { info: '#58a6ff', warn: '#f0ad4e', ok: '#3ddc84', err: '#ff6b6b' };
        hudEl.style.borderColor = colors[level] || colors.info;
        hudEl.textContent = message;
    }

    function stopSniper(reason) {
        if (stopped) return;
        stopped = true;
        snipeLocked = true;

        GM_setValue(STORAGE_ACTIVE, false);

        if (pollTimer !== null) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
        if (reloadTimer !== null) {
            clearInterval(reloadTimer);
            reloadTimer = null;
        }
        if (urlWatchTimer !== null) {
            clearInterval(urlWatchTimer);
            urlWatchTimer = null;
        }
        if (observer) {
            observer.disconnect();
            observer = null;
        }

        if (reason) {
            console.info('[RSI Sniper] Остановлен:', reason);
        }
    }

    /** Ушли с Odin или открыли корзину — больше не мешаем */
    function stopIfLeftTargetPage() {
        if (stopped) return true;

        const currentUrl = location.pathname + location.search + location.hash;

        if (!isTargetPage()) {
            const wentToCart = isCartOrCheckoutPage();
            stopSniper(wentToCart ? 'переход в корзину/checkout' : 'уход со страницы Odin');
            removeHud();

            if (wentToCart && CONFIG.stopAfterSuccess) {
                GM_setValue(STORAGE_SUCCESS, true);
            }
            return true;
        }

        watchedUrl = currentUrl;
        return false;
    }

    function markSuccess() {
        stopSniper('add to cart');
        GM_setValue(STORAGE_SUCCESS, true);
        playSuccessSound();
        notifySuccess();
        showHud('В корзину! Скрипт остановлен.', 'ok');

        if (CONFIG.goToCartOnSuccess) {
            location.replace(CONFIG.cartUrl);
        }
    }

    function trySnipe() {
        if (stopped || snipeLocked) return false;
        if (stopIfLeftTargetPage()) return false;

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
                ? `Ожидание… #${attempts} (Out of Stock)`
                : `Поиск кнопки… #${attempts}`,
            waiting ? 'warn' : 'info'
        );
        return false;
    }

    function maybeReload() {
        if (stopped) return;
        if (stopIfLeftTargetPage()) return;

        const now = Date.now();
        if (now - lastReloadAt < CONFIG.reloadIntervalMs) return;
        lastReloadAt = now;
        location.reload();
    }

    pollTimer = setInterval(() => {
        if (!stopped) trySnipe();
    }, CONFIG.pollIntervalMs);

    urlWatchTimer = setInterval(() => {
        stopIfLeftTargetPage();
    }, CONFIG.urlWatchIntervalMs);

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

    reloadTimer = setInterval(() => {
        if (!stopped) maybeReload();
    }, Math.max(100, CONFIG.reloadIntervalMs));

    window.addEventListener('popstate', stopIfLeftTargetPage, true);
    window.addEventListener('hashchange', stopIfLeftTargetPage, true);
    window.addEventListener('beforeunload', () => stopSniper('закрытие вкладки'), true);

    document.addEventListener('click', (event) => {
        if (stopped) return;
        const link = event.target.closest('a[href]');
        if (!link) return;
        try {
            const href = link.href || '';
            if (href.includes('/cart') || href.includes('/checkout')) {
                stopSniper('клик по корзине');
                if (CONFIG.stopAfterSuccess) GM_setValue(STORAGE_SUCCESS, true);
                removeHud();
            }
        } catch (e) {
            // ignore
        }
    }, true);

    trySnipe();

    window.rsiSniperStop = () => {
        stopSniper('вручную');
        removeHud();
        showHud('Остановлен вручную', 'info');
    };

    window.rsiSniperReset = () => {
        GM_setValue(STORAGE_SUCCESS, false);
        GM_setValue(STORAGE_ACTIVE, false);
        location.reload();
    };

    console.info('[RSI Sniper] Только страница Odin. rsiSniperStop() / rsiSniperReset()');
})();
