// ==UserScript==
// @name         RSI Sniper PING (test)
// @namespace    https://robertsspaceindustries.com/
// @version      1.0.0
// @description  Тест: работает ли Tampermonkey на RSI
// @match        *://robertsspaceindustries.com/*
// @match        *://*.robertsspaceindustries.com/*
// @run-at       document-idle
// @grant        none
// ==/UserScript==

(function () {
  'use strict';
  const id = 'rsi-sniper-ping';
  if (document.getElementById(id)) return;

  const el = document.createElement('div');
  el.id = id;
  el.textContent = 'TM PING OK on RSI';
  el.style.cssText = [
    'position:fixed',
    'top:80px',
    'right:16px',
    'z-index:2147483647',
    'padding:10px 14px',
    'background:#ff3b3b',
    'color:#fff',
    'font:700 14px sans-serif',
    'border-radius:8px',
    'box-shadow:0 6px 20px rgba(0,0,0,.5)',
  ].join(';');
  document.documentElement.appendChild(el);
  console.error('[RSI Sniper PING] Tampermonkey работает на', location.href);
})();
