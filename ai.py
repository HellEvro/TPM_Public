#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Оболочка для защищённого AI лаунчера.
Вся рабочая логика находится в bot_engine/ai/_ai_launcher.pyc
"""

from typing import TYPE_CHECKING, Any
from bot_engine.ai import _infobot_ai_protected as _protected_module


if TYPE_CHECKING:
    def main(*args: Any, **kwargs: Any) -> Any: ...


_globals = globals()
_skip = {'__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__'}

for _key, _value in _protected_module.__dict__.items():
    if _key in _skip:
        continue
    _globals[_key] = _value

del _globals, _skip, _key, _value


if __name__ == '__main__':
    _protected_module.main()
