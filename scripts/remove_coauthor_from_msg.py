#!/usr/bin/env python3
"""Читает сообщение коммита из stdin, удаляет строки Co-authored-by: Cursor, выводит в stdout."""
import sys
MARKER = "Co-authored-by: Cursor <cursoragent@cursor.com>"
for line in sys.stdin:
    if MARKER not in line.rstrip("\r\n"):
        sys.stdout.write(line)
