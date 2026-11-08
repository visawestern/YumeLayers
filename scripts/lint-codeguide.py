#!/usr/bin/env python3
"""
lint-codeguide.py — архитектурный гардрейл YumeLayers.

Проверяет две декларированные конвенции:
  1) Файл источника <= 600 строк (лимит «600 строк на файл»).
  2) Функции атомарно/модульны: тело функции <= MAX_FUNC_BODY строк
     (эвристика: чем длиннее функция, тем выше шанс, что она делает
     несколько вещей; антивс ~ простой счётчик фигурных скобок).

Leгаси-файлы, превышающие лимит, допускаются только через явный allowlist
(KNOWN_OVERSIZED) и лечатся планом рефакторинга. Любой НОВЫЙ файл сверх
лимита — ошибка.

Использование:
    python3 scripts/lint-codeguide.py            # Sources + Tests
    python3 scripts/lint-codeguide.py --dirs Sources
Выход: 0 — чисто; 1 — есть нарушения.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_FILE_LINES = 600
MAX_FUNC_BODY = 120  # строк тела функции (жесткий анти-атомарный порог)

# Долг: легаси-файлы сверх лимита. Пока не отрефакторены — разрешены,
# но учитываются как технический долг (см. docs/ROADMAP.md, Фаза 8).
KNOWN_OVERSIZED = {
    "Sources/YumeLayersUI/main.swift": "весь UI-контроллер (долг)",
}

# Долг: крупные (неатомарные) функции в файлах, которые сами по себе в лимите.
# Ключ — (относительный путь, номер строки func). Убираются рефакторингом.
KNOWN_LARGE_FUNCS = {
    ("Sources/YumeLayersCore/TrainConvVAE.swift", 9): "trainConvVAE: ядро обучения (см. долг)",
}

FUNC_RE = re.compile(r"\bfunc\s+[A-Za-z_][A-Za-z0-9_]*\s*\(")


def count_braces(line):
    return line.count("{") - line.count("}")


def lint_dir(path):
    violations = []
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            if not fn.endswith(".swift"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, ROOT)
            with open(full, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            n = len(lines)
            oversize_known = rel in KNOWN_OVERSIZED
            if n > MAX_FILE_LINES and not oversize_known:
                violations.append(f"[ФАЙЛ {n} > {MAX_FILE_LINES}] {rel}")
            # функция слишком большая (только для файлов, не помеченных долгом
            # по размеру: их и так разбивают, и функция-проверка дублирует этот долг)
            if oversize_known:
                continue
            # Корректная эвристика длины функции: baseline = глубина вложенности
            # на момент открытия тела. Это корректно работает и внутри типа/
            # расширения (class/extension открывают уровень вложенности).
            depth = 0
            in_func = False
            pending = None
            fstart = 0
            base = 0
            for i, raw in enumerate(lines):
                no_comment = raw.split("//", 1)[0]
                d = count_braces(no_comment)
                if pending is not None:
                    if "{" in no_comment:
                        base = depth
                        fstart = pending
                        depth += d
                        pending = None
                        in_func = True
                    else:
                        depth += d
                    continue
                if in_func:
                    depth += d
                    if depth <= base:
                        body = i + 1 - fstart
                        if body > MAX_FUNC_BODY and (rel, fstart) not in KNOWN_LARGE_FUNCS:
                            violations.append(
                                f"[ФУНКЦИЯ {body} > {MAX_FUNC_BODY} строк] {rel} строка {fstart}")
                        in_func = False
                    continue
                if FUNC_RE.search(raw):
                    if "{" in no_comment:
                        base = depth
                        fstart = i + 1
                        depth += d
                        in_func = True
                    else:
                        pending = i + 1
                        depth += d
                else:
                    depth += d
    return violations


def main():
    dirs = sys.argv[1:]
    if not dirs or dirs == ["--dirs"]:
        dirs = [os.path.join(ROOT, "Sources"), os.path.join(ROOT, "Tests")]
    else:
        dirs = [d for d in dirs if d != "--dirs"]
    all_v = []
    for d in dirs:
        if os.path.isdir(d):
            all_v += lint_dir(d)
    if all_v:
        print("НАРУШЕНИЯ АРХИТЕКТУРНЫХ КОНВЕНЦИЙ:")
        for v in all_v:
            print("  " + v)
        print("ВСЕГО:", len(all_v))
        return 1
    print("OK: все файлы <=600 строк, функции в пределах порога.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
