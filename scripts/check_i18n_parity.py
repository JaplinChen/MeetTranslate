"""Fail if the locale files disagree about which keys exist.

A key present in one locale and missing from another renders as the raw key path on screen for
whoever picked that language. Nothing else catches it: the build succeeds and TypeScript cannot
see inside JSON translation files.

Run: python scripts/check_i18n_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LOCALES = Path(__file__).resolve().parent.parent / "dashboard" / "src" / "i18n" / "locales"


def flatten(node: object, prefix: str = "") -> set[str]:
    if not isinstance(node, dict):
        return {prefix.rstrip(".")}
    return {key for name, value in node.items() for key in flatten(value, f"{prefix}{name}.")}


def main() -> int:
    files = sorted(LOCALES.glob("*.json"))
    if len(files) < 2:
        print(f"expected at least two locale files under {LOCALES}, found {len(files)}")
        return 1

    keys = {f.stem: flatten(json.loads(f.read_text(encoding="utf-8"))) for f in files}
    # Reference is the union, so a key missing everywhere but one still gets reported.
    every = set().union(*keys.values())

    failed = False
    for name, present in sorted(keys.items()):
        if missing := sorted(every - present):
            failed = True
            print(f"{name}.json is missing {len(missing)} key(s):")
            for key in missing[:20]:
                print(f"  {key}")
            if len(missing) > 20:
                print(f"  … and {len(missing) - 20} more")

    if failed:
        return 1

    print(f"{len(files)} locales, {len(every)} keys each — parity OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
