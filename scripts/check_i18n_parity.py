"""Fail if the locale files disagree about which keys exist, or lack one the code asks for.

A key present in one locale and missing from another renders as the raw key path on screen for
whoever picked that language. Nothing else catches it: the build succeeds and TypeScript cannot
see inside JSON translation files.

Comparing the locales only to each other is not enough, though. The LLM settings page called for
thirteen keys that no locale defined, so the files agreed perfectly and this script passed while
the page showed "llm.verify" on a button. So the second check reads every t('...') in the
dashboard source and requires a definition somewhere.

A t() call carrying a defaultValue still has to be translated. It renders English in every
language, which is a quieter bug than a raw key path but a bug all the same.

Run: python scripts/check_i18n_parity.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard" / "src"
LOCALES = DASHBOARD / "i18n" / "locales"

# t('some.key') / t("some.key"), with or without an options object after it.
STATIC_CALL = re.compile(r"""\bt\(\s*(['"])([^'"]+)\1""")
# t(`nav.${key}`) — the key is only known at runtime, so it cannot be checked here.
DYNAMIC_CALL = re.compile(r"\bt\(\s*`")


def flatten(node: object, prefix: str = "") -> set[str]:
    if not isinstance(node, dict):
        return {prefix.rstrip(".")}
    return {key for name, value in node.items() for key in flatten(value, f"{prefix}{name}.")}


def keys_used_in_source() -> tuple[dict[str, list[Path]], int]:
    """Every t('...') key in the dashboard, mapped to the files using it, plus a count of the
    t(`...`) calls skipped — reported rather than passed over in silence, since they are a real
    hole in the coverage."""
    used: dict[str, list[Path]] = {}
    dynamic = 0
    for path in sorted(DASHBOARD.rglob("*.ts*")):
        if path.name.endswith((".test.ts", ".test.tsx", ".d.ts")):
            continue
        text = path.read_text(encoding="utf-8")
        for match in STATIC_CALL.finditer(text):
            used.setdefault(match.group(2), []).append(path.relative_to(DASHBOARD))
        dynamic += len(DYNAMIC_CALL.findall(text))
    return used, dynamic


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

    used, dynamic = keys_used_in_source()
    if undefined := sorted(k for k in used if k not in every):
        failed = True
        print(f"{len(undefined)} key(s) used in the source that no locale defines:")
        for key in undefined[:20]:
            where = ", ".join(sorted({str(p) for p in used[key]}))
            print(f"  {key}  ({where})")
        if len(undefined) > 20:
            print(f"  … and {len(undefined) - 20} more")

    if failed:
        return 1

    skipped = f", {dynamic} dynamic t(`…`) call(s) not checkable" if dynamic else ""
    print(f"{len(files)} locales, {len(every)} keys each — parity OK")
    print(f"{len(used)} key(s) used in the source, all defined{skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
