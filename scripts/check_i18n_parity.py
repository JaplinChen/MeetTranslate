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

A key built at runtime — t(`nav.${key}`) — cannot be resolved here, but its group can: the check
requires something to exist under "nav.", so a whole block going missing still fails. Which member
of the group the code wanted stays unverified, and the summary says so rather than implying the
coverage is complete.

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
# t(`nav.${key}`) — only the literal prefix is knowable; the rest is built at runtime.
DYNAMIC_CALL = re.compile(r"\bt\(\s*`([^`]*?)\$\{")


def flatten(node: object, prefix: str = "") -> set[str]:
    if not isinstance(node, dict):
        return {prefix.rstrip(".")}
    return {key for name, value in node.items() for key in flatten(value, f"{prefix}{name}.")}


def keys_used_in_source() -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Keys the dashboard asks for, mapped to the files asking.

    Returns the fully-literal ones and, separately, the literal prefixes of the interpolated calls:
    t(`nav.${key}`) yields "nav.". Which member of that group is wanted is only known at runtime,
    but the group itself must exist — that much is worth holding onto rather than skipping the call
    entirely.
    """
    used: dict[str, list[Path]] = {}
    prefixes: dict[str, list[Path]] = {}
    for path in sorted(DASHBOARD.rglob("*.ts*")):
        if path.name.endswith((".test.ts", ".test.tsx", ".d.ts")):
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(DASHBOARD)
        for match in STATIC_CALL.finditer(text):
            used.setdefault(match.group(2), []).append(rel)
        for match in DYNAMIC_CALL.finditer(text):
            if prefix := match.group(1):
                prefixes.setdefault(prefix, []).append(rel)
    return used, prefixes


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

    used, prefixes = keys_used_in_source()
    if undefined := sorted(k for k in used if k not in every):
        failed = True
        print(f"{len(undefined)} key(s) used in the source that no locale defines:")
        for key in undefined[:20]:
            where = ", ".join(sorted({str(p) for p in used[key]}))
            print(f"  {key}  ({where})")
        if len(undefined) > 20:
            print(f"  … and {len(undefined) - 20} more")

    if empty := sorted(p for p in prefixes if not any(k.startswith(p) for k in every)):
        failed = True
        print(f"{len(empty)} interpolated key group(s) with nothing under them:")
        for prefix in empty:
            where = ", ".join(sorted({str(p) for p in prefixes[prefix]}))
            print(f"  {prefix}${{…}}  ({where})")

    if failed:
        return 1

    print(f"{len(files)} locales, {len(every)} keys each — parity OK")
    print(f"{len(used)} literal key(s) used in the source, all defined")
    print(
        f"{len(prefixes)} interpolated group(s) present ({', '.join(sorted(prefixes))}) — "
        "membership is decided at runtime and is not checked"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
