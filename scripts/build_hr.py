"""Build homophone-replacer rules from the glossary.

Whisper gets a company or product name phonetically right and orthographically wrong: the pinyin
matches, the characters do not. The homophone replacer fixes exactly that class of error, so every
Chinese glossary term becomes a rule "its pinyin → its characters".

    python -m scripts.build_hr            # writes models/hr/replace.txt (+ replace.fst if pynini)

pynini has no Windows wheel. On Windows this writes replace.txt only; build the .fst from it under
WSL, Linux, macOS, or Colab (see --help output) and copy the result back to models/hr/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config  # noqa: E402
from server.store import Store  # noqa: E402

HAN = range(0x4E00, 0xA000)


def to_pinyin(text: str) -> str:
    """Tone-numbered pinyin, no separators — the form replace.fst matches against."""
    from pypinyin import Style, lazy_pinyin

    # Neutral tone (5 in pypinyin) is written as 1 upstream, so normalize it.
    return "".join(p.replace("5", "1") for p in lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))


def chinese_terms() -> list[str]:
    seen = {t.source for t in Store().glossary() if any(ord(c) in HAN for c in t.source)}
    return sorted(seen)


def build_fst(rules: dict[str, str], out: Path) -> bool:
    try:
        import pynini
        from pynini.lib import utf8
    except ImportError:
        return False

    cross = None
    for pinyin, term in rules.items():
        one = pynini.cross(pinyin, term)
        cross = one if cross is None else (cross | one)
    rule = pynini.cdrewrite(cross.optimize(), "", "", utf8.VALID_UTF8_CHAR.star)
    rule.write(str(out))
    return True


def main() -> int:
    terms = chinese_terms()
    if not terms:
        print("no Chinese glossary terms — nothing to build", file=sys.stderr)
        return 1

    rules = {to_pinyin(t): t for t in terms}
    config.HR_DIR.mkdir(parents=True, exist_ok=True)

    txt = config.HR_DIR / "replace.txt"
    txt.write_text("".join(f"{p}\t{t}\n" for p, t in sorted(rules.items())), encoding="utf-8")
    print(f"{len(rules)} rules → {txt}")

    fst = config.HR_DIR / "replace.fst"
    if build_fst(rules, fst):
        print(f"wrote {fst}")
    else:
        print(f"pynini not available — build {fst.name} elsewhere from {txt.name}:\n"
              "  pip install --only-binary :all: pynini && python -m scripts.build_hr")

    needed = {"dict/": config.HR_DIR / "dict", "lexicon.txt": config.HR_DIR / "lexicon.txt"}
    missing = [name for name, path in needed.items() if not path.exists()]
    if missing:
        print(f"\nstill missing in {config.HR_DIR}: {', '.join(missing)}\n"
              "  curl -L -o dict.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2\n"
              "  tar -xjf dict.tar.bz2 && rm dict.tar.bz2\n"
              "  curl -L -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
