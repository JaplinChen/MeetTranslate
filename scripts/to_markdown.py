"""Turn benchmark transcripts into documents someone can read.

    python -m scripts.to_markdown transcripts/final/*.txt

The line-per-utterance form is what the pipeline produces and what every measurement in this
project runs over, but a ninety-minute interview is a thousand of those lines and nobody reads it
that way. Consecutive lines from one speaker are joined into a turn, which is how a transcript is
normally laid out and roughly a tenth as many blocks on the page.

Speaker codes stay as they are. Naming them is done once in the app, against the voiceprint, and
guessing here would put names on the page that nothing stands behind.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LINE = re.compile(r"^\[(\d+):(\d+)\] (S\d+) \((\w+)\) (.*)$")
LANGUAGE = {"zh": "中文", "vi": "越南語", "en": "英語"}
# Silence that ends a turn even when the same person carries on afterwards.
PAUSE_SECONDS = 20


def read(path: Path) -> list[tuple[int, str, str, str]]:
    """(seconds, speaker, language, text) for every line."""
    out = []
    for row in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := LINE.match(row):
            out.append((int(m.group(1)) * 60 + int(m.group(2)), m.group(3), m.group(4), m.group(5)))
    return out


def turns(lines: list[tuple[int, str, str, str]]) -> list[tuple[int, str, str, str]]:
    """Consecutive lines from one speaker in one language, joined into a turn.

    A pause ends a turn even when the same person resumes. Without that, a meeting where one
    speaker holds the floor produces paragraphs of seventeen hundred characters covering half an
    hour — technically one turn, unreadable as a document, and impossible to cite by time.
    """
    out: list[list] = []
    for at, speaker, lang, text in lines:
        same = out and out[-1][1] == speaker and out[-1][2] == lang
        if same and at - out[-1][4] <= PAUSE_SECONDS:
            out[-1][3] += text if text[:1] in "，。、,." else " " + text
            out[-1][4] = at
        else:
            out.append([at, speaker, lang, text, at])
    return [(a, b, c, d) for a, b, c, d, _ in out]


def clock(seconds: int) -> str:
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def render(name: str, lines: list[tuple[int, str, str, str]]) -> str:
    grouped = turns(lines)
    spoken = Counter()
    langs = Counter()
    for _, speaker, lang, text in lines:
        spoken[speaker] += len(text)
        langs[lang] += 1
    total = sum(spoken.values()) or 1
    main = langs.most_common(1)[0][0] if langs else "zh"

    out = [f"# {name}", ""]
    out += [f"逐字稿由語音辨識產生，含有辨識錯誤。共 {len(lines)} 句、{len(grouped)} 段發言，"
            f"時長約 {lines[-1][0] // 60} 分鐘。" if lines else "（無內容）", ""]

    out += ["## 發言者", ""]
    out += ["發言者以聲紋分辨，程式取不到與會者名單，因此標為 `S1`、`S2`。", ""]
    out += ["| 代號 | 發言佔比 |", "|---|---|"]
    for speaker, chars in spoken.most_common():
        share = chars * 100 / total
        if share >= 1:
            out.append(f"| {speaker} | {share:.0f}% |")
    minor = sum(1 for _, c in spoken.items() if c * 100 / total < 1)
    if minor:
        out.append(f"| 其餘 {minor} 位 | 各低於 1% |")

    out += ["", "## 語言", "",
            "、".join(f"{LANGUAGE.get(k, k)} {v} 句" for k, v in langs.most_common()), ""]
    out += ["## 逐字稿", ""]

    for at, speaker, lang, text in grouped:
        mark = "" if lang == main else f" _{LANGUAGE.get(lang, lang)}_"
        out.append(f"**{clock(at)} {speaker}**{mark}")
        out.append("")
        out.append(text)
        out.append("")

    return "\n".join(out) + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("transcripts/markdown"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.transcripts:
        lines = read(path)
        if not lines:
            print(f"{path.name}: no transcript lines", file=sys.stderr)
            continue
        target = args.out / f"{path.stem}.md"
        target.write_text(render(path.stem, lines), encoding="utf-8")
        print(f"{path.stem}: {len(lines)} 句 -> {len(turns(lines))} 段  {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
