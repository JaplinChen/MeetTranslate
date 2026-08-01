"""Repair an existing transcript with an LLM that can see the conversation around each line.

    python -m scripts.refine_transcript transcripts/clean/DXC-0721-開發.txt --topic "SAP ERP 導入訪談"

Writes `<name>.refined.txt` next to the input and prints every change, so the pass can be judged
rather than trusted. The API key comes from llm.json or ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import llm, refine  # noqa: E402
from server.store import Store  # noqa: E402

LINE = re.compile(r"^\[(\d+:\d+)\] (S\d+) \((\w+)\) (.*)$")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--topic", default="SAP ERP 導入訪談", help="what the meeting is about")
    ap.add_argument("--model", default="", help="override the model in llm.json")
    ap.add_argument("--ollama", nargs="?", const="qwen3:14b", default=None,
                    help="use a local Ollama model instead of the cloud (default qwen3:14b)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N lines, for a trial run")
    ap.add_argument("--think", action="store_true",
                    help="let a reasoning model deliberate; far slower, and not obviously better")
    args = ap.parse_args()

    cfg = llm.load_llm()
    if args.ollama:
        chat = refine.ollama_chat(args.model or args.ollama, llm.DEFAULT_ENDPOINTS["ollama"],
                                  think=args.think)
        label = f"ollama/{args.model or args.ollama}"
    else:
        key = os.environ.get("ANTHROPIC_API_KEY") or cfg.api_key
        if not key:
            print("no API key: set ANTHROPIC_API_KEY, configure llm.json, or pass --ollama",
                  file=sys.stderr)
            return 1
        chat = refine.anthropic_chat(key, args.model or cfg.model)
        label = args.model or cfg.model

    stamps: list[tuple[str, str, str]] = []
    lines: list[refine.Line] = []
    for row in args.transcript.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := LINE.match(row):
            stamp, speaker, lang, text = m.groups()
            stamps.append((stamp, speaker, lang))
            lines.append(refine.Line(speaker, lang, text))

    if not lines:
        print(f"no transcript lines in {args.transcript}", file=sys.stderr)
        return 1

    if args.limit:
        stamps, lines = stamps[: args.limit], lines[: args.limit]

    chunks = (len(lines) + refine.CHUNK_LINES - 1) // refine.CHUNK_LINES
    print(f"{len(lines)} lines, {chunks} requests, model {label}")
    coverage = refine.Coverage()
    fixed = refine.Refiner(chat, topic=args.topic).refine(lines, Store().glossary(),
                                                          coverage=coverage)

    changed = 0
    out = []
    for (stamp, speaker, lang), before, after in zip(stamps, (l.text for l in lines), fixed):
        if before != after:
            changed += 1
            print(f"  [{stamp}] {before}\n         -> {after}")
        out.append(f"[{stamp}] {speaker} ({lang}) {after}")

    target = args.transcript.with_suffix(".refined.txt")
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\n{changed}/{len(lines)} lines changed -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
