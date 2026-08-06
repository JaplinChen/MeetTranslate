"""End-to-end checks over the HTTP API with a stubbed translator.

Run: python -m server.test_e2e

Real audio and a real Claude key are not available here, so the translator is replaced by a stub
and the pipeline is fed a recorded wav. What this proves is the wiring: capture control, the
store, the WebSocket fan-out, and that a `line` event followed by an `update` event rewrites the
subtitle in place rather than appending a duplicate.

The checks themselves live in the `test_e2e_*` modules and the fixtures in `e2e_support`; this
file is the runner. Every check runs against one temp directory and one store, in one alphabetical
order across all modules — the same order they ran in when they shared a namespace, because
several of them read what an earlier one wrote.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from fastapi.testclient import TestClient

from . import main
from . import test_e2e_api, test_e2e_ask, test_e2e_asr, test_e2e_pipeline, test_e2e_refine
from . import test_e2e_retry, test_e2e_sessions
from .e2e_support import isolate

MODULES = (test_e2e_api, test_e2e_ask, test_e2e_asr, test_e2e_pipeline, test_e2e_refine,
           test_e2e_retry, test_e2e_sessions)


def collect() -> list[tuple[str, Callable]]:
    """Every check in every module, in one global alphabetical order.

    Sorted across modules rather than within them: that is the order these ran in as one file, and
    the checks are not independent — they share a store, so one that seeds a session is relied on
    by a later one that reads it.
    """
    found: dict[str, Callable] = {}
    for module in MODULES:
        for name, fn in vars(module).items():
            if not name.startswith("test_"):
                continue
            if name in found:
                raise AssertionError(f"two checks named {name}; one would shadow the other")
            found[name] = fn
    return sorted(found.items())


def main_() -> None:
    checks = collect()
    # ignore_cleanup_errors: SQLite on Windows keeps a handle briefly after close, and a failed
    # rmtree must not mask a passing run.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp = Path(tmpdir)
        isolate(tmp)

        with TestClient(main.app) as client:
            for name, fn in checks:
                fn(tmp) if "tmp" in fn.__code__.co_varnames[: fn.__code__.co_argcount] else fn(client)
                print(f"ok  {name}")

        main.store.close()
    # Counted, not just "all passed": a check that stops being collected — a module dropped from
    # MODULES, a rename that loses the test_ prefix — is otherwise a silently shorter green run.
    print(f"\nall {len(checks)} e2e checks passed")


if __name__ == "__main__":
    main_()
