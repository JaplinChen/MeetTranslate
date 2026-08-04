"""One card, one claimant. The gate that keeps the post-meeting pass off the live pipeline's GPU.

Two Whisper models do not fit the way the arithmetic suggests they might. A large-v3 in float16
plus a batched pipeline at BATCH_SIZE=32 is sized to have the card to itself; loading a second one
beside the live recogniser does not merely halve throughput, it pushes the live realtime factor
past 1, and once that happens `Pipeline.tap` fills its 600-block backlog in a minute and starts
discarding audio. Automating the post-meeting pass without this gate would trade a transcript
nobody re-ran for subtitles the room watched go missing.

The live meeting always wins. A background pass is worth minutes of GPU time; a meeting happening
right now is not repeatable. Cancellation is cooperative — `postprocess` checks between batches —
and safe to act on, because `Store.replace_lines` only swaps the transcript once, at the end. An
abandoned pass leaves the previous transcript exactly as it was.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger("meettranslate.jobs")

# Seconds a starting meeting waits for a cancelled pass to notice and let go. Cancellation is
# checked between decode batches, so the wait is one batch, not one meeting.
YIELD_TIMEOUT = 30.0

_gpu = threading.BoundedSemaphore(1)


@dataclass
class Job:
    """What the dashboard needs to say about a session's post-meeting pass."""

    state: str = "refining"  # refining | refined | failed | cancelled
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


_jobs: dict[int, Job] = {}
_lock = threading.Lock()


def state(session_id: int) -> dict | None:
    with _lock:
        job = _jobs.get(session_id)
        return {"state": job.state, "error": job.error} if job else None


def states() -> dict[int, dict]:
    with _lock:
        return {sid: {"state": j.state, "error": j.error} for sid, j in _jobs.items()}


def schedule(session_id: int, run: Callable[[threading.Event], None]) -> bool:
    """Run `run` on a worker once the GPU is free. False when this session already has a pass.

    `run` is handed the cancel event and is expected to check it; a pass that ignores it simply
    finishes, which is correct but makes the next meeting wait.
    """
    with _lock:
        existing = _jobs.get(session_id)
        # Thread liveness, not just the recorded state: a worker that has finished its run but not
        # yet been marked still holds the gate, and replacing its entry here would orphan it —
        # the permit never comes back and every later pass waits on a job nobody is tracking.
        if existing and (existing.state == "refining"
                         or (existing.thread and existing.thread.is_alive())):
            return False
        job = Job()
        _jobs[session_id] = job

    def worker() -> None:
        with _gpu:
            if job.cancel.is_set():
                _finish(job, "cancelled")
                return
            try:
                run(job.cancel)
            except Cancelled:
                _finish(job, "cancelled")
                return
            except Exception as exc:
                # A failed pass is a visible state, not a log line nobody reads. The transcript it
                # was rewriting is still whole, so this is recoverable by re-running.
                log.exception("post-meeting pass failed for session %d", session_id)
                _finish(job, "failed", f"{type(exc).__name__}: {exc}")
                return
        _finish(job, "refined")

    job.thread = threading.Thread(target=worker, name=f"reprocess-{session_id}", daemon=True)
    job.thread.start()
    return True


def _finish(job: Job, state_: str, error: str = "") -> None:
    with _lock:
        job.state, job.error = state_, error


class Cancelled(Exception):
    """Raised inside a pass when a meeting needs the card back. Nothing was written."""


# Defined above `schedule` in spirit but placed here for readability; `schedule` catches it to mark
# the job cancelled rather than failed. One name for the concept, so a pass that yields politely is
# never filed as a crash.


def claim_gpu(timeout: float = YIELD_TIMEOUT) -> bool:
    """Take the card for a live meeting, asking any background pass to stand down first."""
    with _lock:
        running = [j for j in _jobs.values() if j.state == "refining"]
    for job in running:
        job.cancel.set()
    return _gpu.acquire(timeout=timeout)


def release_gpu() -> None:
    try:
        _gpu.release()
    except ValueError:
        # Releasing a gate this process never took would mask the bug rather than fix it, so it is
        # logged and swallowed: a stop must never fail on account of bookkeeping.
        log.warning("release_gpu called without a matching claim")


@contextmanager
def borrow_gpu(timeout: float = 900.0):
    """Wait your turn for the card. For work that is not a live meeting and must not preempt one.

    An import runs the same post-meeting pass, but nothing about it is time-critical, so it queues
    behind whatever holds the card instead of asking it to stand down.
    """
    if not _gpu.acquire(timeout=timeout):
        raise TimeoutError("the GPU is busy")
    try:
        yield
    finally:
        release_gpu()


def cancel_all(wait: float = 0.0) -> None:
    """Ask every running pass to stop. Used on shutdown, where nothing should outlive the process."""
    with _lock:
        running = [j for j in _jobs.values() if j.state == "refining"]
    for job in running:
        job.cancel.set()
    if wait:
        for job in running:
            if job.thread:
                job.thread.join(timeout=wait)


def reset() -> None:
    """Drop all job state. Tests only."""
    with _lock:
        _jobs.clear()
