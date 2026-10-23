"""Worker process (``python -m app.workers.worker``) and the in-process worker thread used in inline mode."""

from __future__ import annotations

import os
import signal
import socket
import threading
import time

from app.core.config import get_settings
from app.observability.logging import configure_logging, get_logger
from app.persistence.db import session_scope
from app.services.model_manager import get_registry
from app.workers.queue import requeue_stale
from app.workers.runner import run_one

log = get_logger(__name__)


class Worker:
    def __init__(self, name: str | None = None) -> None:
        self.name = name or f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident() % 10000}"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        s = get_settings()
        last_maintenance = 0.0
        last_cleanup = 0.0
        log.info("worker started", extra={"worker": self.name})
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_maintenance > 60:
                last_maintenance = now
                try:
                    with session_scope() as db:
                        n = requeue_stale(db, s.worker_stale_lock_seconds, s.job_max_attempts)
                    if n:
                        log.warning("recovered stale jobs", extra={"count": n})
                    get_registry().unload_idle()
                    if now - last_cleanup > 3600:
                        last_cleanup = now
                        from app.services.jobs import cleanup_expired_media
                        from app.services.storage import get_storage

                        with session_scope() as db:
                            removed = cleanup_expired_media(db, get_storage())
                        if removed:
                            log.info("retention cleanup", extra={"media_deleted": removed})
                except Exception:  # noqa: BLE001
                    log.exception("maintenance failed")
            try:
                did = run_one(self.name)
            except Exception:  # noqa: BLE001 - keep the worker alive
                log.exception("unexpected worker error")
                did = None
            if did is None:
                self._stop.wait(s.worker_poll_seconds)
        log.info("worker stopped", extra={"worker": self.name})

    def start_thread(self) -> threading.Thread:
        self._thread = threading.Thread(target=self.run_forever, name="rc-inline-worker", daemon=True)
        self._thread.start()
        return self._thread

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)


def main() -> None:  # pragma: no cover - process entry point
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    from app.persistence.db import init_engine

    init_engine()
    w = Worker()
    signal.signal(signal.SIGTERM, lambda *_: w.stop())
    signal.signal(signal.SIGINT, lambda *_: w.stop())
    w.run_forever()
    get_registry().unload()


if __name__ == "__main__":  # pragma: no cover
    main()
