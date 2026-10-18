"""Cross-process lock for one GeneResearcher analysis directory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TextIO


class AnalysisDirectoryLockError(RuntimeError):
    """Raised when another GeneResearcher session owns the analysis directory."""


class AnalysisDirectoryLock:
    """Hold an operating-system lock for the lifetime of an analysis session.

    The lock file remains in place so concurrent processes always lock the same
    inode. The OS releases the advisory lock automatically if a process exits
    unexpectedly, so a stale file does not block a later run.
    """

    def __init__(self, lock_path: str | Path, analysis_directory: str | Path) -> None:
        self.lock_path = Path(lock_path)
        self.analysis_directory = Path(analysis_directory)
        self._handle: TextIO | None = None

    def acquire(self) -> AnalysisDirectoryLock:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        self._handle = handle

        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write("\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            owner_pid = self._read_owner_pid(handle)
            handle.close()
            self._handle = None
            owner = f" (process {owner_pid})" if owner_pid else ""
            raise AnalysisDirectoryLockError(
                f"Another GeneResearcher session{owner} is using analysis directory "
                f"'{self.analysis_directory}'. Wait for it to finish or close the earlier "
                "AnalysisSession. If an earlier notebook cell was interrupted, restart "
                "that notebook runtime or Python kernel before retrying. This run has not "
                "cleared the existing analysis data."
            ) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        return self

    @staticmethod
    def _read_owner_pid(handle: TextIO) -> str | None:
        try:
            handle.seek(0)
            value = handle.read().strip()
        except OSError:
            return None
        return value if value.isdecimal() else None

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return

        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
