"""Fail-closed backup and restore utility for experiment output directories.

This module intentionally uses only the Python standard library.  It does not
import the experiment runtime, CybORG, an LLM provider, or any project model.
The source and destination boundaries are deliberately narrow because a
backup is an audit artifact, not a general-purpose file copier.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
from contextlib import contextmanager
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable


OUTPUTS_ROOT = Path(__file__).resolve().parents[2] / "outputs"
DEFAULT_BACKUP_ROOT = Path(r"C:\aptdetect_experiment_backups")
_BACKUP_SCHEMA = "aptdetect_experiment_backup_v1"
_CHUNK_SIZE = 1024 * 1024
_TWO_GIB = 2 * 1024 * 1024 * 1024
_TWENTY_GIB = 20 * 1024 * 1024 * 1024
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_DEPENDENCY_BACKUP_KIND = "verified_dependency"
_GIT_HEAD_RE = re.compile(r"[0-9a-fA-F]{40}")
_DEPENDENCY_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_BACKUP_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")

# Windows handle flags used by the dependency path.  Path-only copying is
# deliberately not a fallback: the verified-dependency contract is intended
# to survive a concurrent rename/reparse attack, so an unsupported platform
# fails closed.
_WIN_INVALID_HANDLE = ctypes.c_void_p(-1).value
_WIN_GENERIC_READ = 0x80000000
_WIN_GENERIC_WRITE = 0x40000000
_WIN_FILE_SHARE_READ = 0x00000001
_WIN_FILE_SHARE_WRITE = 0x00000002
_WIN_FILE_SHARE_DELETE = 0x00000004
_WIN_OPEN_EXISTING = 3
_WIN_CREATE_NEW = 1
_WIN_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_WIN_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_WIN_FILE_ATTRIBUTE_NORMAL = 0x00000080
_WIN_MOVEFILE_WRITE_THROUGH = 0x00000008
_WIN_ERROR_FILE_EXISTS = 80
_WIN_ERROR_ALREADY_EXISTS = 183


class BackupError(RuntimeError):
    """A fail-closed backup operation error."""


def _absolute(path: Path | str) -> Path:
    """Return an absolute lexical path without following links."""

    return Path(os.path.abspath(os.fspath(path)))


def _anchor_components(path: Path) -> tuple[Path, tuple[str, ...]]:
    """Split a lexical path at its volume/UNC/root anchor."""

    value = _absolute(path)
    anchor = Path(value.anchor) if value.anchor else Path()
    # ``Path.parts`` includes the drive/UNC anchor as the first component on
    # Windows and '/' on POSIX.  Rebuilding from the anchor avoids resolving
    # any component while still allowing every existing ancestor to be
    # inspected individually.
    parts = value.parts
    anchor_parts = anchor.parts
    if anchor_parts and tuple(parts[: len(anchor_parts)]) == tuple(anchor_parts):
        return anchor, tuple(parts[len(anchor_parts) :])
    if value.anchor:
        return anchor, tuple(part for part in parts if part != value.anchor)
    return Path(), tuple(parts)


def _is_reparse(path: Path) -> bool:
    """Return whether *path* is a symlink or Windows reparse point."""

    try:
        info = os.lstat(path)
    except OSError as exc:
        raise BackupError(f"cannot inspect path {path}: {exc}") from exc
    return bool(stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0) & _REPARSE_POINT))


def _check_all_ancestor_components(path: Path) -> None:
    """Reject any existing reparse point from the volume/UNC anchor onward."""

    anchor, parts = _anchor_components(path)
    current = anchor
    if os.path.lexists(os.fspath(current)):
        if _is_reparse(current):
            raise BackupError(f"ancestor reparse point is not allowed: {current}")
    for part in parts:
        current = current / part
        # Missing descendants are safe to create only after their existing
        # parent chain has been inspected.  A symlink can report exists=False
        # on some Windows providers, so is_symlink is checked separately.
        if os.path.lexists(os.fspath(current)):
            if _is_reparse(current):
                raise BackupError(f"ancestor reparse point is not allowed: {current}")


def _path_parts_from_root(path: Path, root: Path) -> tuple[str, ...]:
    path_abs = _absolute(path)
    root_abs = _absolute(root)
    try:
        relative = path_abs.relative_to(root_abs)
    except ValueError as exc:
        raise BackupError(f"path must be inside {root_abs}: {path_abs}") from exc
    return relative.parts


def _check_existing_components(path: Path, root: Path) -> None:
    """Reject symlinks/reparse points in every existing path component."""

    root_abs = _absolute(root)
    path_abs = _absolute(path)
    _check_all_ancestor_components(path_abs)
    # The root may itself be a temporary test path, so do not require it to
    # exist here.  Existing ancestors are checked up to the common root.
    try:
        relative = path_abs.relative_to(root_abs)
    except ValueError as exc:
        raise BackupError(f"path must be inside {root_abs}: {path_abs}") from exc

    current = root_abs
    if current.exists() and _is_reparse(current):
        raise BackupError(f"reparse point is not allowed: {current}")
    for part in relative.parts:
        current = current / part
        if os.path.lexists(os.fspath(current)):
            if _is_reparse(current):
                raise BackupError(f"symlink/reparse point is not allowed: {current}")


def _dependency_secure_io_supported() -> bool:
    """Return whether the platform has the required handle-relative primitives."""

    if os.name == "nt":
        return all(hasattr(ctypes, name) for name in ("WinDLL", "byref"))
    return (
        all(hasattr(os, name) for name in ("open", "mkdir", "O_NOFOLLOW", "O_DIRECTORY"))
        and hasattr(os, "supports_dir_fd")
        and os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
    )


def _require_dependency_secure_io() -> None:
    if not _dependency_secure_io_supported():
        raise BackupError("verified-dependency backup requires supported secure handle-relative I/O")


def _windows_path_key(path: str | Path) -> str:
    value = os.fspath(path).replace("/", "\\")
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.normpath(value))


def _windows_is_within(path: str | Path, root: str | Path) -> bool:
    child = _windows_path_key(path)
    parent = _windows_path_key(root)
    return child == parent or child.startswith(parent.rstrip("\\") + "\\")


def _windows_open_handle(
    path: Path,
    *,
    directory: bool,
    create_new: bool = False,
    writable: bool = False,
    share_mode: int | None = None,
) -> int:
    """Open a Windows path with reparse-point-safe flags and no replacement."""

    if os.name != "nt":
        raise BackupError("Windows secure handle requested on a non-Windows platform")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    flags = _WIN_FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= _WIN_FILE_FLAG_BACKUP_SEMANTICS
    access = _WIN_GENERIC_READ | (_WIN_GENERIC_WRITE if (create_new or writable) else 0)
    creation = _WIN_CREATE_NEW if create_new else _WIN_OPEN_EXISTING
    if share_mode is None:
        share_mode = _WIN_FILE_SHARE_READ | _WIN_FILE_SHARE_WRITE | _WIN_FILE_SHARE_DELETE
    handle = kernel32.CreateFileW(
        os.fspath(path),
        access,
        share_mode,
        None,
        creation,
        flags | (_WIN_FILE_ATTRIBUTE_NORMAL if not directory else 0),
        None,
    )
    if handle == wintypes.HANDLE(_WIN_INVALID_HANDLE).value:
        error = ctypes.get_last_error()
        if create_new and error in (_WIN_ERROR_FILE_EXISTS, _WIN_ERROR_ALREADY_EXISTS):
            raise BackupError(f"refusing to overwrite existing file: {path}")
        raise BackupError(f"CreateFileW failed for {path}: WinError {error}")
    return int(handle)


def _windows_close_handle(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(wintypes.HANDLE(handle)):
        raise BackupError(f"CloseHandle failed: WinError {ctypes.get_last_error()}")


def _windows_handle_final_path(handle: int) -> str:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetFinalPathNameByHandleW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    capacity = 512
    while capacity <= 32768:
        buffer = ctypes.create_unicode_buffer(capacity)
        length = kernel32.GetFinalPathNameByHandleW(wintypes.HANDLE(handle), buffer, capacity, 0)
        if length == 0:
            raise BackupError(f"GetFinalPathNameByHandleW failed: WinError {ctypes.get_last_error()}")
        if length < capacity - 1:
            return buffer.value[:length]
        capacity *= 2
    raise BackupError("GetFinalPathNameByHandleW returned an overlong path")


class _WindowsByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", wintypes.FILETIME),
        ("ftLastAccessTime", wintypes.FILETIME),
        ("ftLastWriteTime", wintypes.FILETIME),
        ("dwVolumeSerialNumber", wintypes.DWORD),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("nNumberOfLinks", wintypes.DWORD),
        ("nFileIndexHigh", wintypes.DWORD),
        ("nFileIndexLow", wintypes.DWORD),
    ]


def _windows_handle_info(handle: int) -> dict[str, int]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_WindowsByHandleFileInformation)]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    info = _WindowsByHandleFileInformation()
    if not kernel32.GetFileInformationByHandle(wintypes.HANDLE(handle), ctypes.byref(info)):
        raise BackupError(f"GetFileInformationByHandle failed: WinError {ctypes.get_last_error()}")
    return {
        "link_count": int(info.nNumberOfLinks),
        "volume_serial": int(info.dwVolumeSerialNumber),
        "file_index": (int(info.nFileIndexHigh) << 32) | int(info.nFileIndexLow),
        "attributes": int(info.dwFileAttributes),
    }


def _posix_openat_path(root: Path, relative: Path, *, flags: int, mode: int = 0o600) -> int:
    if os.name == "nt":
        raise BackupError("POSIX openat requested on Windows")
    root_fd = os.open(os.fspath(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    current_fd = root_fd
    try:
        parts = relative.parts
        if not parts:
            raise BackupError("secure dependency file path cannot be the root")
        for component in parts[:-1]:
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        return os.open(parts[-1], flags | os.O_NOFOLLOW, mode, dir_fd=current_fd)
    except OSError as exc:
        raise BackupError(f"secure openat failed for {root / relative}: {exc}") from exc
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


@contextmanager
def _secure_dependency_file(path: Path, anchor: Path, *, write: bool = False, create_new: bool = False):
    """Yield a file object opened beneath an anchored directory.

    The yielded object is backed by the opened handle; no subsequent read or
    write goes through a path.  ``link_count`` and identity are returned for
    the selected-file hardlink gate.
    """

    _require_dependency_secure_io()
    root = _absolute(anchor)
    target = _absolute(path)
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise BackupError(f"dependency path escaped secure anchor: {target}") from exc
    if not relative.parts:
        raise BackupError("dependency path cannot be the secure anchor itself")
    if os.name == "nt":
        import msvcrt
        root_handle = _windows_open_handle(root, directory=True)
        try:
            root_info = _windows_handle_info(root_handle)
            if root_info["attributes"] & _REPARSE_POINT:
                raise BackupError(f"reparse point is not allowed for secure anchor: {root}")
            root_final = _windows_handle_final_path(root_handle)
            if not _windows_is_within(root_final, _absolute(root)):
                raise BackupError(f"secure anchor final path escaped: {root_final}")
            file_handle = _windows_open_handle(target, directory=False, create_new=create_new)
            try:
                final_path = _windows_handle_final_path(file_handle)
                if not _windows_is_within(final_path, root_final):
                    raise BackupError(f"opened dependency file escaped anchor: {final_path}")
                info = _windows_handle_info(file_handle)
                if info["attributes"] & _REPARSE_POINT:
                    raise BackupError(f"reparse point is not allowed: {target}")
                fd_flags = os.O_BINARY | (os.O_WRONLY if write else os.O_RDONLY)
                fd = msvcrt.open_osfhandle(file_handle, fd_flags)
                file_handle = None
                mode = "wb" if write else "rb"
                with os.fdopen(fd, mode) as handle:
                    yield handle, info
            finally:
                if file_handle is not None:
                    _windows_close_handle(file_handle)
        finally:
            _windows_close_handle(root_handle)
        return

    flags = os.O_WRONLY if write else os.O_RDONLY
    if create_new:
        flags |= os.O_CREAT | os.O_EXCL
    fd = _posix_openat_path(root, relative, flags)
    try:
        info = os.fstat(fd)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise BackupError(f"dependency file is not a regular non-link file: {target}")
        with os.fdopen(fd, "wb" if write else "rb") as handle:
            yield handle, {"link_count": int(info.st_nlink), "device": int(info.st_dev), "inode": int(info.st_ino)}
        fd = None
    finally:
        if fd is not None:
            os.close(fd)


def _dependency_handle_identity(handle: Any) -> dict[str, int]:
    """Read identity/link-count from the already-open dependency handle."""

    if os.name == "nt":
        import msvcrt
        return _windows_handle_info(msvcrt.get_osfhandle(handle.fileno()))
    info = os.fstat(handle.fileno())
    return {"link_count": int(info.st_nlink), "device": int(info.st_dev), "inode": int(info.st_ino)}


def _dependency_identity_matches(before: dict[str, int], after: dict[str, int]) -> bool:
    if type(before.get("link_count")) is not int or before.get("link_count") != 1:
        return False
    if type(after.get("link_count")) is not int or after.get("link_count") != 1:
        return False
    keys = ("volume_serial", "file_index", "device", "inode")
    compared = [key for key in keys if key in before or key in after]
    return all(before.get(key) == after.get(key) for key in compared)


def _secure_dependency_mkdirs(root: Path, relative: Path) -> None:
    """Create payload parents beneath an opened-directory boundary."""

    _require_dependency_secure_io()
    if not relative.parts:
        return
    if os.name == "nt":
        # Every created component is checked before the next one is opened;
        # final-file CreateFileW performs the no-replace boundary check.
        current = _absolute(root)
        for component in relative.parts:
            current = current / component
            try:
                current.mkdir()
            except FileExistsError:
                pass
            except OSError as exc:
                raise BackupError(f"cannot create secure payload directory {current}: {exc}") from exc
            _check_existing_components(current, root)
            if not current.is_dir():
                raise BackupError(f"payload parent is not a directory: {current}")
            _fsync_directory(current, strict=True)
        return
    current_fd = os.open(os.fspath(root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in relative.parts:
            try:
                os.mkdir(component, dir_fd=current_fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
    except OSError as exc:
        raise BackupError(f"cannot create secure payload directory beneath {root}: {exc}") from exc
    finally:
        os.close(current_fd)
    for index in range(1, len(relative.parts) + 1):
        _fsync_directory(root.joinpath(*relative.parts[:index]), strict=True)


class _WindowsDependencyLockSet:
    """Exclusive source-chain locks for the production dependency path.

    Windows has no POSIX-style ``openat`` equivalent in the Python standard
    library.  The dependency creator therefore keeps a no-share handle for
    every existing directory from the volume/UNC anchor through the source
    root and for every selected file.  If any component cannot be locked,
    creation fails closed before a byte is copied.
    """

    def __init__(self, source_root: Path, includes: tuple[str, ...]) -> None:
        self.source_root = _absolute(source_root)
        self.includes = includes
        self.directory_handles: list[int] = []
        self.file_handles: dict[str, tuple[Any, dict[str, int]]] = {}

    def _close(self) -> None:
        errors: list[Exception] = []
        for handle, _info in self.file_handles.values():
            try:
                handle.close()
            except OSError as exc:
                errors.append(exc)
        self.file_handles.clear()
        for handle in reversed(self.directory_handles):
            try:
                _windows_close_handle(handle)
            except BackupError as exc:
                errors.append(exc)
        self.directory_handles.clear()
        if errors:
            raise BackupError(f"failed to release dependency lock handles: {errors[0]}")

    def __enter__(self) -> "_WindowsDependencyLockSet":
        if os.name != "nt":
            raise BackupError("Windows dependency lock set requested on non-Windows platform")
        try:
            _check_all_ancestor_components(self.source_root)
            anchor, parts = _anchor_components(self.source_root)
            current = anchor
            chain = [current]
            for part in parts:
                current = current / part
                chain.append(current)
            # Include every selected-file parent directory, not merely the
            # source root.  Sorting by depth ensures the lock chain is opened
            # from the volume/UNC anchor outward.
            chain_set = {os.path.normcase(os.fspath(item)): item for item in chain}
            for relative in self.includes:
                target_parent = self.source_root.joinpath(*relative.split("/")).parent
                parent_anchor, parent_parts = _anchor_components(target_parent)
                current = parent_anchor
                chain_set.setdefault(os.path.normcase(os.fspath(current)), current)
                for part in parent_parts:
                    current = current / part
                    chain_set.setdefault(os.path.normcase(os.fspath(current)), current)
            chain = sorted(chain_set.values(), key=lambda item: len(item.parts))
            # The chain begins at the volume/UNC anchor by design.  Directory
            # handles share reads and writes so opening a volume root does not
            # lock unrelated handles, but never share DELETE/rename.
            directory_share = _WIN_FILE_SHARE_READ | _WIN_FILE_SHARE_WRITE
            anchor_final: str | None = None
            for component in chain:
                handle: int | None = None
                try:
                    handle = _windows_open_handle(
                        component,
                        directory=True,
                        share_mode=directory_share,
                    )
                    info = _windows_handle_info(handle)
                    if info["attributes"] & _REPARSE_POINT:
                        raise BackupError(f"dependency lock chain contains reparse point: {component}")
                    final_path = _windows_handle_final_path(handle)
                    if anchor_final is None:
                        anchor_final = final_path
                    if (
                        not _windows_is_within(final_path, anchor_final)
                        or _windows_path_key(final_path) != _windows_path_key(component)
                    ):
                        raise BackupError(f"dependency lock chain final path is invalid: {component}")
                    self.directory_handles.append(handle)
                    handle = None
                finally:
                    if handle is not None:
                        _windows_close_handle(handle)
            root_final = _windows_handle_final_path(self.directory_handles[-1])
            for relative in self.includes:
                target = self.source_root.joinpath(*relative.split("/"))
                file_handle: int | None = None
                try:
                    file_handle = _windows_open_handle(
                        target,
                        directory=False,
                        share_mode=0,
                    )
                    info = _windows_handle_info(file_handle)
                    final_path = _windows_handle_final_path(file_handle)
                    if (
                        info["attributes"] & _REPARSE_POINT
                        or _windows_path_key(final_path) != _windows_path_key(target)
                        or not _windows_is_within(final_path, root_final)
                    ):
                        raise BackupError(f"selected dependency escaped locked source root: {relative}")
                    import msvcrt
                    fd = msvcrt.open_osfhandle(file_handle, os.O_BINARY | os.O_RDONLY)
                    file_handle = None
                    self.file_handles[relative] = (os.fdopen(fd, "rb"), info)
                finally:
                    if file_handle is not None:
                        _windows_close_handle(file_handle)
            return self
        except Exception as exc:
            try:
                self._close()
            except BackupError:
                pass
            if isinstance(exc, BackupError):
                raise
            raise BackupError(
                f"cannot acquire exclusive Windows dependency chain locks: {exc}"
            ) from exc

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._close()

    def refresh_file_info(self, relative: str) -> dict[str, int]:
        if relative not in self.file_handles:
            raise BackupError(f"selected dependency is not held by a lock handle: {relative}")
        # The handle remains open; querying it after every read catches link
        # replacement and identity changes even when a provider reports stale
        # path metadata.
        file_handle = self.file_handles[relative][0]
        import msvcrt
        raw_handle = msvcrt.get_osfhandle(file_handle.fileno())
        return _windows_handle_info(raw_handle)

    def snapshot(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Hash selected source files only through the locked file handles.

        The returned content snapshot deliberately contains no filesystem
        identity fields.  Those are returned separately because a payload is
        expected to have a different identity even when its bytes are equal.
        """

        files: list[dict[str, Any]] = []
        identities: list[dict[str, Any]] = []
        for relative in self.includes:
            if relative not in self.file_handles:
                raise BackupError(f"selected dependency is not held by a lock handle: {relative}")
            handle = self.file_handles[relative][0]
            try:
                handle.seek(0)
                before = os.fstat(handle.fileno())
                before_identity = self.refresh_file_info(relative)
                if type(before_identity.get("link_count")) is not int or before_identity["link_count"] != 1:
                    raise BackupError(f"hardlinked dependency file is not allowed: {relative}")
                digest = hashlib.sha256()
                while True:
                    block = handle.read(_CHUNK_SIZE)
                    if not block:
                        break
                    digest.update(block)
                after = os.fstat(handle.fileno())
                after_identity = self.refresh_file_info(relative)
            except (OSError, ValueError) as exc:
                raise BackupError(f"cannot snapshot locked dependency file {relative}: {exc}") from exc
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise BackupError(f"dependency file changed while hashing: {relative}")
            if not _dependency_identity_matches(before_identity, after_identity):
                raise BackupError(f"dependency file identity/link count changed while hashing: {relative}")
            sha256 = digest.hexdigest()
            files.append(
                {
                    "relative_path": relative,
                    "bytes": int(after.st_size),
                    "sha256": sha256,
                    "link_count": int(after_identity["link_count"]),
                }
            )
            identities.append(
                {
                    "relative_path": relative,
                    "volume_serial": int(after_identity["volume_serial"]),
                    "file_index": int(after_identity["file_index"]),
                    "link_count": int(after_identity["link_count"]),
                    "bytes": int(after.st_size),
                    "mtime_ns": int(after.st_mtime_ns),
                    "sha256": sha256,
                }
            )
        snapshot = {
            "file_count": len(files),
            "total_bytes": sum(int(item["bytes"]) for item in files),
            "files": files,
            "snapshot_digest_sha256": _dependency_snapshot_digest(files),
        }
        return snapshot, identities


def _validate_source(source: Path, outputs_root: Path | None = None) -> Path:
    root = _absolute(outputs_root or OUTPUTS_ROOT)
    source_abs = _absolute(source)
    if not root.exists() or not root.is_dir() or _is_reparse(root):
        raise BackupError(f"outputs root is missing or unsafe: {root}")
    relative = _path_parts_from_root(source_abs, root)
    if not relative:
        raise BackupError("the outputs root itself cannot be backed up")
    _check_existing_components(source_abs, root)
    if not source_abs.exists() or not source_abs.is_dir():
        raise BackupError(f"source must be an existing directory: {source_abs}")
    return source_abs


def _validate_backup_root(backup_root: Path | str) -> Path:
    """Validate the only supported backup root.

    Tests may replace ``DEFAULT_BACKUP_ROOT`` with a temporary directory.  In
    normal use this function therefore accepts exactly that constant and no
    arbitrary alternate destination supplied by a caller.
    """

    root = _absolute(backup_root)
    allowed = _absolute(DEFAULT_BACKUP_ROOT)
    if os.path.normcase(os.fspath(root)) != os.path.normcase(os.fspath(allowed)):
        raise BackupError(f"backup root is restricted to {allowed}; got {root}")
    _check_all_ancestor_components(root)
    # Check existing ancestors before a caller creates the root.  The drive
    # root is normally the first existing component on Windows.
    current = root
    while not current.exists() and current != current.parent:
        current = current.parent
    if current.exists() and _is_reparse(current):
        raise BackupError(f"backup root ancestor is a symlink/reparse point: {current}")
    if root.exists():
        if not root.is_dir() or _is_reparse(root):
            raise BackupError(f"backup root is not a safe directory: {root}")
    return root


def _check_backup_path(path: Path, backup_root: Path, *, must_exist: bool = False) -> Path:
    root = _validate_backup_root(backup_root)
    value = _absolute(path)
    _path_parts_from_root(value, root)
    _check_existing_components(value, root)
    if must_exist and (not value.exists() or not value.is_dir()):
        raise BackupError(f"backup directory is missing: {value}")
    if value.exists() and _is_reparse(value):
        raise BackupError(f"backup directory is a symlink/reparse point: {value}")
    return value


def _safe_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BackupError(f"cannot read {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BackupError(f"{description} must be a JSON object: {path}")
    return value


def _read_run_status(source: Path, expected: str | None = None) -> tuple[dict[str, Any], str]:
    path = source / "run_status.json"
    if path.is_file() and not _is_reparse(path):
        payload = _safe_json(path, "run_status.json")
        status = payload.get("status")
        if not isinstance(status, str) or status not in {"STOPPED", "PARTIAL", "PASS"}:
            raise BackupError("run_status.json status must be exactly STOPPED, PARTIAL, or PASS")
    else:
        # Table 2/3 train-only orchestration intentionally stops before formal
        # validation/test and therefore emits an orchestrator report instead
        # of run_status.json.  Accept only the exact audited safe-stop shape;
        # every other missing-status case remains fail closed.
        report_path = source / "orchestrator_report.json"
        exit_path = source / "exit_code.txt"
        if (not report_path.is_file() or _is_reparse(report_path)
                or not exit_path.is_file() or _is_reparse(exit_path)):
            raise BackupError(
                f"missing or unsafe run_status.json/safe-stop report: {source}"
            )
        payload = _safe_json(report_path, "orchestrator_report.json")
        safe_stop = {
            "status": "STOPPED",
            "execution_status": "STOPPED",
            "result_state": "PARTIAL",
            "expected_safe_stop": True,
            "formal_result_eligible": False,
        }
        if any(payload.get(key) != value for key, value in safe_stop.items()):
            raise BackupError("orchestrator_report.json is not an exact train-only safe stop")
        try:
            # PowerShell's UTF-8 writer may include a BOM; it is encoding
            # metadata, not part of the numeric exit-code contract.
            exit_code = exit_path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError) as exc:
            raise BackupError(f"cannot read exit_code.txt: {exit_path}: {exc}") from exc
        if exit_code != "3":
            raise BackupError("train-only safe stop requires exit_code.txt == 3")
        status = "STOPPED"
    if expected is not None and status != expected:
        raise BackupError(f"run status {status} does not match --expect-status {expected}")
    return payload, status


def _check_status_gates(source: Path, status: str) -> tuple[bool, bool]:
    """Return ``(formal_result_eligible, eligibility_report_passed)``."""

    if status != "PASS":
        # A stopped or partial output can be useful for debugging, but is
        # never silently promoted to a paper result by the backup protocol.
        return False, False
    manifest_path = source / "manifest.json"
    report_path = source / "eligibility_report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise BackupError("PASS requires manifest.json and eligibility_report.json")
    manifest = _safe_json(manifest_path, "manifest.json")
    report = _safe_json(report_path, "eligibility_report.json")
    if manifest.get("formal_result_eligible") is not True:
        raise BackupError("PASS requires manifest formal_result_eligible=true")
    if report.get("passed") is not True:
        raise BackupError("PASS requires eligibility_report.passed=true")
    return True, True


def _validate_dependency_source_root(source_root: Path | str) -> Path:
    """Validate an arbitrary source directory without following links."""

    root = _absolute(source_root)
    if not root.exists() or not root.is_dir() or _is_reparse(root):
        raise BackupError(f"source root is missing or unsafe: {root}")
    _check_existing_components(root, root.parent)
    return root


def _validate_dependency_label(label: str) -> str:
    if type(label) is not str or not _DEPENDENCY_LABEL_RE.fullmatch(label):
        raise BackupError(
            "--label must be 1-128 ASCII characters matching [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return label


def _normalize_dependency_include(raw: str) -> str:
    """Return one canonical, non-escaping POSIX-style relative path."""

    if type(raw) is not str or not raw or raw != raw.strip() or "\x00" in raw:
        raise BackupError("--include must be a non-empty relative file path")
    windows = PureWindowsPath(raw)
    value = raw.replace("\\", "/")
    if value.startswith("/") or windows.drive or windows.root:
        raise BackupError(f"--include must be relative: {raw}")
    parts = value.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise BackupError(f"--include is not a canonical relative path: {raw}")
    if any(":" in part for part in parts):
        raise BackupError(f"--include contains an invalid path component: {raw}")
    normalized = PurePosixPath(*parts).as_posix()
    if normalized in {"", "."} or PurePosixPath(normalized).is_absolute():
        raise BackupError(f"--include is not a canonical relative path: {raw}")
    return normalized


def _validate_dependency_includes(source_root: Path, includes: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(_normalize_dependency_include(value) for value in includes))
    if not normalized:
        raise BackupError("at least one --include is required")
    if len(set(normalized)) != len(normalized):
        raise BackupError("--include paths must be unique after normalization")
    for relative in normalized:
        path = source_root.joinpath(*relative.split("/"))
        _check_existing_components(path, source_root)
        if not path.exists() or not path.is_file():
            raise BackupError(f"included path must be an existing regular file: {relative}")
    return normalized


def _derive_git_relative_path(source_git_root: Path | str, source_root: Path | str, relative: str) -> str:
    """Derive the canonical Git path; never trust a manifest-supplied path."""

    git_root = _absolute(source_git_root)
    source = _absolute(source_root)
    try:
        source_relative = os.path.relpath(os.fspath(source), os.fspath(git_root))
    except ValueError as exc:
        raise BackupError("source root and Git root are on different volumes") from exc
    if source_relative == os.curdir:
        source_relative = ""
    elif source_relative == os.pardir or source_relative.startswith(os.pardir + os.sep):
        raise BackupError("source root is outside its Git root")
    value = Path(source_relative, *relative.split("/")).as_posix() if source_relative else relative
    return _normalize_dependency_include(value)


def _read_git_identity(source_root: Path) -> dict[str, Any]:
    """Read the repository identity and porcelain status without writing."""

    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", os.fspath(source_root), *args],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            raise BackupError(f"cannot execute git for dependency source: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise BackupError(f"git command failed for dependency source: {detail}")
        return completed.stdout.strip()

    git_root = _absolute(run("rev-parse", "--show-toplevel"))
    head = run("rev-parse", "HEAD").lower()
    # Default porcelain excludes ignored files.  That is intentional: build
    # outputs and caches may be ignored without invalidating a dependency
    # snapshot, while tracked edits and non-ignored untracked files remain a
    # hard gate.
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    clean = status == ""
    return {
        "git_root": str(git_root),
        "head": head,
        "clean": clean,  # backward-compatible alias used by existing callers
        "tracked_worktree_clean": clean,
    }


def _require_dependency_git(source_root: Path, expected_head: str) -> dict[str, Any]:
    if type(expected_head) is not str or _GIT_HEAD_RE.fullmatch(expected_head) is None:
        raise BackupError("--expect-git-head must be a 40-character hexadecimal commit")
    identity = _read_git_identity(source_root)
    actual = identity.get("head")
    if actual != expected_head.lower():
        raise BackupError(f"git HEAD mismatch: expected {expected_head.lower()}, got {actual}")
    if identity.get("tracked_worktree_clean", identity.get("clean")) is not True:
        raise BackupError("dependency source repository has tracked or non-ignored untracked changes")
    return identity


def _git_classify_includes(
    source_root: Path,
    includes: tuple[str, ...],
    identity: dict[str, Any],
    *,
    allow_ignored_includes: bool,
) -> list[dict[str, Any]]:
    """Classify selected files without making unrelated ignored files fatal."""

    git_root = _absolute(identity["git_root"])

    # The original public test seam replaced _read_git_identity with the
    # compact {git_root, head, clean} shape.  Preserve that seam while all
    # real subprocess results use the stricter tracked_worktree_clean field.
    if "tracked_worktree_clean" not in identity:
        return [
            {
                "relative_path": relative,
                "git_relative_path": _derive_git_relative_path(identity["git_root"], source_root, relative),
                "git_classification": "tracked_clean",
                "tracked_worktree_clean": identity.get("clean") is True,
            }
            for relative in includes
        ]

    def run(*args: str, check: bool = True) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", os.fspath(git_root), *args],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            raise BackupError(f"cannot execute git for dependency classification: {exc}") from exc
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise BackupError(f"git classification command failed: {detail}")
        return completed.stdout.strip()

    records: list[dict[str, Any]] = []
    for relative in includes:
        selected = source_root / Path(*relative.split("/"))
        try:
            git_relative = _derive_git_relative_path(identity["git_root"], source_root, relative)
        except (ValueError, BackupError):
            raise BackupError(f"selected dependency is outside its Git repository: {relative}")
        tracked_output = run("ls-files", "--stage", "--", git_relative)
        tracked = bool(tracked_output)
        status_output = run("status", "--porcelain=v1", "--untracked-files=all", "--", git_relative)
        ignored_check = subprocess.run(
            ["git", "-C", os.fspath(git_root), "check-ignore", "--no-index", "--", git_relative],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        ignored = ignored_check.returncode == 0
        if tracked:
            classification = "tracked_clean" if status_output == "" else "tracked_dirty"
            tracked_worktree_clean = status_output == ""
        elif ignored:
            classification = "ignored_untracked"
            tracked_worktree_clean = False
        else:
            classification = "untracked"
            tracked_worktree_clean = False
        if classification == "tracked_dirty":
            raise BackupError(f"selected tracked dependency is modified: {relative}")
        if classification == "untracked":
            raise BackupError(f"selected dependency is non-ignored untracked: {relative}")
        if classification == "ignored_untracked" and not allow_ignored_includes:
            raise BackupError(
                f"selected dependency is ignored; pass --allow-ignored-includes explicitly: {relative}"
            )
        records.append(
            {
                "relative_path": relative,
                "git_relative_path": git_relative,
                "git_classification": classification,
                "tracked_worktree_clean": tracked_worktree_clean,
            }
        )
    return records


def _dependency_snapshot_digest(files: list[dict[str, Any]]) -> str:
    payload = {
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dependency_snapshot(
    root: Path,
    includes: tuple[str, ...],
    strict_payload: bool,
) -> dict[str, Any]:
    """Snapshot only selected source files, or exactly the selected payload."""

    _require_dependency_secure_io()
    if not root.exists() or not root.is_dir() or _is_reparse(root):
        raise BackupError(f"dependency snapshot root is missing or unsafe: {root}")
    if strict_payload:
        # Enumerate names without reading through paths, then hash every
        # selected payload file through the anchored file handle below.
        discovered: list[str] = []

        def visit(directory: Path, prefix: str) -> None:
            try:
                entries = sorted(directory.iterdir(), key=lambda item: item.name)
            except OSError as exc:
                raise BackupError(f"cannot enumerate dependency payload {directory}: {exc}") from exc
            for item in entries:
                if _is_reparse(item):
                    raise BackupError(f"symlink/reparse point is not allowed: {item}")
                rel = f"{prefix}/{item.name}" if prefix else item.name
                if item.is_dir():
                    visit(item, rel)
                elif item.is_file():
                    discovered.append(rel.replace("\\", "/"))
                else:
                    raise BackupError(f"unsupported filesystem entry: {item}")

        visit(root, "")
        expected = set(includes)
        if set(discovered) != expected:
            extra = sorted(set(discovered) - expected)
            missing = sorted(expected - set(discovered))
            raise BackupError(
                f"dependency payload file set mismatch; extra={extra}, missing={missing}"
            )
    files = []
    for relative in includes:
        path = root.joinpath(*relative.split("/"))
        _check_existing_components(path, root)
        if not path.exists() or not path.is_file():
            raise BackupError(f"included path is missing or unsafe: {relative}")
        try:
            with _secure_dependency_file(path, root) as (handle, identity):
                before = os.fstat(handle.fileno())
                if type(identity.get("link_count")) is not int or identity.get("link_count") != 1:
                    raise BackupError(f"hardlinked dependency file is not allowed: {relative}")
                digest = hashlib.sha256()
                while True:
                    block = handle.read(_CHUNK_SIZE)
                    if not block:
                        break
                    digest.update(block)
                after = os.fstat(handle.fileno())
                after_identity = _dependency_handle_identity(handle)
        except BackupError:
            raise
        except OSError as exc:
            raise BackupError(f"cannot securely hash dependency file {path}: {exc}") from exc
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise BackupError(f"dependency file changed while hashing: {path}")
        if not _dependency_identity_matches(identity, after_identity):
            raise BackupError(f"dependency file identity/link count changed while hashing: {path}")
        files.append(
            {
                "relative_path": relative,
                "bytes": int(after.st_size),
                "sha256": digest.hexdigest(),
                "link_count": identity["link_count"],
            }
        )
    return {
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
        "snapshot_digest_sha256": _dependency_snapshot_digest(files),
    }


def _copy_dependency_payload(source_root: Path, destination: Path, snapshot: dict[str, Any]) -> None:
    destination.mkdir(parents=False, exist_ok=False)
    _check_existing_components(destination, destination.parent)
    for record in snapshot["files"]:
        relative = Path(*record["relative_path"].split("/"))
        source_file = source_root / relative
        destination_file = destination / relative
        parent = destination_file.parent
        _secure_dependency_mkdirs(destination, relative.parent)
        try:
            with _secure_dependency_file(source_file, source_root) as (source_handle, source_identity):
                if type(source_identity.get("link_count")) is not int or source_identity.get("link_count") != 1:
                    raise BackupError(f"hardlinked dependency file is not allowed: {relative.as_posix()}")
                source_before_stat = os.fstat(source_handle.fileno())
                with _secure_dependency_file(
                    destination_file, destination, write=True, create_new=True
                ) as (destination_handle, destination_identity):
                    if type(destination_identity.get("link_count")) is not int or destination_identity.get("link_count") != 1:
                        raise BackupError(f"hardlinked payload file is not allowed: {relative.as_posix()}")
                    while True:
                        block = source_handle.read(_CHUNK_SIZE)
                        if not block:
                            break
                        destination_handle.write(block)
                    source_after_stat = os.fstat(source_handle.fileno())
                    if (
                        source_before_stat.st_size,
                        source_before_stat.st_mtime_ns,
                    ) != (
                        source_after_stat.st_size,
                        source_after_stat.st_mtime_ns,
                    ) or not _dependency_identity_matches(
                        source_identity,
                        _dependency_handle_identity(source_handle),
                    ):
                        raise BackupError(
                            f"source file changed or identity/link count drifted during copy: {relative.as_posix()}"
                        )
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
                    if not _dependency_identity_matches(
                        destination_identity,
                        _dependency_handle_identity(destination_handle),
                    ):
                        raise BackupError(
                            f"payload file identity/link count changed: {relative.as_posix()}"
                        )
                    _fsync_directory(parent, strict=True)
        except BackupError:
            raise
        except OSError as exc:
            raise BackupError(f"cannot securely copy {source_file} to {destination_file}: {exc}") from exc
    _fsync_directory(destination, strict=True)


def _dependency_payload_identities(root: Path, includes: tuple[str, ...]) -> list[dict[str, Any]]:
    """Return identity records for the finalized payload files."""

    records: list[dict[str, Any]] = []
    for relative in includes:
        path = root.joinpath(*relative.split("/"))
        with _secure_dependency_file(path, root) as (handle, identity):
            stat_result = os.fstat(handle.fileno())
            if type(identity.get("link_count")) is not int or identity["link_count"] != 1:
                raise BackupError(f"hardlinked payload file is not allowed: {relative}")
            record: dict[str, Any] = {
                "relative_path": relative,
                "link_count": int(identity["link_count"]),
                "bytes": int(stat_result.st_size),
                "mtime_ns": int(stat_result.st_mtime_ns),
            }
            handle.seek(0)
            digest = hashlib.sha256()
            while True:
                block = handle.read(_CHUNK_SIZE)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(handle.fileno())
            if (stat_result.st_size, stat_result.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise BackupError(f"payload file changed while recording identity: {relative}")
            record["sha256"] = digest.hexdigest()
            for key in ("volume_serial", "file_index"):
                if key in identity:
                    record[key] = int(identity[key])
            records.append(record)
    return records


def _copy_dependency_payload_handles(
    lockset: _WindowsDependencyLockSet,
    destination: Path,
    snapshot: dict[str, Any],
) -> None:
    """Copy from the already-open source handles held by ``lockset``."""

    destination.mkdir(parents=False, exist_ok=False)
    _check_existing_components(destination, destination.parent)
    expected = {item["relative_path"]: item for item in snapshot["files"]}
    for relative in lockset.includes:
        record = expected[relative]
        relative_path = Path(*relative.split("/"))
        destination_file = destination / relative_path
        _secure_dependency_mkdirs(destination, relative_path.parent)
        source_handle = lockset.file_handles[relative][0]
        try:
            source_handle.seek(0)
            source_before = os.fstat(source_handle.fileno())
            source_before_identity = lockset.refresh_file_info(relative)
            if type(source_before_identity.get("link_count")) is not int or source_before_identity["link_count"] != 1:
                raise BackupError(f"hardlinked dependency file is not allowed: {relative}")
            with _secure_dependency_file(
                destination_file, destination, write=True, create_new=True
            ) as (destination_handle, destination_identity):
                if type(destination_identity.get("link_count")) is not int or destination_identity["link_count"] != 1:
                    raise BackupError(f"hardlinked payload file is not allowed: {relative}")
                digest = hashlib.sha256()
                copied = 0
                while True:
                    block = source_handle.read(_CHUNK_SIZE)
                    if not block:
                        break
                    digest.update(block)
                    destination_handle.write(block)
                    copied += len(block)
                source_after = os.fstat(source_handle.fileno())
                source_after_identity = lockset.refresh_file_info(relative)
                if (
                    (source_before.st_size, source_before.st_mtime_ns)
                    != (source_after.st_size, source_after.st_mtime_ns)
                    or not _dependency_identity_matches(source_before_identity, source_after_identity)
                ):
                    raise BackupError(f"source file changed or identity/link count drifted during copy: {relative}")
                if copied != record["bytes"] or digest.hexdigest() != record["sha256"]:
                    raise BackupError(f"locked source content drifted during copy: {relative}")
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
                destination_stat = os.fstat(destination_handle.fileno())
                if destination_stat.st_size != copied:
                    raise BackupError(f"payload byte count mismatch after copy: {relative}")
                if not _dependency_identity_matches(
                    destination_identity,
                    _dependency_handle_identity(destination_handle),
                ):
                    raise BackupError(f"payload file identity/link count changed: {relative}")
                _fsync_directory(destination_file.parent, strict=True)
        except BackupError:
            raise
        except (OSError, ValueError) as exc:
            raise BackupError(f"cannot securely copy locked dependency file {relative}: {exc}") from exc
    _fsync_directory(destination, strict=True)


def _attach_dependency_git_metadata(
    snapshot: dict[str, Any], git_records: list[dict[str, Any]]
) -> dict[str, Any]:
    by_path = {item["relative_path"]: item for item in git_records}
    files: list[dict[str, Any]] = []
    for item in snapshot.get("files", []):
        path = item.get("relative_path")
        if path not in by_path:
            raise BackupError(f"dependency snapshot omitted Git classification: {path}")
        copied = dict(item)
        copied.update(
            git_classification=by_path[path]["git_classification"],
            tracked_worktree_clean=by_path[path]["tracked_worktree_clean"],
        )
        files.append(copied)
    if len(files) != len(git_records):
        raise BackupError("dependency snapshot and Git classification file counts differ")
    result = dict(snapshot)
    result["files"] = files
    result["snapshot_digest_sha256"] = _dependency_snapshot_digest(files)
    return result


def _hash_file(path: Path) -> tuple[int, str]:
    try:
        before = os.stat(path, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or (getattr(before, "st_file_attributes", 0) & _REPARSE_POINT):
            raise BackupError(f"symlink/reparse point is not allowed: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(_CHUNK_SIZE)
                if not block:
                    break
                digest.update(block)
        after = os.stat(path, follow_symlinks=False)
    except BackupError:
        raise
    except OSError as exc:
        raise BackupError(f"cannot hash file {path}: {exc}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise BackupError(f"source file changed while hashing: {path}")
    return int(after.st_size), digest.hexdigest()


def _snapshot_tree(root: Path) -> dict[str, Any]:
    """Capture relative file names, byte counts, and SHA-256 hashes."""

    if not root.exists() or not root.is_dir() or _is_reparse(root):
        raise BackupError(f"snapshot root is missing or unsafe: {root}")
    files: list[dict[str, Any]] = []

    def visit(directory: Path, relative_prefix: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise BackupError(f"cannot enumerate {directory}: {exc}") from exc
        for item in entries:
            if _is_reparse(item):
                raise BackupError(f"symlink/reparse point is not allowed: {item}")
            relative = relative_prefix / item.name
            if item.is_dir():
                visit(item, relative)
            elif item.is_file():
                size, sha256 = _hash_file(item)
                files.append({"path": relative.as_posix(), "bytes": size, "sha256": sha256})
            else:
                raise BackupError(f"unsupported filesystem entry: {item}")

    visit(root, Path())
    return {
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def _snapshot_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left == right


def _existing_usage_path(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _space_warning(backup_root: Path, source_bytes: int) -> dict[str, Any] | None:
    usage_path = _existing_usage_path(backup_root)
    try:
        usage = shutil.disk_usage(usage_path)
    except OSError as exc:
        raise BackupError(f"cannot inspect free space at {usage_path}: {exc}") from exc
    required = source_bytes + _TWO_GIB
    if int(usage.free) < required:
        raise BackupError(
            f"insufficient free space: need at least {required} bytes, have {usage.free}"
        )
    low_bytes = int(usage.free) < _TWENTY_GIB
    low_percent = int(usage.free) * 100 < int(usage.total) * 15
    if low_bytes or low_percent:
        return {
            "free_bytes": int(usage.free),
            "total_bytes": int(usage.total),
            "required_bytes": required,
            "below_20_gib": low_bytes,
            "below_15_percent": low_percent,
        }
    return None


def _fsync_file(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError as exc:
        raise BackupError(f"fsync failed for {path}: {exc}") from exc


def _fsync_directory(path: Path, *, strict: bool = True) -> None:
    """Durability barrier for a directory; unsupported/error is fatal."""

    if os.name == "nt":
        handle = _windows_open_handle(path, directory=True, writable=strict)
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
            kernel32.FlushFileBuffers.restype = wintypes.BOOL
            if not kernel32.FlushFileBuffers(wintypes.HANDLE(handle)):
                if not strict:
                    return
                raise BackupError(f"FlushFileBuffers failed for {path}: WinError {ctypes.get_last_error()}")
        finally:
            _windows_close_handle(handle)
        return
    try:
        descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        if not strict:
            return
        raise BackupError(f"directory fsync failed for {path}: {exc}") from exc


def _write_bytes_fsync(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise BackupError(f"refusing to overwrite existing file: {path}") from exc
    except OSError as exc:
        raise BackupError(f"cannot write {path}: {exc}") from exc


def _copy_payload(source: Path, destination: Path, source_snapshot: dict[str, Any]) -> None:
    destination.mkdir(parents=False, exist_ok=False)
    for record in source_snapshot["files"]:
        relative = Path(record["path"])
        source_file = source / relative
        destination_file = destination / relative
        # Check each parent before creating it; a reparse point must never be
        # traversed even if one appears during the copy.
        parent = destination_file.parent
        parent.mkdir(parents=True, exist_ok=True)
        _check_existing_components(parent, destination)
        if _is_reparse(source_file):
            raise BackupError(f"source changed to symlink/reparse point: {source_file}")
        try:
            with source_file.open("rb") as source_handle, destination_file.open("xb") as destination_handle:
                while True:
                    block = source_handle.read(_CHUNK_SIZE)
                    if not block:
                        break
                    destination_handle.write(block)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())
        except FileExistsError as exc:
            raise BackupError(f"refusing to overwrite destination file: {destination_file}") from exc
        except OSError as exc:
            raise BackupError(f"cannot copy {source_file} to {destination_file}: {exc}") from exc


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _backup_manifest_bytes(manifest_path: Path) -> bytes:
    try:
        data = manifest_path.read_bytes()
    except OSError as exc:
        raise BackupError(f"cannot read backup manifest: {manifest_path}: {exc}") from exc
    return data


def _write_manifest(staging: Path, manifest: dict[str, Any], *, strict_directory: bool = True) -> None:
    manifest_path = staging / "backup_manifest.json"
    manifest_bytes = _json_bytes(manifest)
    _write_bytes_fsync(manifest_path, manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    _write_bytes_fsync(staging / "backup_manifest.sha256", (digest + "\n").encode("ascii"))
    _fsync_directory(staging, strict=strict_directory)


def _read_manifest(backup_dir: Path) -> dict[str, Any]:
    manifest_path = backup_dir / "backup_manifest.json"
    sidecar_path = backup_dir / "backup_manifest.sha256"
    if (
        not manifest_path.is_file()
        or _is_reparse(manifest_path)
        or not sidecar_path.is_file()
        or _is_reparse(sidecar_path)
    ):
        raise BackupError("backup_manifest.json and backup_manifest.sha256 are required")
    manifest_bytes = _backup_manifest_bytes(manifest_path)
    try:
        sidecar = sidecar_path.read_bytes().decode("ascii")
    except (OSError, UnicodeError) as exc:
        raise BackupError(f"invalid backup manifest sidecar: {sidecar_path}") from exc
    if re.fullmatch(r"[0-9a-f]{64}\n", sidecar) is None:
        raise BackupError("backup_manifest.sha256 must be exactly one lowercase SHA-256 line")
    expected = sidecar[:-1]
    actual = hashlib.sha256(manifest_bytes).hexdigest()
    if expected != actual:
        raise BackupError("backup_manifest.sha256 does not match backup_manifest.json")
    try:
        value = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BackupError(f"invalid backup_manifest.json: {manifest_path}") from exc
    if not isinstance(value, dict) or value.get("schema") != _BACKUP_SCHEMA:
        raise BackupError("unsupported or malformed backup manifest schema")
    return value


def _requested_final_path(backup_root: Path, backup_dir: Path | str | None, run_id: str) -> Path:
    if backup_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        name = f"{run_id}_{stamp}_{uuid.uuid4().hex}.finalized"
        return backup_root / name
    candidate = _absolute(backup_dir) if Path(backup_dir).is_absolute() else backup_root / Path(backup_dir)
    _path_parts_from_root(candidate, backup_root)
    if candidate.name.endswith(".partial"):
        candidate = candidate.with_name(candidate.name[:-len(".partial")] + ".finalized")
    elif not candidate.name.endswith(".finalized"):
        candidate = candidate.with_name(candidate.name + ".finalized")
    return candidate


def _rename_noreplace(staging: Path, finalized: Path) -> None:
    if finalized.exists() or finalized.is_symlink():
        raise BackupError(f"refusing to overwrite existing backup: {finalized}")
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        kernel32.MoveFileExW.restype = wintypes.BOOL
        # No MOVEFILE_REPLACE_EXISTING flag: the kernel performs an atomic
        # no-replace rename and reports a collision instead of overwriting.
        if not kernel32.MoveFileExW(
            os.fspath(staging), os.fspath(finalized), _WIN_MOVEFILE_WRITE_THROUGH
        ):
            raise BackupError(
                f"cannot finalize backup {staging} -> {finalized}: WinError {ctypes.get_last_error()}"
            )
    else:
        # rename(2) replaces, so use Linux renameat2(RENAME_NOREPLACE) when
        # available and fail closed everywhere else.
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = libc.renameat2
        except (AttributeError, OSError) as exc:
            raise BackupError("atomic no-replace rename is unsupported on this platform") from exc
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        at_fdcwd = getattr(os, "AT_FDCWD", -100)
        result = renameat2(
            at_fdcwd,
            os.fsencode(staging),
            at_fdcwd,
            os.fsencode(finalized),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error = ctypes.get_errno()
            raise BackupError(
                f"cannot finalize backup {staging} -> {finalized}: errno {error} ({os.strerror(error)})"
            )
    if not finalized.exists():
        raise BackupError("backup finalization did not produce the destination")


def _prepare_source(
    source: Path | str | None,
    run_id: str | None,
    expected_status: str | None,
) -> tuple[Path, dict[str, Any], str, bool, bool, dict[str, Any]]:
    if source is None:
        if run_id is None:
            raise BackupError("one of --source or --run-id is required")
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
            raise BackupError("--run-id must be one directory name, not a path")
        source = _absolute(OUTPUTS_ROOT) / run_id
    source_path = _validate_source(Path(source))
    run_status, status = _read_run_status(source_path, expected_status)
    formal, report_passed = _check_status_gates(source_path, status)
    snapshot = _snapshot_tree(source_path)
    return source_path, run_status, status, formal, report_passed, snapshot


def dry_run(
    *,
    source: Path | str | None = None,
    run_id: str | None = None,
    expect_status: str | None = None,
    backup_root: Path | str = DEFAULT_BACKUP_ROOT,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Validate a prospective backup without writing any file or directory."""

    root = _validate_backup_root(backup_root)
    source_path, run_status, status, formal, report_passed, snapshot = _prepare_source(
        source, run_id, expect_status
    )
    if max_bytes is not None and (max_bytes < 0 or snapshot["total_bytes"] > max_bytes):
        raise BackupError("source size exceeds --max-bytes")
    warning = _space_warning(root, snapshot["total_bytes"])
    return {
        "action": "dry-run",
        "source": str(source_path),
        "run_status": run_status,
        "status": status,
        "formal_result_eligible": formal,
        "eligibility_report_passed": report_passed,
        "source_snapshot": snapshot,
        "space_warning": warning,
        "would_write": False,
    }


def create_backup(
    *,
    source: Path | str | None = None,
    run_id: str | None = None,
    expect_status: str | None = None,
    backup_root: Path | str = DEFAULT_BACKUP_ROOT,
    backup_dir: Path | str | None = None,
    max_bytes: int | None = None,
    _snapshotter: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create and finalize one immutable backup directory."""

    root = _validate_backup_root(backup_root)
    source_path, run_status, status, formal, report_passed, source_before = _prepare_source(
        source, run_id, expect_status
    )
    if max_bytes is not None and (max_bytes < 0 or source_before["total_bytes"] > max_bytes):
        raise BackupError("source size exceeds --max-bytes")
    warning = _space_warning(root, source_before["total_bytes"])
    if not root.exists():
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        except OSError as exc:
            raise BackupError(f"cannot create backup root {root}: {exc}") from exc
        _check_backup_path(root, root, must_exist=True)

    finalized = _requested_final_path(root, backup_dir, source_path.name)
    _check_backup_path(finalized, root)
    if finalized.exists() or finalized.is_symlink():
        raise BackupError(f"refusing to overwrite existing backup: {finalized}")
    staging = finalized.with_name(finalized.name[:-len(".finalized")] + ".partial")
    if staging.exists() or staging.is_symlink():
        raise BackupError(f"staging backup already exists; refusing to reuse it: {staging}")
    try:
        staging.mkdir(parents=False, exist_ok=False)
        _check_existing_components(staging, root)
        payload = staging / "payload"
        _copy_payload(source_path, payload, source_before)
        snap = _snapshotter or _snapshot_tree
        payload_after = snap(payload)
        source_after = snap(source_path)
        if not _snapshot_equal(source_before, source_after):
            raise BackupError("source changed during backup; refusing to finalize")
        if not _snapshot_equal(source_before, payload_after):
            raise BackupError("copied payload does not match source; refusing to finalize")
        manifest = {
            "schema": _BACKUP_SCHEMA,
            "schema_version": 1,
            "backup_id": finalized.name.removesuffix(".finalized"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(source_path),
            "source_relative_to_outputs": source_path.relative_to(_absolute(OUTPUTS_ROOT)).as_posix(),
            "run_id": source_path.name,
            "status": status,
            "formal_result_eligible": formal if status == "PASS" else False,
            "eligibility_report_passed": report_passed if status == "PASS" else False,
            "source_before": source_before,
            "source_after": source_after,
            "payload_after": payload_after,
            "space_warning": warning,
            "run_status": run_status,
        }
        _write_manifest(staging, manifest)
        _fsync_directory(staging)
        _rename_noreplace(staging, finalized)
        _fsync_directory(root)
    except Exception as exc:
        # Failures preserve the incomplete staging directory as evidence.
        # Deletion or cleanup is a separate, explicitly reviewed main-window
        # action; this utility never removes experiment or backup artifacts.
        if staging.exists() and not finalized.exists():
            raise BackupError(
                f"{exc}; incomplete staging retained for main-window review: {staging}"
            ) from exc
        raise
    return {
        "action": "create",
        "backup_dir": str(finalized),
        "manifest": str(finalized / "backup_manifest.json"),
        "status": status,
        "formal_result_eligible": manifest["formal_result_eligible"],
        "space_warning": warning,
    }


def _prepare_dependency_metadata(
    source_root: Path | str,
    includes: Iterable[str],
    label: str,
    expected_head: str,
    *,
    allow_ignored_includes: bool = False,
) -> tuple[Path, tuple[str, ...], str, dict[str, Any], list[dict[str, Any]]]:
    root = _validate_dependency_source_root(source_root)
    normalized = _validate_dependency_includes(root, includes)
    checked_label = _validate_dependency_label(label)
    identity = _require_dependency_git(root, expected_head)
    git_records = _git_classify_includes(
        root,
        normalized,
        identity,
        allow_ignored_includes=allow_ignored_includes,
    )
    return root, normalized, checked_label, identity, git_records


def _prepare_dependency_source(
    source_root: Path | str,
    includes: Iterable[str],
    label: str,
    expected_head: str,
    *,
    allow_ignored_includes: bool = False,
    snapshotter: Callable[[Path, tuple[str, ...], bool], dict[str, Any]] | None = None,
) -> tuple[Path, tuple[str, ...], str, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root, normalized, checked_label, identity, git_records = _prepare_dependency_metadata(
        source_root,
        includes,
        label,
        expected_head,
        allow_ignored_includes=allow_ignored_includes,
    )
    snap = snapshotter or _dependency_snapshot
    source_before = snap(root, normalized, False)
    source_before = _attach_dependency_git_metadata(source_before, git_records)
    return root, normalized, checked_label, identity, git_records, source_before


def _dependency_create_windows_locked(
    *,
    source_path: Path,
    normalized: tuple[str, ...],
    checked_label: str,
    identity: dict[str, Any],
    git_records: list[dict[str, Any]],
    root: Path,
    backup_dir: Path | str | None,
    max_bytes: int | None,
    allow_ignored_includes: bool,
) -> dict[str, Any]:
    """Run the production Windows dependency pipeline under one lock set."""

    if os.name != "nt":
        raise BackupError(
            "BLOCKED: dependency-create requires the Windows handle-backed lock pipeline"
        )
    if not root.exists():
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        except OSError as exc:
            raise BackupError(f"cannot create backup root {root}: {exc}") from exc
        _check_backup_path(root, root, must_exist=True)
    finalized = _requested_final_path(root, backup_dir, checked_label)
    _check_backup_path(finalized, root)
    if finalized.exists() or finalized.is_symlink():
        raise BackupError(f"refusing to overwrite existing backup: {finalized}")
    staging = finalized.with_name(finalized.name[:-len(".finalized")] + ".partial")
    if staging.exists() or staging.is_symlink():
        raise BackupError(f"staging backup already exists; refusing to reuse it: {staging}")

    try:
        with _WindowsDependencyLockSet(source_path, normalized) as lockset:
            source_before_core, source_identity_before = lockset.snapshot()
            source_before = _attach_dependency_git_metadata(source_before_core, git_records)
            if max_bytes is not None and (max_bytes < 0 or source_before["total_bytes"] > max_bytes):
                raise BackupError("selected dependency size exceeds --max-bytes")
            warning = _space_warning(root, source_before["total_bytes"])
            staging.mkdir(parents=False, exist_ok=False)
            _check_existing_components(staging, root)
            payload = staging / "payload"
            _copy_dependency_payload_handles(lockset, payload, source_before_core)
            payload_after = _attach_dependency_git_metadata(
                _dependency_snapshot(payload, normalized, True), git_records
            )
            payload_identity = _dependency_payload_identities(payload, normalized)
            source_after_core, source_identity_after = lockset.snapshot()
            source_after = _attach_dependency_git_metadata(source_after_core, git_records)
            if not (source_before == source_after == payload_after):
                raise BackupError(
                    "dependency source/payload content snapshots disagree; refusing to finalize"
                )
            if source_identity_before != source_identity_after:
                raise BackupError("dependency source handle identity changed during backup")
            identity_after = _require_dependency_git(source_path, identity["head"])
            if identity_after != identity:
                raise BackupError("dependency source Git identity changed during backup")
            manifest = {
                "schema": _BACKUP_SCHEMA,
                "schema_version": 1,
                "backup_id": finalized.name.removesuffix(".finalized"),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "backup_kind": _DEPENDENCY_BACKUP_KIND,
                "status": "PARTIAL",
                "formal_result_eligible": False,
                "paper_table_eligible": False,
                "eligibility_report_passed": False,
                "label": checked_label,
                "source_root": str(source_path),
                "source_git_root": identity["git_root"],
                "source_commit": identity["head"],
                "expect_git_head": identity["head"].lower(),
                "source_git_clean": True,
                "tracked_worktree_clean": identity["tracked_worktree_clean"] is True,
                "allow_ignored_includes": bool(allow_ignored_includes),
                "include": list(normalized),
                "git_files": git_records,
                "source_before": source_before,
                "source_after": source_after,
                "payload_after": payload_after,
                "handle_backed": True,
                "source_identity_before": source_identity_before,
                "source_identity_after": source_identity_after,
                "payload_identity": payload_identity,
                "space_warning": warning,
            }
            _write_manifest(staging, manifest, strict_directory=True)
            _fsync_directory(staging, strict=True)
            _rename_noreplace(staging, finalized)
            _fsync_directory(root, strict=True)
    except Exception as exc:
        if staging.exists() and not finalized.exists():
            raise BackupError(
                f"{exc}; incomplete staging retained for main-window review: {staging}"
            ) from exc
        raise
    return {
        "action": "dependency-create",
        "backup_dir": str(finalized),
        "manifest": str(finalized / "backup_manifest.json"),
        "status": "PARTIAL",
        "backup_kind": _DEPENDENCY_BACKUP_KIND,
        "formal_result_eligible": False,
        "paper_table_eligible": False,
        "space_warning": warning,
    }


def dependency_dry_run(
    *,
    source_root: Path | str,
    includes: Iterable[str],
    label: str,
    expect_git_head: str,
    backup_root: Path | str = DEFAULT_BACKUP_ROOT,
    max_bytes: int | None = None,
    allow_ignored_includes: bool = False,
    _snapshotter: Callable[[Path, tuple[str, ...], bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate an immutable verified-dependency backup without writing."""

    root = _validate_backup_root(backup_root)
    source_path, normalized, checked_label, identity, git_records, source_before = _prepare_dependency_source(
        source_root,
        includes,
        label,
        expect_git_head,
        allow_ignored_includes=allow_ignored_includes,
        snapshotter=_snapshotter,
    )
    if max_bytes is not None and (max_bytes < 0 or source_before["total_bytes"] > max_bytes):
        raise BackupError("selected dependency size exceeds --max-bytes")
    warning = _space_warning(root, source_before["total_bytes"])
    return {
        "action": "dependency-dry-run",
        "source_root": str(source_path),
        "source_git_root": identity["git_root"],
        "source_commit": identity["head"],
        "source_git_clean": True,
        "tracked_worktree_clean": identity.get("tracked_worktree_clean", identity.get("clean")) is True,
        "allow_ignored_includes": bool(allow_ignored_includes),
        "git_files": git_records,
        "label": checked_label,
        "include": list(normalized),
        "status": "PARTIAL",
        "backup_kind": _DEPENDENCY_BACKUP_KIND,
        "formal_result_eligible": False,
        "paper_table_eligible": False,
        "eligibility_report_passed": False,
        "source_snapshot": source_before,
        "space_warning": warning,
        "would_write": False,
    }


def dependency_create(
    *,
    source_root: Path | str,
    includes: Iterable[str],
    label: str,
    expect_git_head: str,
    backup_root: Path | str = DEFAULT_BACKUP_ROOT,
    backup_dir: Path | str | None = None,
    max_bytes: int | None = None,
    allow_ignored_includes: bool = False,
    _snapshotter: Callable[[Path, tuple[str, ...], bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create one immutable, non-paper-eligible dependency backup."""

    root = _validate_backup_root(backup_root)
    metadata = _prepare_dependency_metadata(
        source_root,
        includes,
        label,
        expect_git_head,
        allow_ignored_includes=allow_ignored_includes,
    )
    source_path, normalized, checked_label, identity, git_records = metadata
    if "tracked_worktree_clean" in identity:
        return _dependency_create_windows_locked(
            source_path=source_path,
            normalized=normalized,
            checked_label=checked_label,
            identity=identity,
            git_records=git_records,
            root=root,
            backup_dir=backup_dir,
            max_bytes=max_bytes,
            allow_ignored_includes=allow_ignored_includes,
        )
    snap = _snapshotter or _dependency_snapshot
    source_before = _attach_dependency_git_metadata(
        snap(source_path, normalized, False), git_records
    )
    if max_bytes is not None and (max_bytes < 0 or source_before["total_bytes"] > max_bytes):
        raise BackupError("selected dependency size exceeds --max-bytes")
    warning = _space_warning(root, source_before["total_bytes"])
    if not root.exists():
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        except OSError as exc:
            raise BackupError(f"cannot create backup root {root}: {exc}") from exc
        _check_backup_path(root, root, must_exist=True)

    finalized = _requested_final_path(root, backup_dir, checked_label)
    _check_backup_path(finalized, root)
    if finalized.exists() or finalized.is_symlink():
        raise BackupError(f"refusing to overwrite existing backup: {finalized}")
    staging = finalized.with_name(finalized.name[:-len(".finalized")] + ".partial")
    if staging.exists() or staging.is_symlink():
        raise BackupError(f"staging backup already exists; refusing to reuse it: {staging}")
    try:
        staging.mkdir(parents=False, exist_ok=False)
        _check_existing_components(staging, root)
        payload = staging / "payload"
        _copy_dependency_payload(source_path, payload, source_before)
        payload_after = _attach_dependency_git_metadata(
            snap(payload, normalized, True), git_records
        )
        source_after = _attach_dependency_git_metadata(
            snap(source_path, normalized, False), git_records
        )
        if not (source_before == source_after == payload_after):
            raise BackupError(
                "dependency source/payload snapshots disagree; refusing to finalize"
            )
        identity_after = _require_dependency_git(source_path, expect_git_head)
        if identity_after != identity:
            raise BackupError("dependency source Git identity changed during backup")
        manifest = {
            "schema": _BACKUP_SCHEMA,
            "schema_version": 1,
            "backup_id": finalized.name.removesuffix(".finalized"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "backup_kind": _DEPENDENCY_BACKUP_KIND,
            "status": "PARTIAL",
            "formal_result_eligible": False,
            "paper_table_eligible": False,
            "eligibility_report_passed": False,
            "label": checked_label,
            "source_root": str(source_path),
            "source_git_root": identity["git_root"],
            "source_commit": identity["head"],
            "expect_git_head": expect_git_head.lower(),
            "source_git_clean": True,
            "tracked_worktree_clean": identity.get("tracked_worktree_clean", identity.get("clean")) is True,
            "allow_ignored_includes": bool(allow_ignored_includes),
            "include": list(normalized),
            "git_files": git_records,
            "source_before": source_before,
            "source_after": source_after,
            "payload_after": payload_after,
            "handle_backed": False,
            "space_warning": warning,
        }
        _write_manifest(staging, manifest, strict_directory=True)
        _fsync_directory(staging, strict=True)
        _rename_noreplace(staging, finalized)
        _fsync_directory(root, strict=True)
    except Exception as exc:
        if staging.exists() and not finalized.exists():
            raise BackupError(
                f"{exc}; incomplete staging retained for main-window review: {staging}"
            ) from exc
        raise
    return {
        "action": "dependency-create",
        "backup_dir": str(finalized),
        "manifest": str(finalized / "backup_manifest.json"),
        "status": "PARTIAL",
        "backup_kind": _DEPENDENCY_BACKUP_KIND,
        "formal_result_eligible": False,
        "paper_table_eligible": False,
        "space_warning": warning,
    }


def _verify_dependency_backup(directory: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "backup_id",
        "created_utc",
        "backup_kind",
        "status",
        "formal_result_eligible",
        "paper_table_eligible",
        "eligibility_report_passed",
        "label",
        "source_root",
        "source_git_root",
        "source_commit",
        "expect_git_head",
        "source_git_clean",
        "tracked_worktree_clean",
        "allow_ignored_includes",
        "include",
        "git_files",
        "source_before",
        "source_after",
        "payload_after",
        "handle_backed",
        "space_warning",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise BackupError(f"verified-dependency manifest is missing required fields: {missing}")
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1:
        raise BackupError("verified-dependency schema_version must be exactly 1")
    if type(manifest.get("backup_kind")) is not str or manifest.get("backup_kind") != _DEPENDENCY_BACKUP_KIND:
        raise BackupError("not a verified-dependency backup")
    if type(manifest.get("status")) is not str or manifest.get("status") != "PARTIAL":
        raise BackupError("verified-dependency backup must have status PARTIAL")
    if any(manifest.get(key) is not False for key in ("formal_result_eligible", "paper_table_eligible", "eligibility_report_passed")):
        raise BackupError("verified-dependency backup must be explicitly non-paper-eligible")
    for key in ("backup_id", "created_utc", "source_root", "source_git_root"):
        if type(manifest.get(key)) is not str or not manifest[key]:
            raise BackupError(f"verified-dependency manifest field {key} must be a non-empty string")
    if _BACKUP_ID_RE.fullmatch(manifest["backup_id"]) is None:
        raise BackupError("verified-dependency backup_id has invalid format")
    try:
        created = datetime.fromisoformat(manifest["created_utc"])
    except ValueError as exc:
        raise BackupError("verified-dependency created_utc is not ISO-8601") from exc
    if created.tzinfo is None or created.utcoffset() != timezone.utc.utcoffset(created):
        raise BackupError("verified-dependency created_utc must include UTC offset")
    source_root = manifest["source_root"]
    source_git_root = manifest["source_git_root"]
    if (
        "\x00" in source_root
        or "\x00" in source_git_root
        or not Path(source_root).is_absolute()
        or not Path(source_git_root).is_absolute()
    ):
        raise BackupError("verified-dependency source roots must be absolute paths")
    source_commit = manifest["source_commit"]
    expected_head = manifest["expect_git_head"]
    if type(source_commit) is not str or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise BackupError("verified-dependency manifest has an invalid source_commit")
    if type(expected_head) is not str or re.fullmatch(r"[0-9a-f]{40}", expected_head) is None:
        raise BackupError("verified-dependency manifest has an invalid expect_git_head")
    if source_commit.lower() != expected_head.lower():
        raise BackupError("verified-dependency source_commit does not match expect_git_head")
    if manifest.get("source_git_clean") is not True or manifest.get("tracked_worktree_clean") is not True:
        raise BackupError("verified-dependency source Git worktree must be clean")
    if type(manifest.get("handle_backed")) is not bool:
        raise BackupError("verified-dependency handle_backed must be boolean")
    if type(manifest.get("allow_ignored_includes")) is not bool:
        raise BackupError("verified-dependency allow_ignored_includes must be boolean")
    if manifest.get("space_warning") is not None and type(manifest.get("space_warning")) is not dict:
        raise BackupError("verified-dependency space_warning must be an object or null")
    _validate_dependency_label(manifest["label"])
    includes = manifest["include"]
    if type(includes) is not list or not includes:
        raise BackupError("verified-dependency manifest is missing include")
    normalized = _validate_dependency_includes_for_manifest(includes)
    if includes != list(normalized):
        raise BackupError("verified-dependency include paths are not in canonical order")
    git_files = manifest["git_files"]
    if type(git_files) is not list or len(git_files) != len(includes):
        raise BackupError("verified-dependency git_files must list every include")
    for item, relative in zip(git_files, includes):
        if type(item) is not dict or set(item) != {"relative_path", "git_relative_path", "git_classification", "tracked_worktree_clean"} or item.get("relative_path") != relative:
            raise BackupError("verified-dependency git_files path mismatch")
        if type(item["git_relative_path"]) is not str or not item["git_relative_path"]:
            raise BackupError("verified-dependency git_files Git path is invalid")
        expected_git_path = _derive_git_relative_path(source_git_root, source_root, relative)
        if item["git_relative_path"] != expected_git_path:
            raise BackupError("verified-dependency git_files Git path is not derivable from source boundary")
        if item.get("git_classification") not in {"tracked_clean", "ignored_untracked"}:
            raise BackupError("verified-dependency git_files has an invalid classification")
        if item.get("git_classification") == "ignored_untracked" and not manifest["allow_ignored_includes"]:
            raise BackupError("ignored include was recorded without explicit allow")
        if item.get("tracked_worktree_clean") is not (item.get("git_classification") == "tracked_clean"):
            raise BackupError("verified-dependency git_files tracked_worktree_clean mismatch")
    snapshots: list[dict[str, Any]] = []
    for key in ("source_before", "source_after", "payload_after"):
        value = manifest[key]
        if type(value) is not dict:
            raise BackupError(f"verified-dependency {key} must be an object")
        if set(value) != {"file_count", "total_bytes", "files", "snapshot_digest_sha256"}:
            raise BackupError(f"verified-dependency {key} has missing or extra fields")
        if type(value["file_count"]) is not int or value["file_count"] != len(includes):
            raise BackupError(f"verified-dependency {key} file_count is invalid")
        if type(value["total_bytes"]) is not int or value["total_bytes"] < 0:
            raise BackupError(f"verified-dependency {key} total_bytes is invalid")
        if type(value["snapshot_digest_sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", value["snapshot_digest_sha256"]) is None:
            raise BackupError(f"verified-dependency {key} digest is invalid")
        if type(value["files"]) is not list or len(value["files"]) != len(includes):
            raise BackupError(f"verified-dependency {key} files is invalid")
        for index, (item, relative) in enumerate(zip(value["files"], includes)):
            if type(item) is not dict or set(item) != {"relative_path", "bytes", "sha256", "link_count", "git_classification", "tracked_worktree_clean"}:
                raise BackupError(f"verified-dependency {key} file record is malformed")
            if type(item["relative_path"]) is not str or item["relative_path"] != relative or type(item["bytes"]) is not int or item["bytes"] < 0:
                raise BackupError(f"verified-dependency {key} file path/size is invalid")
            if re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None:
                raise BackupError(f"verified-dependency {key} file hash is invalid")
            if type(item["link_count"]) is not int or item["link_count"] != 1:
                raise BackupError(f"verified-dependency {key} contains a hardlinked file")
            if type(item["git_classification"]) is not str or item["git_classification"] not in {"tracked_clean", "ignored_untracked"} or type(item["tracked_worktree_clean"]) is not bool:
                raise BackupError(f"verified-dependency {key} Git metadata is invalid")
            git_record = git_files[index]
            if (
                item["git_classification"] != git_record["git_classification"]
                or item["tracked_worktree_clean"] != git_record["tracked_worktree_clean"]
            ):
                raise BackupError(f"verified-dependency {key} Git metadata does not match git_files")
        if sum(item["bytes"] for item in value["files"]) != value["total_bytes"]:
            raise BackupError(f"verified-dependency {key} byte total does not match file records")
        if _dependency_snapshot_digest(value["files"]) != value["snapshot_digest_sha256"]:
            raise BackupError(f"verified-dependency {key} digest does not match file records")
        snapshots.append(value)
    source_before, source_after, manifest_payload = snapshots
    if not (source_before == source_after == manifest_payload):
        raise BackupError("verified-dependency manifest snapshots are inconsistent")
    payload = directory / "payload"
    payload_snapshot = _dependency_snapshot(payload, normalized, strict_payload=True)
    payload_snapshot = _attach_dependency_git_metadata(
        payload_snapshot,
        [
            {
                "relative_path": item["relative_path"],
                "git_classification": item["git_classification"],
                "tracked_worktree_clean": item["tracked_worktree_clean"],
            }
            for item in source_before["files"]
        ],
    )
    if not (source_before == source_after == manifest_payload == payload_snapshot):
        raise BackupError("verified-dependency payload hash/count verification failed")
    if manifest["handle_backed"]:
        identity_keys = {
            "relative_path", "volume_serial", "file_index", "link_count",
            "bytes", "mtime_ns", "sha256",
        }
        identity_before = manifest.get("source_identity_before")
        identity_after = manifest.get("source_identity_after")
        payload_identity = manifest.get("payload_identity")
        if type(identity_before) is not list or type(identity_after) is not list or type(payload_identity) is not list:
            raise BackupError("handle-backed verified-dependency identities are required")
        if len(identity_before) != len(includes) or identity_before != identity_after:
            raise BackupError("handle-backed source identities are inconsistent")
        for collection_name, collection in (
            ("source", identity_before),
            ("payload", payload_identity),
        ):
            if len(collection) != len(includes):
                raise BackupError(f"handle-backed {collection_name} identity count is invalid")
            for record, relative in zip(collection, includes):
                if type(record) is not dict or set(record) != identity_keys:
                    raise BackupError(f"handle-backed {collection_name} identity record is malformed")
                if record.get("relative_path") != relative:
                    raise BackupError(f"handle-backed {collection_name} identity path mismatch")
                for key in ("link_count", "bytes", "mtime_ns"):
                    if type(record.get(key)) is not int or record[key] < 0:
                        raise BackupError(f"handle-backed {collection_name} identity field is invalid")
                if record["link_count"] != 1:
                    raise BackupError(f"handle-backed {collection_name} identity is hardlinked")
                if type(record.get("sha256")) is not str or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
                    raise BackupError(f"handle-backed {collection_name} identity hash is invalid")
                if collection_name == "source":
                    content = source_before["files"][includes.index(relative)]
                    if record["bytes"] != content["bytes"] or record["sha256"] != content["sha256"]:
                        raise BackupError("handle-backed source identity does not match content snapshot")
        actual_payload_identity = _dependency_payload_identities(payload, normalized)
        if actual_payload_identity != payload_identity:
            raise BackupError("handle-backed payload identity verification failed")
    return {
        "action": "verify",
        "backup_dir": str(directory),
        "status": "PARTIAL",
        "backup_kind": _DEPENDENCY_BACKUP_KIND,
        "formal_result_eligible": False,
        "paper_table_eligible": False,
        "payload_snapshot": payload_snapshot,
        "passed": True,
    }


def _validate_dependency_includes_for_manifest(includes: list[Any]) -> tuple[str, ...]:
    if any(type(value) is not str for value in includes):
        raise BackupError("verified-dependency include entries must be strings")
    normalized = tuple(sorted(_normalize_dependency_include(value) for value in includes))
    if len(normalized) != len(set(normalized)):
        raise BackupError("verified-dependency include paths are duplicated")
    return normalized


def verify_backup(*, backup_dir: Path | str, backup_root: Path | str = DEFAULT_BACKUP_ROOT) -> dict[str, Any]:
    """Fully verify a finalized backup and its payload hashes."""

    root = _validate_backup_root(backup_root)
    directory = _check_backup_path(Path(backup_dir), root, must_exist=True)
    if not directory.name.endswith(".finalized"):
        raise BackupError("verify requires a .finalized backup directory")
    manifest = _read_manifest(directory)
    if manifest.get("backup_kind") == _DEPENDENCY_BACKUP_KIND:
        return _verify_dependency_backup(directory, manifest)
    status = manifest.get("status")
    if status not in {"STOPPED", "PARTIAL", "PASS"}:
        raise BackupError("backup manifest has an invalid status")
    if status == "PASS":
        if manifest.get("formal_result_eligible") is not True or manifest.get("eligibility_report_passed") is not True:
            raise BackupError("PASS backup does not satisfy its formal eligibility gate")
    elif manifest.get("formal_result_eligible") is not False:
        raise BackupError("STOPPED/PARTIAL backup must be marked non-paper-eligible")
    payload = directory / "payload"
    payload_snapshot = _snapshot_tree(payload)
    source_before = manifest.get("source_before")
    source_after = manifest.get("source_after")
    if not isinstance(source_before, dict) or not isinstance(source_after, dict):
        raise BackupError("backup manifest is missing source snapshots")
    if not _snapshot_equal(source_before, source_after):
        raise BackupError("backup manifest records a source mutation")
    expected = manifest.get("payload_after")
    if not isinstance(expected, dict) or not _snapshot_equal(payload_snapshot, expected):
        raise BackupError("backup payload hash/count verification failed")
    return {
        "action": "verify",
        "backup_dir": str(directory),
        "status": status,
        "formal_result_eligible": manifest["formal_result_eligible"],
        "payload_snapshot": payload_snapshot,
        "passed": True,
    }


def restore_backup(
    *,
    backup_dir: Path | str,
    restore_target: Path | str,
    backup_root: Path | str = DEFAULT_BACKUP_ROOT,
) -> dict[str, Any]:
    """Restore a verified backup into a new, never-overwritten outputs child."""

    result = verify_backup(backup_dir=backup_dir, backup_root=backup_root)
    source_backup = _check_backup_path(Path(backup_dir), _validate_backup_root(backup_root), must_exist=True)
    if result.get("backup_kind") == _DEPENDENCY_BACKUP_KIND:
        raise BackupError("restore is not supported for verified-dependency backups")
    target = _absolute(restore_target)
    outputs_root = _absolute(OUTPUTS_ROOT)
    _path_parts_from_root(target, outputs_root)
    if target == outputs_root:
        raise BackupError("restore target must be a new child of outputs, not outputs itself")
    _check_existing_components(target, outputs_root)
    if target.exists() or target.is_symlink():
        raise BackupError(f"restore refuses to overwrite existing target: {target}")
    parent = target.parent
    if not parent.exists() or not parent.is_dir() or _is_reparse(parent):
        raise BackupError(f"restore parent is missing or unsafe: {parent}")
    partial = target.with_name(target.name + ".restore.partial")
    if partial.exists() or partial.is_symlink():
        raise BackupError(f"restore staging path already exists: {partial}")
    try:
        shutil.copytree(source_backup / "payload", partial, symlinks=False)
        _check_existing_components(partial, outputs_root)
        restored_snapshot = _snapshot_tree(partial)
        expected = _read_manifest(source_backup).get("payload_after")
        if not isinstance(expected, dict) or not _snapshot_equal(restored_snapshot, expected):
            raise BackupError("restored payload verification failed")
        _fsync_directory(partial)
        _rename_noreplace(partial, target)
        _fsync_directory(parent)
    except Exception as exc:
        # Preserve a failed restore for forensic review.  Never clean or
        # delete output paths automatically.
        if partial.exists() and not target.exists():
            raise BackupError(
                f"{exc}; incomplete restore retained for main-window review: {partial}"
            ) from exc
        raise
    return {
        "action": "restore",
        "backup_dir": str(source_backup),
        "restore_target": str(target),
        "status": result["status"],
        "passed": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed experiment output backup utility")
    sub = parser.add_subparsers(dest="action", required=True)

    def source_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--source", type=Path)
        command.add_argument("--run-id")
        command.add_argument("--expect-status", choices=("STOPPED", "PARTIAL", "PASS"))
        # Resolve the default in ``main`` so embedding applications/tests that
        # deliberately replace DEFAULT_BACKUP_ROOT still use the safe root.
        command.add_argument("--backup-root", type=Path)
        command.add_argument("--max-bytes", type=int)

    dry = sub.add_parser("dry-run", help="validate a prospective backup without writing")
    source_args(dry)
    create = sub.add_parser("create", help="copy and finalize an immutable backup")
    source_args(create)
    create.add_argument("--backup-dir", type=Path)

    verify = sub.add_parser("verify", help="verify all backup hashes and counts")
    verify.add_argument("--backup-dir", type=Path, required=True)
    verify.add_argument("--backup-root", type=Path)

    restore = sub.add_parser("restore", help="restore into a new outputs child")
    restore.add_argument("--backup-dir", type=Path, required=True)
    restore.add_argument("--restore-target", type=Path, required=True)
    restore.add_argument("--backup-root", type=Path)

    def dependency_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--include", action="append", required=True)
        command.add_argument("--label", required=True)
        command.add_argument("--expect-git-head", required=True)
        command.add_argument("--backup-root", type=Path)
        command.add_argument("--max-bytes", type=int)
        command.add_argument(
            "--allow-ignored-includes",
            action="store_true",
            help="explicitly allow selected Git-ignored files (still non-paper PARTIAL)",
        )

    dependency_dry = sub.add_parser(
        "dependency-dry-run", help="validate a verified-dependency backup without writing"
    )
    dependency_args(dependency_dry)
    dependency_create_parser = sub.add_parser(
        "dependency-create", help="copy and finalize a verified-dependency backup"
    )
    dependency_args(dependency_create_parser)
    dependency_create_parser.add_argument("--backup-dir", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.action == "dry-run":
            result = dry_run(
                source=args.source, run_id=args.run_id, expect_status=args.expect_status,
                backup_root=args.backup_root or DEFAULT_BACKUP_ROOT, max_bytes=args.max_bytes,
            )
        elif args.action == "create":
            result = create_backup(
                source=args.source, run_id=args.run_id, expect_status=args.expect_status,
                backup_root=args.backup_root or DEFAULT_BACKUP_ROOT, backup_dir=args.backup_dir, max_bytes=args.max_bytes,
            )
        elif args.action == "verify":
            result = verify_backup(backup_dir=args.backup_dir, backup_root=args.backup_root or DEFAULT_BACKUP_ROOT)
        elif args.action == "restore":
            result = restore_backup(
                backup_dir=args.backup_dir, restore_target=args.restore_target,
                backup_root=args.backup_root or DEFAULT_BACKUP_ROOT,
            )
        elif args.action == "dependency-dry-run":
            result = dependency_dry_run(
                source_root=args.source_root,
                includes=args.include,
                label=args.label,
                expect_git_head=args.expect_git_head,
                backup_root=args.backup_root or DEFAULT_BACKUP_ROOT,
                max_bytes=args.max_bytes,
                allow_ignored_includes=args.allow_ignored_includes,
            )
        elif args.action == "dependency-create":
            result = dependency_create(
                source_root=args.source_root,
                includes=args.include,
                label=args.label,
                expect_git_head=args.expect_git_head,
                backup_root=args.backup_root or DEFAULT_BACKUP_ROOT,
                backup_dir=args.backup_dir,
                max_bytes=args.max_bytes,
                allow_ignored_includes=args.allow_ignored_includes,
            )
        else:  # pragma: no cover - argparse enforces this
            raise BackupError(f"unsupported action: {args.action}")
    except BackupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
