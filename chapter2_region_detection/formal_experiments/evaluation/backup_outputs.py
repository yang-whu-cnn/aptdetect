"""Fail-closed backup and restore utility for experiment output directories.

This module intentionally uses only the Python standard library.  It does not
import the experiment runtime, CybORG, an LLM provider, or any project model.
The source and destination boundaries are deliberately narrow because a
backup is an audit artifact, not a general-purpose file copier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


OUTPUTS_ROOT = Path(__file__).resolve().parents[2] / "outputs"
DEFAULT_BACKUP_ROOT = Path(r"C:\aptdetect_experiment_backups")
_BACKUP_SCHEMA = "aptdetect_experiment_backup_v1"
_CHUNK_SIZE = 1024 * 1024
_TWO_GIB = 2 * 1024 * 1024 * 1024
_TWENTY_GIB = 20 * 1024 * 1024 * 1024
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class BackupError(RuntimeError):
    """A fail-closed backup operation error."""


def _absolute(path: Path | str) -> Path:
    """Return an absolute lexical path without following links."""

    return Path(os.path.abspath(os.fspath(path)))


def _is_reparse(path: Path) -> bool:
    """Return whether *path* is a symlink or Windows reparse point."""

    try:
        info = os.lstat(path)
    except OSError as exc:
        raise BackupError(f"cannot inspect path {path}: {exc}") from exc
    return bool(stat.S_ISLNK(info.st_mode) or (getattr(info, "st_file_attributes", 0) & _REPARSE_POINT))


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
        if current.exists() or current.is_symlink():
            if _is_reparse(current):
                raise BackupError(f"symlink/reparse point is not allowed: {current}")


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


def _fsync_directory(path: Path) -> None:
    # Windows does not permit opening a directory for fsync in every Python
    # build.  The files themselves are always fsynced; POSIX gets a directory
    # fsync as an additional durability barrier.
    try:
        descriptor = os.open(os.fspath(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


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


def _write_manifest(staging: Path, manifest: dict[str, Any]) -> None:
    manifest_path = staging / "backup_manifest.json"
    manifest_bytes = _json_bytes(manifest)
    _write_bytes_fsync(manifest_path, manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    _write_bytes_fsync(staging / "backup_manifest.sha256", (digest + "\n").encode("ascii"))
    _fsync_directory(staging)


def _read_manifest(backup_dir: Path) -> dict[str, Any]:
    manifest_path = backup_dir / "backup_manifest.json"
    sidecar_path = backup_dir / "backup_manifest.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise BackupError("backup_manifest.json and backup_manifest.sha256 are required")
    manifest_bytes = _backup_manifest_bytes(manifest_path)
    try:
        expected = sidecar_path.read_text(encoding="ascii").strip().split()[0]
    except (OSError, UnicodeError, IndexError) as exc:
        raise BackupError(f"invalid backup manifest sidecar: {sidecar_path}") from exc
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
    try:
        # On the supported Windows deployment os.rename maps to a
        # non-replacing MoveFile operation.  The pre-check plus a unique name
        # also protects normal POSIX test runs; a collision still fails after
        # the operation is checked below.
        os.rename(staging, finalized)
    except FileExistsError as exc:
        raise BackupError(f"refusing to overwrite existing backup: {finalized}") from exc
    except OSError as exc:
        raise BackupError(f"cannot finalize backup {staging} -> {finalized}: {exc}") from exc
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


def verify_backup(*, backup_dir: Path | str, backup_root: Path | str = DEFAULT_BACKUP_ROOT) -> dict[str, Any]:
    """Fully verify a finalized backup and its payload hashes."""

    root = _validate_backup_root(backup_root)
    directory = _check_backup_path(Path(backup_dir), root, must_exist=True)
    if not directory.name.endswith(".finalized"):
        raise BackupError("verify requires a .finalized backup directory")
    manifest = _read_manifest(directory)
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
        else:  # pragma: no cover - argparse enforces this
            raise BackupError(f"unsupported action: {args.action}")
    except BackupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
