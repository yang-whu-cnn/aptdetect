import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from formal_experiments.evaluation import backup_outputs as backups


class BackupOutputTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.backup_root = self.root / "backups"
        self.source = self.outputs / "run-1"
        self.source.mkdir()
        self._constants = mock.patch.multiple(
            backups, OUTPUTS_ROOT=self.outputs, DEFAULT_BACKUP_ROOT=self.backup_root
        )
        self._constants.start()

    def tearDown(self):
        self._constants.stop()
        self.tempdir.cleanup()

    def write_status(self, status):
        (self.source / "run_status.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )

    def write_pass_gate(self, *, eligible=True, passed=True):
        (self.source / "manifest.json").write_text(
            json.dumps({"formal_result_eligible": eligible}), encoding="utf-8"
        )
        (self.source / "eligibility_report.json").write_text(
            json.dumps({"passed": passed}), encoding="utf-8"
        )

    def make_partial(self):
        self.write_status("PARTIAL")
        (self.source / "result.bin").write_bytes(b"small result")

    def create_partial_backup(self):
        self.make_partial()
        return backups.create_backup(source=self.source, backup_root=self.backup_root)

    def rewrite_backup_manifest(self, backup_dir, mutate):
        manifest_path = Path(backup_dir) / "backup_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutate(manifest)
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        (Path(backup_dir) / "backup_manifest.sha256").write_bytes(
            (hashlib.sha256(manifest_bytes).hexdigest() + "\n").encode("ascii")
        )
        return manifest

    def test_running_missing_and_unknown_status_are_rejected(self):
        for status in ("RUNNING", "UNKNOWN"):
            with self.subTest(status=status):
                self.write_status(status)
                with self.assertRaises(backups.BackupError):
                    backups.dry_run(source=self.source, backup_root=self.backup_root)
        (self.source / "run_status.json").unlink()
        with self.assertRaises(backups.BackupError):
            backups.dry_run(source=self.source, backup_root=self.backup_root)

    def test_exact_train_only_orchestrator_safe_stop_is_accepted(self):
        (self.source / "orchestrator_report.json").write_text(
            json.dumps({
                "status": "STOPPED",
                "execution_status": "STOPPED",
                "result_state": "PARTIAL",
                "expected_safe_stop": True,
                "formal_result_eligible": False,
            }),
            encoding="utf-8",
        )
        (self.source / "exit_code.txt").write_text("3\n", encoding="utf-8-sig")
        result = backups.dry_run(
            source=self.source,
            backup_root=self.backup_root,
            expect_status="STOPPED",
        )
        self.assertEqual(result["status"], "STOPPED")
        self.assertFalse(result["formal_result_eligible"])

    def test_train_only_orchestrator_safe_stop_is_fail_closed(self):
        base = {
            "status": "STOPPED",
            "execution_status": "STOPPED",
            "result_state": "PARTIAL",
            "expected_safe_stop": True,
            "formal_result_eligible": False,
        }
        for key, bad_value in (
            ("status", "RUNNING"),
            ("expected_safe_stop", False),
            ("formal_result_eligible", True),
        ):
            with self.subTest(key=key):
                payload = dict(base)
                payload[key] = bad_value
                (self.source / "orchestrator_report.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                (self.source / "exit_code.txt").write_text("3", encoding="utf-8")
                with self.assertRaises(backups.BackupError):
                    backups.dry_run(source=self.source, backup_root=self.backup_root)
        (self.source / "orchestrator_report.json").write_text(
            json.dumps(base), encoding="utf-8"
        )
        (self.source / "exit_code.txt").write_text("0", encoding="utf-8")
        with self.assertRaises(backups.BackupError):
            backups.dry_run(source=self.source, backup_root=self.backup_root)

    def test_pass_requires_both_formal_gates(self):
        self.write_status("PASS")
        for eligible, passed in ((False, True), (True, False)):
            with self.subTest(eligible=eligible, passed=passed):
                self.write_pass_gate(eligible=eligible, passed=passed)
                with self.assertRaises(backups.BackupError):
                    backups.dry_run(source=self.source, backup_root=self.backup_root)
        self.write_pass_gate()
        result = backups.dry_run(source=self.source, backup_root=self.backup_root)
        self.assertTrue(result["formal_result_eligible"])
        self.assertTrue(result["eligibility_report_passed"])

    def test_partial_backup_is_explicitly_non_paper_eligible(self):
        result = self.create_partial_backup()
        self.assertFalse(result["formal_result_eligible"])
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertFalse(manifest["formal_result_eligible"])
        self.assertFalse(manifest["eligibility_report_passed"])

    def test_source_root_and_path_escape_are_rejected(self):
        self.write_status("PARTIAL")
        with self.assertRaises(backups.BackupError):
            backups.dry_run(source=self.outputs, backup_root=self.backup_root)
        with self.assertRaises(backups.BackupError):
            backups.dry_run(source=self.outputs / ".." / "outputs", backup_root=self.backup_root)
        outside = self.root / "outside"
        outside.mkdir()
        with self.assertRaises(backups.BackupError):
            backups.dry_run(source=outside, backup_root=self.backup_root)

    def test_symlink_and_reparse_entries_are_rejected(self):
        self.make_partial()
        link = self.source / "link"
        try:
            link.symlink_to(self.root / "elsewhere")
        except (OSError, NotImplementedError):
            # Windows CI without symlink privilege still exercises the exact
            # same gate through the reparse predicate below.
            pass
        else:
            with self.assertRaises(backups.BackupError):
                backups.dry_run(source=self.source, backup_root=self.backup_root)
        with mock.patch.object(backups, "_is_reparse", side_effect=lambda path: Path(path).name == "result.bin"):
            with self.assertRaises(backups.BackupError):
                backups.dry_run(source=self.source, backup_root=self.backup_root)

    def test_backup_root_is_strictly_limited(self):
        self.make_partial()
        with self.assertRaises(backups.BackupError):
            backups.dry_run(source=self.source, backup_root=self.root / "other-backups")

    def test_hash_counts_and_manifest_sidecar_are_recorded(self):
        result = self.create_partial_backup()
        backup_dir = Path(result["backup_dir"])
        manifest_path = backup_dir / "backup_manifest.json"
        sidecar = backup_dir / "backup_manifest.sha256"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        self.assertEqual(manifest["source_before"], manifest["source_after"])
        self.assertEqual(manifest["source_before"], manifest["payload_after"])
        self.assertEqual(manifest["source_before"]["file_count"], 2)
        self.assertEqual(
            sidecar.read_text(encoding="ascii").strip(), hashlib.sha256(manifest_bytes).hexdigest()
        )
        self.assertTrue(backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)["passed"])

    def test_source_race_fails_closed(self):
        self.make_partial()
        original = backups._snapshot_tree
        calls = {"count": 0}

        def race_snapshot(path):
            calls["count"] += 1
            # ``create_backup`` snapshots the source in its preparation
            # phase, then calls this injected snapshotter for payload and
            # source-after respectively.
            if calls["count"] == 2:
                (self.source / "result.bin").write_bytes(b"changed after copy")
            return original(path)

        with self.assertRaises(backups.BackupError):
            backups.create_backup(
                source=self.source, backup_root=self.backup_root, _snapshotter=race_snapshot
            )
        self.assertFalse(list(self.backup_root.glob("*.finalized")))
        retained = list(self.backup_root.glob("*.partial"))
        self.assertEqual(len(retained), 1)
        self.assertTrue((retained[0] / "payload" / "result.bin").is_file())

    def test_existing_backup_is_never_overwritten(self):
        first = self.create_partial_backup()
        with self.assertRaises(backups.BackupError):
            backups.create_backup(
                source=self.source,
                backup_root=self.backup_root,
                backup_dir=first["backup_dir"],
            )
        self.assertTrue(Path(first["backup_dir"]).is_dir())

    def test_restore_requires_new_target_and_never_overwrites(self):
        result = self.create_partial_backup()
        target = self.outputs / "restored-run"
        restored = backups.restore_backup(
            backup_dir=result["backup_dir"], restore_target=target, backup_root=self.backup_root
        )
        self.assertTrue(restored["passed"])
        self.assertEqual((target / "result.bin").read_bytes(), b"small result")
        with self.assertRaises(backups.BackupError):
            backups.restore_backup(
                backup_dir=result["backup_dir"], restore_target=target, backup_root=self.backup_root
            )
        with self.assertRaises(backups.BackupError):
            backups.restore_backup(
                backup_dir=result["backup_dir"],
                restore_target=self.outputs,
                backup_root=self.backup_root,
            )

    def test_failed_restore_retains_partial_for_main_window_review(self):
        result = self.create_partial_backup()
        target = self.outputs / "failed-restore"
        original = backups._snapshot_tree
        calls = {"count": 0}

        def mismatched_snapshot(path):
            calls["count"] += 1
            snapshot = original(path)
            # The first snapshot verifies the finalized backup.  Corrupt only
            # the in-memory comparison for the copied restore staging tree.
            if calls["count"] == 2:
                snapshot = dict(snapshot)
                snapshot["total_bytes"] += 1
            return snapshot

        with mock.patch.object(backups, "_snapshot_tree", side_effect=mismatched_snapshot):
            with self.assertRaises(backups.BackupError):
                backups.restore_backup(
                    backup_dir=result["backup_dir"],
                    restore_target=target,
                    backup_root=self.backup_root,
                )
        self.assertFalse(target.exists())
        self.assertTrue(target.with_name(target.name + ".restore.partial").is_dir())

    def test_space_gate_can_be_mocked(self):
        self.make_partial()
        usage = shutil_usage(total=100 * 1024**3, free=2 * 1024**3 - 1)
        with mock.patch.object(backups.shutil, "disk_usage", return_value=usage):
            with self.assertRaises(backups.BackupError):
                backups.dry_run(source=self.source, backup_root=self.backup_root)
        usage = shutil_usage(total=100 * 1024**3, free=10 * 1024**3)
        with mock.patch.object(backups.shutil, "disk_usage", return_value=usage):
            result = backups.dry_run(source=self.source, backup_root=self.backup_root)
        self.assertTrue(result["space_warning"]["below_20_gib"])

    def test_dry_run_does_not_write(self):
        self.make_partial()
        result = backups.dry_run(source=self.source, backup_root=self.backup_root)
        self.assertFalse(result["would_write"])
        self.assertFalse(self.backup_root.exists())

    def make_dependency_source(self):
        source = self.root / "dependency-source"
        (source / "nested").mkdir(parents=True)
        (source / "artifact.json").write_text('{"ok": true}\n', encoding="utf-8")
        (source / "nested" / "coverage.json").write_text('{"coverage": 1}\n', encoding="utf-8")
        (source / "ignored.txt").write_text("not selected\n", encoding="utf-8")
        return source

    def dependency_identity(self, head="a" * 40, clean=True):
        return {"git_root": str(self.root), "head": head, "clean": clean}

    def dependency_args(self, source, *, includes=None, label="priorrl-candidate", head="a" * 40):
        return {
            "source_root": source,
            "includes": includes or ["nested/coverage.json", "artifact.json"],
            "label": label,
            "expect_git_head": head,
            "backup_root": self.backup_root,
        }

    def make_git_dependency_source(self):
        source = self.root / "git-dependency-source"
        source.mkdir()
        (source / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
        (source / "tracked.json").write_text('{"tracked": true}\n', encoding="utf-8")
        (source / "selected.ignored").write_text("ignored\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(source), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(source), "config", "user.name", "backup-tests"], check=True)
        subprocess.run(["git", "-C", str(source), "add", ".gitignore", "tracked.json"], check=True)
        subprocess.run(["git", "-C", str(source), "commit", "-qm", "init"], check=True)
        head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        return source, head

    def test_dependency_dry_run_is_read_only_and_subset_aware(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_dry_run(**self.dependency_args(source))
        self.assertFalse(result["would_write"])
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["backup_kind"], "verified_dependency")
        self.assertFalse(result["formal_result_eligible"])
        self.assertFalse(result["paper_table_eligible"])
        self.assertEqual(result["include"], ["artifact.json", "nested/coverage.json"])
        self.assertEqual(result["source_snapshot"]["file_count"], 2)
        self.assertFalse(self.backup_root.exists())

    def test_dependency_ignored_include_requires_explicit_opt_in(self):
        source, head = self.make_git_dependency_source()
        with self.assertRaises(backups.BackupError):
            backups.dependency_dry_run(
                source_root=source,
                includes=["selected.ignored"],
                label="ignored",
                expect_git_head=head,
                backup_root=self.backup_root,
            )
        result = backups.dependency_dry_run(
            source_root=source,
            includes=["selected.ignored"],
            label="ignored",
            expect_git_head=head,
            backup_root=self.backup_root,
            allow_ignored_includes=True,
        )
        self.assertTrue(result["allow_ignored_includes"])
        self.assertEqual(result["git_files"][0]["git_classification"], "ignored_untracked")
        self.assertFalse(result["git_files"][0]["tracked_worktree_clean"])

    @unittest.skipUnless(os.name == "nt", "production dependency locking is Windows-only")
    def test_dependency_production_handle_pipeline_uses_real_git_identity(self):
        source, head = self.make_git_dependency_source()
        result = backups.dependency_create(
            source_root=source,
            includes=["selected.ignored"],
            label="real-production",
            expect_git_head=head,
            backup_root=self.backup_root,
            backup_dir="real-production",
            allow_ignored_includes=True,
        )
        backup_dir = Path(result["backup_dir"])
        manifest = json.loads((backup_dir / "backup_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["handle_backed"])
        self.assertEqual(manifest["source_identity_before"], manifest["source_identity_after"])
        self.assertNotEqual(
            manifest["source_identity_before"][0].get("file_index"),
            manifest["payload_identity"][0].get("file_index"),
        )
        self.assertTrue(backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)["passed"])

    def test_windows_lockset_uses_source_root_handle_for_sibling_directories(self):
        source = self.root / "sibling-source"
        delay_only = source / "delay_only"
        fail_only = source / "fail_only"
        delay_only.mkdir(parents=True)
        fail_only.mkdir()
        (delay_only / "delay.json").write_text("delay\n", encoding="utf-8")
        (fail_only / "fail.json").write_text("fail\n", encoding="utf-8")
        includes = ("delay_only/delay.json", "fail_only/fail.json")

        handles: dict[int, Path] = {}
        next_handle = 100

        def open_handle(path, **_kwargs):
            nonlocal next_handle
            handle = next_handle
            next_handle += 1
            handles[handle] = Path(path)
            return handle

        def handle_info(handle):
            return {
                "attributes": 0,
                "link_count": 1,
                "volume_serial": 1,
                "file_index": handle,
            }

        def final_path(handle):
            return os.path.abspath(os.fspath(handles[handle]))

        def open_osfhandle(handle, _flags):
            return os.open(os.fspath(handles[handle]), os.O_RDONLY)

        fake_msvcrt = mock.Mock()
        fake_msvcrt.open_osfhandle.side_effect = open_osfhandle
        with mock.patch.object(backups.os, "name", "nt"), mock.patch.object(
            backups, "_windows_open_handle", side_effect=open_handle
        ), mock.patch.object(backups, "_windows_handle_info", side_effect=handle_info), mock.patch.object(
            backups, "_windows_handle_final_path", side_effect=final_path
        ), mock.patch.object(backups, "_windows_close_handle"), mock.patch.dict(
            sys.modules, {"msvcrt": fake_msvcrt}
        ):
            with backups._WindowsDependencyLockSet(source, includes) as lockset:
                self.assertEqual(lockset.source_root_final, os.path.abspath(os.fspath(source)))
                self.assertEqual(set(lockset.file_handles), set(includes))

    def test_windows_lockset_rejects_outside_sibling(self):
        source = self.root / "sibling-source"
        source.mkdir()
        outside = self.root / "outside.json"
        outside.write_text("outside\n", encoding="utf-8")
        handles: dict[int, Path] = {}
        next_handle = 100

        def open_handle(path, **_kwargs):
            nonlocal next_handle
            handle = next_handle
            next_handle += 1
            handles[handle] = Path(path)
            return handle

        def handle_info(_handle):
            return {"attributes": 0, "link_count": 1, "volume_serial": 1, "file_index": 1}

        def final_path(handle):
            return os.path.abspath(os.fspath(handles[handle]))

        with mock.patch.object(backups.os, "name", "nt"), mock.patch.object(
            backups, "_windows_open_handle", side_effect=open_handle
        ), mock.patch.object(backups, "_windows_handle_info", side_effect=handle_info), mock.patch.object(
            backups, "_windows_handle_final_path", side_effect=final_path
        ), mock.patch.object(backups, "_windows_close_handle"):
            with self.assertRaisesRegex(backups.BackupError, "selected dependency escaped locked source root"):
                with backups._WindowsDependencyLockSet(source, ("../outside.json",)):
                    pass

    def test_finalized_manifest_backup_id_binding_applies_to_ordinary_backup(self):
        result = self.create_partial_backup()
        backup_dir = Path(result["backup_dir"])
        self.rewrite_backup_manifest(backup_dir, lambda manifest: manifest.update(backup_id="different"))
        with self.assertRaisesRegex(backups.BackupError, "directory/manifest backup_id mismatch"):
            backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)

    def test_ordinary_double_finalized_suffix_rejected_on_create_and_verify(self):
        self.make_partial()
        with self.assertRaisesRegex(backups.BackupError, "reserved suffix"):
            backups.create_backup(
                source=self.source,
                backup_root=self.backup_root,
                backup_dir="ordinary.finalized.finalized",
            )

        result = backups.create_backup(
            source=self.source,
            backup_root=self.backup_root,
            backup_dir="ordinary",
        )
        backup_dir = Path(result["backup_dir"])
        attacked_dir = backup_dir.with_name("ordinary.finalized.finalized")
        backup_dir.rename(attacked_dir)
        self.rewrite_backup_manifest(
            attacked_dir,
            lambda manifest: manifest.update(backup_id="ordinary.finalized"),
        )
        with self.assertRaisesRegex(backups.BackupError, "exactly one .finalized suffix"):
            backups.verify_backup(backup_dir=attacked_dir, backup_root=self.backup_root)

    def test_dependency_unrelated_ignored_file_does_not_dirty_gate(self):
        source, head = self.make_git_dependency_source()
        result = backups.dependency_dry_run(
            source_root=source,
            includes=["tracked.json"],
            label="tracked",
            expect_git_head=head,
            backup_root=self.backup_root,
        )
        self.assertTrue(result["tracked_worktree_clean"])

    def test_dependency_hardlink_is_rejected(self):
        source = self.make_dependency_source()
        hardlink = source / "hardlink.json"
        try:
            os.link(source / "artifact.json", hardlink)
        except OSError as exc:
            self.skipTest(f"hardlinks unavailable: {exc}")
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            with self.assertRaises(backups.BackupError):
                backups.dependency_dry_run(
                    **self.dependency_args(
                        source,
                        includes=["hardlink.json"],
                    )
                )

    def test_dependency_ancestor_reparse_is_rejected(self):
        source = self.make_dependency_source()
        ancestor = source.parent / "ancestor-link"
        try:
            ancestor.symlink_to(source.parent, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            with self.assertRaises(backups.BackupError):
                backups.dependency_dry_run(
                    **self.dependency_args(ancestor / source.name)
                )

    def test_dependency_mocked_intermediate_ancestor_reparse_is_rejected(self):
        source = self.make_dependency_source()
        middle = source.parent
        real_is_reparse = backups._is_reparse

        def reparse_middle(path):
            if Path(path) == middle:
                return True
            return real_is_reparse(path)

        with mock.patch.object(backups, "_is_reparse", side_effect=reparse_middle):
            with self.assertRaises(backups.BackupError):
                backups._validate_dependency_source_root(source)

    def test_dependency_verify_rejects_schema_version_drift(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="schema")
        backup_dir = Path(result["backup_dir"])
        manifest_path = backup_dir / "backup_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 2
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        (backup_dir / "backup_manifest.sha256").write_text(
            hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="ascii"
        )
        with self.assertRaises(backups.BackupError):
            backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)

    def test_dependency_verify_rejects_boolean_integer_schema_version(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="schema-bool")
        backup_dir = Path(result["backup_dir"])
        manifest_path = backup_dir / "backup_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = True
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        (backup_dir / "backup_manifest.sha256").write_text(
            hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="ascii"
        )
        with self.assertRaises(backups.BackupError):
            backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)

    def test_dependency_verify_rejects_noncanonical_git_relative_path(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="git-path")
        backup_dir = Path(result["backup_dir"])
        manifest_path = backup_dir / "backup_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["git_files"][0]["git_relative_path"] = "../escape.json"
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        (backup_dir / "backup_manifest.sha256").write_text(
            hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="ascii"
        )
        with self.assertRaises(backups.BackupError):
            backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)

    def test_dependency_verify_rejects_sidecar_trailing_garbage(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="sidecar-garbage")
        sidecar = Path(result["backup_dir"]) / "backup_manifest.sha256"
        sidecar.write_text(sidecar.read_text(encoding="ascii") + "garbage\n", encoding="ascii")
        with self.assertRaises(backups.BackupError):
            backups.verify_backup(backup_dir=result["backup_dir"], backup_root=self.backup_root)

    def test_dependency_nested_directories_are_fsynced(self):
        source = self.make_dependency_source()
        real_fsync = backups._fsync_directory
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            with mock.patch.object(backups, "_fsync_directory", wraps=real_fsync) as fsync:
                backups.dependency_create(**self.dependency_args(source), backup_dir="nested-fsync")
        paths = {Path(call.args[0]).as_posix() for call in fsync.call_args_list if call.args}
        self.assertTrue(any(path.endswith("payload/nested") for path in paths))

    def test_dependency_link_count_race_is_rejected(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            with mock.patch.object(
                backups,
                "_dependency_handle_identity",
                return_value={"link_count": 2, "device": 1, "inode": 1},
            ):
                with self.assertRaises(backups.BackupError):
                    backups.dependency_dry_run(**self.dependency_args(source))

    def test_dependency_verify_requires_all_three_snapshots_equal(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="triple")
        backup_dir = Path(result["backup_dir"])
        manifest_path = backup_dir / "backup_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_after"]["total_bytes"] += 1
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        (backup_dir / "backup_manifest.sha256").write_text(
            hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="ascii"
        )
        with self.assertRaises(backups.BackupError):
            backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)

    def test_dependency_create_and_verify_records_non_paper_manifest(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="priorrl-candidate")
        backup_dir = Path(result["backup_dir"])
        manifest = json.loads((backup_dir / "backup_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["backup_kind"], "verified_dependency")
        self.assertEqual(manifest["status"], "PARTIAL")
        self.assertFalse(manifest["formal_result_eligible"])
        self.assertFalse(manifest["paper_table_eligible"])
        self.assertFalse(manifest["eligibility_report_passed"])
        self.assertTrue(manifest["source_git_clean"])
        self.assertEqual(manifest["include"], ["artifact.json", "nested/coverage.json"])
        self.assertEqual(manifest["source_before"], manifest["source_after"])
        self.assertEqual(manifest["source_before"], manifest["payload_after"])
        verified = backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)
        self.assertTrue(verified["passed"])
        self.assertEqual(verified["backup_kind"], "verified_dependency")

    def test_dependency_finalized_manifest_backup_id_binding_cannot_be_bypassed_by_sidecar(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="bound-dependency")
        backup_dir = Path(result["backup_dir"])
        self.rewrite_backup_manifest(backup_dir, lambda manifest: manifest.update(backup_id="different"))
        with self.assertRaisesRegex(backups.BackupError, "directory/manifest backup_id mismatch"):
            backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)

    def test_dependency_double_finalized_suffix_rejected_on_create_and_verify(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            with self.assertRaisesRegex(backups.BackupError, "reserved suffix"):
                backups.dependency_create(
                    **self.dependency_args(source),
                    backup_dir="dependency.finalized.finalized",
                )

        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(
                **self.dependency_args(source),
                backup_dir="dependency",
            )
        backup_dir = Path(result["backup_dir"])
        attacked_dir = backup_dir.with_name("dependency.finalized.finalized")
        backup_dir.rename(attacked_dir)
        self.rewrite_backup_manifest(
            attacked_dir,
            lambda manifest: manifest.update(backup_id="dependency.finalized"),
        )
        with self.assertRaisesRegex(backups.BackupError, "exactly one .finalized suffix"):
            backups.verify_backup(backup_dir=attacked_dir, backup_root=self.backup_root)

    @unittest.skipUnless(os.name == "nt", "Windows handle-backed identity schema is Windows-only")
    def test_dependency_verify_rejects_invalid_windows_identity_fields(self):
        source = self.make_dependency_source()
        invalid_values = ("not-an-int", True, -1)
        for collection_name in ("source_identity_before", "payload_identity"):
            for field_name in ("volume_serial", "file_index"):
                for invalid_value in invalid_values:
                    with self.subTest(
                        collection=collection_name,
                        field=field_name,
                        value=repr(invalid_value),
                    ):
                        backup_name = (
                            f"identity-{collection_name.replace('_', '-')}-"
                            f"{field_name}-{invalid_values.index(invalid_value)}"
                        )
                        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
                            result = backups.dependency_create(
                                **self.dependency_args(source), backup_dir=backup_name
                            )
                        backup_dir = Path(result["backup_dir"])

                        def mutate(manifest):
                            includes = tuple(manifest["include"])
                            source_identity = backups._dependency_payload_identities(source, includes)
                            payload_identity = backups._dependency_payload_identities(
                                backup_dir / "payload", includes
                            )
                            manifest["handle_backed"] = True
                            manifest["source_identity_before"] = source_identity
                            manifest["source_identity_after"] = [dict(item) for item in source_identity]
                            manifest["payload_identity"] = payload_identity
                            manifest[collection_name][0][field_name] = invalid_value
                            if collection_name == "source_identity_before":
                                manifest["source_identity_after"][0][field_name] = invalid_value

                        self.rewrite_backup_manifest(backup_dir, mutate)
                        with self.assertRaisesRegex(
                            backups.BackupError,
                            rf"handle-backed .* identity {field_name} field is invalid",
                        ):
                            backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)

    def test_dependency_include_validation_is_fail_closed(self):
        source = self.make_dependency_source()
        cases = (
            [str(source / "artifact.json")],
            ["../artifact.json"],
            ["nested/../artifact.json"],
            ["nested/coverage.json", "nested\\coverage.json"],
            ["nested"],
            ["missing.json"],
        )
        for includes in cases:
            with self.subTest(includes=includes):
                with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
                    with self.assertRaises(backups.BackupError):
                        backups.dependency_dry_run(**self.dependency_args(source, includes=includes))

    def test_dependency_symlink_is_rejected(self):
        source = self.make_dependency_source()
        link = source / "linked.json"
        try:
            link.symlink_to(source / "artifact.json")
        except (OSError, NotImplementedError):
            link = source / "artifact.json"
            reparse = lambda path: Path(path).name == "artifact.json"
        else:
            reparse = lambda path: Path(path) == link
        with mock.patch.object(backups, "_is_reparse", side_effect=reparse):
            with self.assertRaises(backups.BackupError):
                backups.dependency_dry_run(
                    **self.dependency_args(source, includes=[link.name])
                )

    def test_dependency_git_head_and_dirty_gates(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity(head="b" * 40)):
            with self.assertRaises(backups.BackupError):
                backups.dependency_dry_run(**self.dependency_args(source))
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity(clean=False)):
            with self.assertRaises(backups.BackupError):
                backups.dependency_dry_run(**self.dependency_args(source))

    def test_dependency_target_conflict_is_never_overwritten(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            first = backups.dependency_create(**self.dependency_args(source), backup_dir="fixed")
            with self.assertRaises(backups.BackupError):
                backups.dependency_create(**self.dependency_args(source), backup_dir="fixed")
        self.assertTrue(Path(first["backup_dir"]).is_dir())
        self.assertFalse(list(self.backup_root.glob("fixed*.partial")))

    def test_dependency_source_race_retains_partial_evidence(self):
        source = self.make_dependency_source()
        original = backups._dependency_snapshot
        calls = {"count": 0}

        def racing_snapshot(root, includes, strict_payload):
            calls["count"] += 1
            snapshot = original(root, includes, strict_payload)
            if calls["count"] == 2:
                (source / "artifact.json").write_text("changed\n", encoding="utf-8")
            return snapshot

        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            with self.assertRaises(backups.BackupError):
                backups.dependency_create(
                    **self.dependency_args(source), _snapshotter=racing_snapshot
                )
        self.assertFalse(list(self.backup_root.glob("*.finalized")))
        self.assertEqual(len(list(self.backup_root.glob("*.partial"))), 1)

    def test_dependency_payload_mismatch_retains_partial_evidence(self):
        source = self.make_dependency_source()
        original = backups._dependency_snapshot

        def mismatching_snapshot(root, includes, strict_payload):
            if strict_payload:
                (root / "artifact.json").write_text("payload changed\n", encoding="utf-8")
            return original(root, includes, strict_payload)

        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            with self.assertRaises(backups.BackupError):
                backups.dependency_create(
                    **self.dependency_args(source), _snapshotter=mismatching_snapshot
                )
        self.assertFalse(list(self.backup_root.glob("*.finalized")))
        self.assertEqual(len(list(self.backup_root.glob("*.partial"))), 1)

    def test_dependency_git_head_drift_during_copy_retains_partial(self):
        source = self.make_dependency_source()
        identities = [self.dependency_identity(), self.dependency_identity(head="b" * 40)]
        with mock.patch.object(backups, "_read_git_identity", side_effect=identities):
            with self.assertRaises(backups.BackupError):
                backups.dependency_create(**self.dependency_args(source))
        self.assertFalse(list(self.backup_root.glob("*.finalized")))
        self.assertEqual(len(list(self.backup_root.glob("*.partial"))), 1)

    def test_dependency_existing_partial_is_never_reused(self):
        source = self.make_dependency_source()
        self.backup_root.mkdir()
        (self.backup_root / "reserved.partial").mkdir()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            with self.assertRaises(backups.BackupError):
                backups.dependency_create(**self.dependency_args(source), backup_dir="reserved")
        self.assertTrue((self.backup_root / "reserved.partial").is_dir())

    def test_dependency_manifest_sidecar_and_payload_tampering_fail(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="tamper-sidecar")
        backup_dir = Path(result["backup_dir"])
        (backup_dir / "backup_manifest.sha256").write_text("0" * 64 + "\n", encoding="ascii")
        with self.assertRaises(backups.BackupError):
            backups.verify_backup(backup_dir=backup_dir, backup_root=self.backup_root)

        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="tamper-payload")
        payload_file = Path(result["backup_dir"]) / "payload" / "artifact.json"
        payload_file.write_text("tampered\n", encoding="utf-8")
        with self.assertRaises(backups.BackupError):
            backups.verify_backup(backup_dir=result["backup_dir"], backup_root=self.backup_root)

        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="tamper-manifest")
        manifest_path = Path(result["backup_dir"]) / "backup_manifest.json"
        manifest_path.write_text(
            manifest_path.read_text(encoding="utf-8").replace('"status": "PARTIAL"', '"status": "PASS"'),
            encoding="utf-8",
        )
        with self.assertRaises(backups.BackupError):
            backups.verify_backup(backup_dir=result["backup_dir"], backup_root=self.backup_root)

    def test_dependency_restore_is_explicitly_rejected(self):
        source = self.make_dependency_source()
        with mock.patch.object(backups, "_read_git_identity", return_value=self.dependency_identity()):
            result = backups.dependency_create(**self.dependency_args(source), backup_dir="no-restore")
        with self.assertRaises(backups.BackupError):
            backups.restore_backup(
                backup_dir=result["backup_dir"],
                restore_target=self.outputs / "restored-dependency",
                backup_root=self.backup_root,
            )


def shutil_usage(total, free):
    # ``disk_usage`` returns a named tuple; a small equivalent is enough for
    # the standard-library code and keeps this test independent of a drive.
    return type("Usage", (), {"total": total, "used": total - free, "free": free})()


if __name__ == "__main__":
    unittest.main()
