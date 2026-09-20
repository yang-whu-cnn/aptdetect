import hashlib
import json
import os
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


def shutil_usage(total, free):
    # ``disk_usage`` returns a named tuple; a small equivalent is enough for
    # the standard-library code and keeps this test independent of a drive.
    return type("Usage", (), {"total": total, "used": total - free, "free": free})()


if __name__ == "__main__":
    unittest.main()
