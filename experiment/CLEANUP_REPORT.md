# Repository cleanup report — 2026-09-22

The repository was synchronized with `origin/codex/v3-formal-experiments`
before cleanup. All removed content was copied to the backup below and verified
file-by-file with SHA256 before source deletion.

- Backup root: `D:\past\aptdetect_pre_cleanup_20260922T180407+0800`
- Snapshot manifest: `snapshot_manifest.json`
- Verified items: 24
- Verified files: 64,268
- Verified size: 2.123 GiB
- Recovery data: `payload/`
- Per-item hashes: `*.sha256.json`
- Git evidence: `git_status.txt`, `git_show_ref.txt`, `git_branch_vv.txt`,
  `git_worktree_list.txt`

Removed after verification:

- Chapter 1 legacy tree and IDE metadata.
- Unused `.venv`, `.venv1`, and `.venv2`; the active `.venv_cc4` was retained.
- Legacy `runs/`, `checkpoints/`, and `experiments/` trees.
- Old smoke, diagnostic, failed-launch, local-online, merge-staging,
  `lwm_rl_v2`, `formal_replay`, and `formal_replay_v2` output trees.

Explicitly retained:

- `.git`, all production code/config/tests/docs, and `experiment/`.
- `.venv_cc4` and all paths referenced by active formal writers.
- `outputs/formal_v3`, DCA repeat 1–5, dependency backups, final replay,
  final world model, final reward model, prior cache, and PriorRL prototypes.
- TERLA backup sources.
- `outputs/rebuild_20260920` and `outputs/rebuild_20260921`: one frozen model
  in the former denied read access, so both were excluded rather than risking
  an incomplete backup.
- Launcher logs while formal processes are active.

The backup is intentionally outside Git. Restoration is a copy operation from
the backup `payload/` to the repository-relative path recorded in
`snapshot_manifest.json`.
