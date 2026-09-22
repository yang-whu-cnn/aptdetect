# CC4 v3 repeat-1 aggregation readiness

This is a read-only readiness snapshot for the first complete `repeat_1` pass. It is
not a paper table, and no figure/table publication is authorized by this file.

Snapshot: 2026-09-22 (live process check and discovery audit; volatile PIDs are only
evidence for this snapshot).

## Machine-readable checklist

```json
{
  "schema": "cc4_v3_repeat1_aggregation_readiness_v1",
  "scope": "one complete 100-episode x 500-tick repeat per physical row",
  "protocol": {
    "test_seeds": "4000..4099",
    "policy_seeds": "51001..51005",
    "repeat_count_required_for_paper_row": 5,
    "current_repeat_index": 1,
    "current_repeat_label": "PROVISIONAL",
    "sample_sd_ddof": 1
  },
  "physical_repeat1_count": 12,
  "logical_rows": {
    "table1/lwm_rl": {"physical_source": "table2/lwm_rl", "alias_only": true},
    "table2/lwm_rl": {"physical_source": "table2/lwm_rl", "alias_only": false},
    "table3/full_reward": {"physical_source": "table2/lwm_rl", "alias_only": true}
  },
  "discovery_evidence": {
    "input_root": "chapter2_region_detection/outputs/formal_v3",
    "accepted_repeat_count": 7,
    "accepted_identities": [
      "table1/dca_cc4#1", "table1/dca_cc4#2", "table1/dca_cc4#3",
      "table1/dca_cc4#4", "table1/dca_cc4#5",
      "table1/priorrl_ppo_cc4#1", "table2/rl_only#1"
    ],
    "generic_validator_consumed": [
      "table1/dca_cc4#1..5", "table1/priorrl_ppo_cc4#1"
    ],
    "true_ppo_admission_bridge_consumed": ["table2/rl_only#1"],
    "provisional_bridge_status": "PROVISIONAL",
    "final_table_generation": "BLOCKED until every physical row has repeats 1..5",
    "paper_values_emitted": false
  },
  "physical_runs": [
    {
      "table": "table1", "row": "dca_cc4", "status": "READY",
      "repeat1_status": "PASS", "paper_status": "READY_5_REPEATS",
      "path": "chapter2_region_detection/outputs/formal_v3/table1/dca_cc4/repeat_1",
      "required_artifacts": ["manifest.json", "eligibility_report.json", "episodes.jsonl", "decisions.jsonl", "metrics.json"]
    },
    {
      "table": "table1", "row": "rsmbrl_cc4", "status": "WAIT",
      "repeat1_status": "SOURCE_PASS_UNDER_OLDER_VALIDATOR",
      "paper_status": "PROVISIONAL_NOT_CURRENTLY_ADMITTED",
      "path": "chapter2_region_detection/outputs/formal_v3/dependencies/backups/rsmbrl_repeat1_pass_20260921T230914Z_v2",
      "source_path": "D:/w/rsmbrl_v32/chapter2_region_detection/outputs/formal_v3/table1/rsmbrl_cc4/repeat_1",
      "aggregation_gate": "BLOCKED",
      "reason": "current validate_formal_run rejects missing method_artifacts.normalizer_sidecar_sha256; do not hand-edit the manifest or eligibility"
    },
    {
      "table": "table1", "row": "carl_cc4", "status": "WAIT",
      "repeat1_status": "TRAIN_VALIDATION_PASS_TEST_NOT_COMPLETE",
      "paper_status": "PROVISIONAL_PENDING_TEST",
      "path": "chapter2_region_detection/outputs/formal_v3/table1/carl_cc4/repeat_1",
      "required_artifacts": ["test manifest", "episodes.jsonl", "decisions.jsonl", "metrics.json", "eligibility_report.json", "released writer lock"]
    },
    {
      "table": "table1", "row": "priorrl_ppo_cc4", "status": "READY",
      "repeat1_status": "PASS", "paper_status": "PROVISIONAL_ONE_REPEAT",
      "path": "chapter2_region_detection/outputs/formal_v3/table1/priorrl_ppo_cc4/repeat_1",
      "required_artifacts": ["manifest.json", "eligibility_report.json", "episodes.jsonl", "decisions.jsonl", "metrics.json", "checkpoint.pt"]
    },
    {
      "table": "table1", "row": "terla_a4", "status": "WAIT",
      "repeat1_status": "VALIDATION_PASS_TEST_NOT_AUTHORIZED",
      "paper_status": "PROVISIONAL_PENDING_TEST",
      "path": "chapter2_region_detection/outputs/formal_v3/table1/terla_a4/repeat_1_eval_v1",
      "required_artifacts": ["separate test output", "manifest.json", "eligibility_report.json", "episodes.jsonl", "decisions.jsonl", "metrics.json"]
    },
    {
      "table": "table1", "row": "uamcts_cc4", "status": "BLOCKED",
      "repeat1_status": "BLOCKED_SUPPORT_DOMAIN",
      "paper_status": "BLOCKED",
      "path": "chapter2_region_detection/outputs/formal_v3/table1/uamcts_cc4/repeat_1",
      "reason": "frozen support-domain/radius miss; fallback and radius widening are forbidden",
      "required_artifacts": ["new scientifically compliant fresh lineage before any formal test"]
    },
    {
      "table": "table2", "row": "rl_only", "status": "READY",
      "repeat1_status": "PASS", "paper_status": "PROVISIONAL_ONE_REPEAT",
      "path": "chapter2_region_detection/outputs/formal_v3/table2/rl_only_trueppo_v2/repeat_1",
      "admission_bridge": "chapter2_region_detection/outputs/formal_v3/admission_derived/table2/rl_only/repeat_1",
      "required_artifacts": ["source true-PPO manifest/eligibility/aggregate", "RELEASED lock", "admission bridge"]
    },
    {
      "table": "table2", "row": "llm_rl", "status": "RUNNING",
      "repeat1_status": "TEST_RUNNING",
      "paper_status": "PROVISIONAL_PENDING_ADMISSION",
      "path": "chapter2_region_detection/outputs/formal_v3/table2/llm_rl_trueppo_20260922_repeat1/repeat_1",
      "runtime_pid_snapshot": 47968,
      "required_artifacts": ["test completion", "aggregate PASS", "admission bridge", "provider_calls=0", "test_time_updates=0"]
    },
    {
      "table": "table2", "row": "wm_rl", "status": "WAIT",
      "repeat1_status": "READY_WAIT_RESOURCE",
      "paper_status": "PROVISIONAL_NOT_STARTED",
      "planned_path": "chapter2_region_detection/outputs/formal_v3/table2/wm_rl_trueppo_20260922_repeat1/repeat_1",
      "preflight_worktree": "D:/w/t2wmready",
      "reason": "resource gate; no target writer may be started until RAM/GPU gate is green"
    },
    {
      "table": "table2", "row": "lwm_rl", "status": "RUNNING",
      "repeat1_status": "TEST_RUNNING",
      "paper_status": "PROVISIONAL_PENDING_ADMISSION",
      "path": "chapter2_region_detection/outputs/formal_v3/table2/lwm_rl_trueppo_v3/repeat_1",
      "runtime_pid_snapshot": 34972,
      "canonical_run_id": "lwm_full",
      "required_artifacts": ["test completion", "aggregate PASS", "admission bridge", "provider_calls=0", "test_time_updates=0"]
    },
    {
      "table": "table3", "row": "delay_only", "status": "WAIT",
      "repeat1_status": "READY_WAIT_RESOURCE",
      "paper_status": "PROVISIONAL_NOT_STARTED",
      "planned_path": "chapter2_region_detection/outputs/formal_v3/table3/delay_only_trueppo_20260922_repeat1/repeat_1",
      "preflight_worktree": "D:/w/t3delay",
      "reason": "resource gate; independent Delay-Only predictor/policy required"
    },
    {
      "table": "table3", "row": "fail_only", "status": "RUNNING",
      "repeat1_status": "VALIDATION_RUNNING",
      "paper_status": "PROVISIONAL_PENDING_TEST",
      "path": "chapter2_region_detection/outputs/formal_v3/table3/fail_only_trueppo_20260922_repeat1",
      "runtime_pid_snapshot": 31940,
      "required_artifacts": ["validation 8/8", "approved prepare-test", "test completion", "aggregate PASS", "admission bridge", "Huber contract"]
    }
  ],
  "required_artifacts_for_each_admitted_repeat": [
    "100 episodes exactly, test seeds 4000..4099, 500 ticks each",
    "raw episodes and decisions, recomputable metrics, manifest and SHA bindings",
    "eligibility_report.passed=true or audited true-PPO source plus admission bridge",
    "frozen artifact before/after unchanged, provider/cache/test-time updates all zero",
    "training curve declaration and SHA binding",
    "repeat_index/policy_seed identity; repeat 1 remains PROVISIONAL until 1..5"
  ]
}
```

## Why the existing chain is sufficient

- Generic runs are discovered and revalidated by `generate_paper_figures.py` in
  `discover_formal_runs()`; the current root accepted DCA and PriorRL.
- True-PPO source manifests are deliberately read-only. `true_ppo_admission.py`
  audits the source and writes only an `admission_derived` pointer. The current
  RL-Only bridge was consumed and remained `PROVISIONAL`; no source manifest was
  overwritten.
- `group_formal_rows()` requires repeat indices exactly `1..5`, and
  `aggregate_five_repeats()` requires the frozen policy seeds and sample SD
  (`ddof=1`). Therefore the existing final aggregator cannot accidentally turn a
  repeat-1 snapshot into a five-repeat paper row.
- The LWM source is one physical run. After admission it must be reused for
  `table1/lwm_rl`, `table2/lwm_rl`, and `table3/full_reward` through the existing
  read-only alias path. An independent `table3/full_reward` run is rejected.

## Safe audit commands

These commands inspect discovery/identity only. Do not point publication commands
at a real output directory while any row is incomplete. The generic validator in
the current implementation owns `eligibility_report.json`; use the already
validated outputs/bridges above for read-only monitoring and never run it against
an active true-PPO source.

```powershell
$py = ".\chapter2_region_detection\.venv_cc4\Scripts\python.exe"
& $py -u -c "import sys; sys.path.insert(0,'chapter2_region_detection'); from formal_experiments.evaluation.generate_paper_figures import discover_formal_runs; a,r=discover_formal_runs('chapter2_region_detection/outputs/formal_v3'); print('accepted=',len(a)); print('rejected=',len(r)); print('\n'.join(f'{x[\"table_id\"]}/{x[\"row_id\"]} repeat={x[\"manifest\"].get(\"repeat_index\")} admission={x.get(\"admission_status\")}' for x in a))"
& $py -u -c "import sys; sys.path.insert(0,'chapter2_region_detection'); from formal_experiments.evaluation.true_ppo_admission import validate_admission_directory; print(validate_admission_directory('chapter2_region_detection/outputs/formal_v3/admission_derived/table2/rl_only/repeat_1')['status'])"
```

After all 12 physical `repeat_1` tests have completed, first create/validate each
true-PPO bridge, then run the following discovery-only grouping check. It must show
the LWM source once and two logical aliases, never a second Full-Reward physical
run:

```powershell
& $py -u -c "import sys; sys.path.insert(0,'chapter2_region_detection'); from formal_experiments.evaluation.generate_paper_figures import discover_formal_runs, group_formal_rows; a,r=discover_formal_runs('chapter2_region_detection/outputs/formal_v3'); g,i=group_formal_rows(a,r); print('issues=',i); print('rows=',sorted(f'{t}/{row}:{len(v)}' for (t,row),v in g.items()))"
```

Only after the user authorizes five repeats and every row has `1..5` should the
final writers be run (these are intentionally not executed for the repeat-1
snapshot):

```powershell
& $py -m formal_experiments.evaluation.aggregate_tables --input-root .\chapter2_region_detection\outputs\formal_v3 --output-dir .\chapter2_region_detection\outputs\formal_v3\aggregate_tables\paper_final
& $py -m formal_experiments.evaluation.generate_paper_figures --input-root .\chapter2_region_detection\outputs\formal_v3 --output-dir .\chapter2_region_detection\outputs\formal_v3\paper_figures
```

Expected current result: no final CSV/Markdown/SVG/PNG is emitted; the only
acceptable repeat-1 deliverable is this `PROVISIONAL` readiness/audit record.
