"""Read-only execution/GPU readiness audit for formal-v3 baselines."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

# Confirmed by the root task in an elevated host context on 2026-09-19.  A
# subprocess failure inside a restricted worker must not override host evidence.
CC4_VENV_HOST_CONFIRMATION = {
    "host_accessible": True,
    "python_version": "3.11.9",
    "torch_version": "2.14.0+cpu",
    "torch_cuda_available": False,
    "evidence": "root_task_elevated_host_check_2026-09-19",
}


def _nvidia_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    gpus = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4:
            gpus.append({"name": fields[0], "memory_total_mib": int(fields[1]),
                         "memory_free_mib": int(fields[2]), "driver": fields[3]})
    return {"available": bool(gpus), "gpus": gpus}


def _torch_inventory() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"importable": False, "error": repr(exc)}
    cuda = bool(torch.cuda.is_available())
    return {
        "importable": True, "version": str(torch.__version__),
        "built_cuda": torch.version.cuda, "cuda_available": cuda,
        "device_count": int(torch.cuda.device_count()),
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def _venv_launcher(path: Path) -> dict[str, Any]:
    executable = path / "Scripts" / "python.exe"
    config = path / "pyvenv.cfg"
    result: dict[str, Any] = {
        "path": str(path),
        "sandbox_launcher_visible": executable.is_file(),
        "host_confirmation": dict(CC4_VENV_HOST_CONFIRMATION),
    }
    if config.is_file():
        result["pyvenv_cfg"] = config.read_text(encoding="utf-8", errors="replace").splitlines()
    if not executable.is_file():
        result["sandbox_accessible"] = False
        return result
    completed = subprocess.run([str(executable), "--version"], capture_output=True, text=True, timeout=10)
    result.update({"sandbox_accessible": completed.returncode == 0,
                   "version_output": (completed.stdout + completed.stderr).strip()})
    return result


def build_report() -> dict[str, Any]:
    prior_source = (ROOT / "baselines" / "priorrl_cc4" / "policy.py").read_text(encoding="utf-8")
    formal_source = (ROOT / "formal_experiments" / "evaluation" / "run_formal_suite.py").read_text(encoding="utf-8")
    rsmbrl_source = (ROOT / "baselines" / "rsmbrl_cc4" / "runtime.py").read_text(encoding="utf-8")
    return {
        "schema": "formal_v3_execution_preflight_v1",
        "read_only": True,
        "python": {"executable": sys.executable, "version": sys.version},
        "nvidia": _nvidia_inventory(),
        "torch": _torch_inventory(),
        "cc4_venv": _venv_launcher(ROOT / ".venv_cc4"),
        "methods": {
            "DCA-CC4 (adapted)": {
                "gpu_useful": False, "execution": "CPU CC4 environment + tabular/rule inference",
                "formal_runner_ready": False,
            },
            "RSMBRL-CC4": {
                "gpu_useful": True,
                "device_parameter_wired": "device=device" in rsmbrl_source,
                "formal_runner_ready": False,
            },
            "PriorRL-PPO-CC4": {
                "gpu_useful": True,
                "state_device_explicit": "device=" in prior_source.split("def forward(self, state):", 1)[1].split("def act", 1)[0],
                "formal_runner_ready": False,
                "cache_blocker": True,
            },
        },
        "formal_suite": {
            "has_formal_validation": '"validate-formal"' in formal_source,
            "has_formal_execution": "run-formal" in formal_source,
            "required_repeats": 5, "episodes_per_repeat": 100, "ticks_per_episode": 500,
        },
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
