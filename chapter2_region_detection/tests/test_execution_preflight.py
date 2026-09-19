import unittest
from unittest.mock import patch

from formal_experiments.evaluation.execution_preflight import build_report


class TestExecutionPreflight(unittest.TestCase):
    @patch("formal_experiments.evaluation.execution_preflight._venv_launcher")
    @patch("formal_experiments.evaluation.execution_preflight._torch_inventory")
    @patch("formal_experiments.evaluation.execution_preflight._nvidia_inventory")
    def test_report_separates_gpu_hardware_runtime_and_runner_readiness(self, nvidia, torch, venv):
        nvidia.return_value = {"available": True, "gpus": [{"name": "test"}]}
        torch.return_value = {"importable": True, "cuda_available": False}
        venv.return_value = {
            "sandbox_launcher_visible": True,
            "sandbox_accessible": False,
            "host_confirmation": {"host_accessible": True},
        }
        report = build_report()
        self.assertTrue(report["nvidia"]["available"])
        self.assertFalse(report["torch"]["cuda_available"])
        self.assertFalse(report["cc4_venv"]["sandbox_accessible"])
        self.assertTrue(report["cc4_venv"]["host_confirmation"]["host_accessible"])
        self.assertFalse(report["formal_suite"]["has_formal_execution"])
        self.assertFalse(report["methods"]["DCA-CC4 (adapted)"]["gpu_useful"])
        self.assertTrue(report["methods"]["RSMBRL-CC4"]["gpu_useful"])


if __name__ == "__main__":
    unittest.main()
