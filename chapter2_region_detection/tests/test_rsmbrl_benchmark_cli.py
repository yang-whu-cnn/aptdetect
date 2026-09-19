import unittest
from pathlib import Path


class TestRSMBRLBenchmarkCli(unittest.TestCase):
    def test_benchmark_has_warmup_repeats_and_cuda_synchronization(self):
        source = (Path(__file__).resolve().parents[1] / "baselines" / "rsmbrl_cc4" / "benchmark.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--warmup"', source)
        self.assertIn('parser.add_argument("--repeats"', source)
        self.assertIn("torch.cuda.synchronize()", source)
        self.assertIn('"seconds_per_call_median"', source)


if __name__ == "__main__":
    unittest.main()
