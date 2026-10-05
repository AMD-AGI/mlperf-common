import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


_MODULE_PATH = Path(__file__).parents[1] / "primus_mllog" / "mlperf_logger.py"
_SPEC = importlib.util.spec_from_file_location("mlperf_logger_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
MLPerfLogger = _MODULE.MLPerfLogger


def _megatron_args():
    return SimpleNamespace(
        global_batch_size=32,
        micro_batch_size=1,
        data_parallel_size=8,
        eval_iters=1,
        train_samples=1024,
        train_iters=32,
        seq_length=128,
        seed=1234,
        iteration=0,
        optimizer="adam",
        lr=1e-4,
        weight_decay=0.1,
        min_lr=0.0,
        lr_decay_iters=32,
        lr_decay_style="cosine",
    )


class PrecisionConfigTest(unittest.TestCase):
    def test_emits_training_6_1_precision_keys(self):
        env = {
            "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR": "mxfp4",
            "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN": "bfloat16",
            "MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM": "bfloat16",
        }
        with patch.dict(os.environ, env, clear=False):
            configs = MLPerfLogger().extract_mlperf_configs(_megatron_args())

        self.assertEqual(configs["lowest_numerical_precision_in_linear"], "mxfp4")
        self.assertEqual(configs["lowest_numerical_precision_in_attn"], "bfloat16")
        self.assertEqual(configs["lowest_numerical_precision_in_comm"], "bfloat16")
        self.assertNotIn("lowest_numerical_precision_linear", configs)

    def test_accepts_legacy_linear_environment_variable(self):
        with patch.dict(
            os.environ,
            {"MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR": "mxfp4"},
            clear=True,
        ):
            configs = MLPerfLogger().extract_mlperf_configs(_megatron_args())

        self.assertEqual(configs["lowest_numerical_precision_in_linear"], "mxfp4")


if __name__ == "__main__":
    unittest.main()
