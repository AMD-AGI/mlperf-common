"""Check the public namespace without requiring a GPU training stack."""

import subprocess
import sys
import unittest


class PackageImportTest(unittest.TestCase):
    def test_basic_loggers_do_not_import_training_dependencies(self):
        subprocess.run(
            [sys.executable, "-c", '''
import importlib.abc
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"primus", "megatron", "torch", "primus_mllog"}:
            raise AssertionError(f"Unexpected training dependency: {fullname}")

sys.meta_path.insert(0, BlockTrainingImports())
from mlperf_common import MLPerfLogger, MLPerfSFTLogger, ThroughputTimer
MLPerfLogger()
assert MLPerfSFTLogger.__module__ == "mlperf_common.mlperf_sft"
assert ThroughputTimer.__module__ == "mlperf_common.mlperf_logger"
'''],
            check=True,
        )

    def test_optional_exports_resolve_and_cache_on_demand(self):
        subprocess.run(
            [sys.executable, "-c", '''
import sys
from types import ModuleType
import mlperf_common

for module_name, names in {
    "mlperf_pretrain_trainer": ["MLPerfMegatronPretrainTrainer"],
    "warmup": ["run_synthetic_warmup", "reset_fp8_state", "seed_fp8_amax"],
}.items():
    full_name = "mlperf_common." + module_name
    assert full_name not in sys.modules
    module = ModuleType(full_name)
    for name in names:
        setattr(module, name, object())
    sys.modules[full_name] = module
    for name in names:
        assert getattr(mlperf_common, name) is getattr(module, name)
        assert mlperf_common.__dict__[name] is getattr(module, name)

try:
    mlperf_common.nonexistent
except AttributeError:
    pass
else:
    raise AssertionError("Unknown exports must raise AttributeError")
'''],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
