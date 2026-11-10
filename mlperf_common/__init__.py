###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MLPerf logging utilities with optional Primus/Megatron integration."""

from importlib import import_module

from .mlperf_logger import MLPerfLogger, ThroughputTimer
from .mlperf_sft import MLPerfSFTLogger

_OPTIONAL_EXPORTS = {
    "MLPerfMegatronPretrainTrainer": ".mlperf_pretrain_trainer",
    "run_synthetic_warmup": ".warmup",
    "reset_fp8_state": ".warmup",
    "seed_fp8_amax": ".warmup",
}


def __getattr__(name):
    module_name = _OPTIONAL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "MLPerfLogger",
    "ThroughputTimer",
    "MLPerfSFTLogger",
    *_OPTIONAL_EXPORTS,
]
