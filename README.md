# primus-mllog

MLPerf-compliant training logging utilities for **ROCm Primus**.

`primus-mllog` wraps the [MLCommons `mlperf_logging`](https://github.com/mlcommons/logging)
library and integrates it with [Primus](https://github.com/AMD-AGI/Primus) Megatron
and Megatron-Bridge training workloads. It emits the `:::MLLOG` events required for an
MLPerf Training submission (init, run, block, eval, and stop markers) with correct
rank-awareness, timing, and convergence handling — plus a synthetic-data warmup that
keeps kernel-compilation and allocator overhead out of the timed run.

---

## Features

- **Rank-aware MLPerf logging** — a thin wrapper (`MLPerfLogger`) around `mllog` that
  only emits from rank 0 (with an all-ranks escape hatch for `CACHE_CLEAR` / `INIT_START`).
- **Drop-in pretraining trainer** — `MLPerfMegatronPretrainTrainer` subclasses the Primus
  `MegatronPretrainTrainer` and injects all required MLPerf events with no changes to your
  training loop.
- **Standalone SFT logger** — `MLPerfSFTLogger` provides explicit lifecycle hooks for
  Megatron-Bridge / fine-tuning loops (e.g. `llama2_70b_lora`) that don't share the Primus
  trainer base class.
- **Automatic config extraction** — pulls hyperparameters (batch size, LR schedule,
  optimizer, parallelism, seed, sample counts, …) straight from Megatron `args` and maps
  them to MLPerf constants.
- **Convergence-aware `RUN_STOP`** — stops timing and logs `SUCCESS` as soon as the target
  eval loss is reached, otherwise logs `ABORTED`.
- **Throughput measurement** — `ThroughputTimer` accumulates in-step time only, excluding
  eval and inter-step overhead, for accurate samples/sec.
- **Synthetic warmup** — runs N forward+backward passes on random tokens before the timed
  run to amortize Triton/CK/hipBLASLt JIT, NCCL init, and allocator layout, then fully
  restores model/optimizer/scheduler/FP8/FP4 state so weights never move.
- **FP8 & FP4 aware** — resets and re-seeds Transformer Engine delayed-scaling state after
  warmup; opt-in MXFP4 / NVFP4 warmup via `WARMUP_RECIPE`.
- **Optional RPD profiling** — segment-scoped ROCm profiling (`train`, `eval`, `train_step`)
  via `rpdTracerControl`.

---

## Requirements

- ROCm-enabled PyTorch (AMD Instinct GPUs, e.g. MI300X / MI355X)
- [`mlperf-logging`](https://github.com/mlcommons/logging)
- [Primus](https://github.com/AMD-AGI/Primus) (for `MLPerfMegatronPretrainTrainer`)
- Megatron-LM / Megatron-Core
- `numpy`
## Installation

Install directly from GitHub (recommended for Dockerfiles — pin to a commit for
reproducible builds):

```bash
pip install git+https://github.com/AMD-AGI/mlperf-common.git@7f0de9b01a0c4153a8bc7a8edb292b3748195e32
```

In a Dockerfile:

```dockerfile
# git must be available in the image
RUN apt-get update && apt-get install -y --no-install-recommends git

RUN pip install git+https://github.com/AMD-AGI/mlperf-common.git@7f0de9b01a0c4153a8bc7a8edb292b3748195e32
```

For local development:

```bash
git clone https://github.com/AMD-AGI/mlperf-common.git
cd mlperf-common
pip install -e .
```

---

## Usage

### Pretraining (Primus Megatron)

Swap your trainer for the MLPerf-enabled subclass. All MLPerf events are handled
automatically:

```python
from primus_mllog import MLPerfMegatronPretrainTrainer

trainer = MLPerfMegatronPretrainTrainer(...)  # same args as MegatronPretrainTrainer
trainer.init()
trainer.run()
```

This emits `CACHE_CLEAR`, `INIT_START` / `INIT_STOP`, `RUN_START`, `EPOCH_START`,
`BLOCK_START` / `BLOCK_STOP`, `EVAL_START` / `EVAL_STOP`, `EVAL_ACCURACY`,
`train_loss`, and a convergence-aware `RUN_STOP`.

### Supervised fine-tuning (Megatron-Bridge)

Use the standalone logger and call its lifecycle hooks from your custom training loop:

```python
from primus_mllog import MLPerfSFTLogger

logger = MLPerfSFTLogger(global_batch_size=8, micro_batch_size=1)

logger.log_cache_clear_and_init_start()
config = MLPerfSFTLogger.extract_sft_configs(
    train_gbs=8, train_mbs=1, train_iters=1024, eval_iters=48,
    seq_length=8192, seed=1234, lr=4e-4, weight_decay=1e-4,
    clip_grad=1.0, lr_warmup_iters=0, data_root="/data",
)
logger.log_init_params(config)
# ... model / optimizer / data init completes ...
logger.log_init_stop_run_start()

for step in training_loop:
    logger.on_train_step(step, loss_dict, lr, consumed_samples)
    if is_eval_step:
        logger.on_eval_start(consumed_samples)
        val_loss = run_eval()
        if logger.on_eval_end(consumed_samples, val_loss):
            break  # target reached

logger.log_run_stop(consumed_samples)
```

---

## Configuration

Behavior is driven by environment variables so submission metadata and knobs can be set
without code changes.

### Output & submission metadata

| Variable | Default | Description |
| --- | --- | --- |
| `MLLOG_OUTPUT_FILE` | `/results/mlperf_output.log` | Path to the MLPerf log file. |
| `MLLOG_SAVE_TO_FILE` | `1` | Write to file (`0`/`false`/`no` → stdout only). |
| `MLLOG_SUBMISSION_BENCHMARK` | `""` | Benchmark name (e.g. `llama2_70b_lora`). |
| `MLLOG_SUBMISSION_DIVISION` | `""` | `closed` / `open`. |
| `MLLOG_SUBMISSION_STATUS` | `""` | e.g. `onprem`, `cloud`. |
| `MLLOG_SUBMISSION_ORG` | `""` | Submitting organization. |
| `MLLOG_SUBMISSION_PLATFORM` | `""` | Platform (e.g. `MI355X`). |
| `MLLOG_CONFIG_FILENAME` | `""` | Config file name recorded in the log. |

### Parallelism & precision metadata

| Variable | Default | Description |
| --- | --- | --- |
| `MLLOG_TENSOR_PARALLELISM` | `1` | Tensor-parallel size. |
| `MLLOG_PIPELINE_PARALLELISM` | `1` | Pipeline-parallel size. |
| `MLLOG_CONTEXT_PARALLELISM` | `1` | Context-parallel size. |
| `MLLOG_EXPERT_PARALLELISM` | `1` | Expert-parallel size. |
| `MLLOG_MICRO_BATCH_SIZE` | `1` | Micro-batch size (metadata). |
| `MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR` | `""` | Lowest linear-layer precision. |

### Training / logging behavior

| Variable | Default | Description |
| --- | --- | --- |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | `1` (pretrain) / `10` (SFT) | Steps between `train_loss` events. |
| `MLLOG_TARGET_EVAL_LOSS` | `0.0` (pretrain) / `0.925` (SFT) | Convergence target; `RUN_STOP=SUCCESS` when reached. |
| `MLLOG_BLOCK_TPUT_LOG` | `0` | Log per-block throughput as `tracked_stats`. |

### Synthetic warmup

| Variable | Default | Description |
| --- | --- | --- |
| `SYNTH_WARMUP_STEPS` | `3` | Warmup forward/backward passes; `0` disables. |
| `SYNTH_WARMUP_EMPTY_CACHE` | `1` | Call `torch.cuda.empty_cache()` each warmup step. |
| `WARMUP_RECIPE` | `""` | `""`/`bf16` (no autocast), `fp8_hybrid`, `fp8_e4m3`, `fp4_mxfp4`, `fp4_nvfp4`. |
| `WARMUP_FP8_RECIPE` | `""` | Legacy alias: `hybrid`/`e4m3` → `fp8_hybrid`/`fp8_e4m3`. |
| `WARMUP_FP8_HISTORY_LEN` | `4` | amax history length for FP8 recipes. |

### RPD profiling

| Variable | Default | Description |
| --- | --- | --- |
| `PROFILER` | `""` | Set to `rpd` to enable RPD tracing (requires `rpdTracerControl`). |
| `PROFILE_SEGMENT` | `train` | Segment to profile: `train`, `eval`, or `train_step`. |
| `ENABLE_PYTHON_TRACE` | `false` | Include Python-level trace in RPD output. |

---

## Package layout

| Module | Contents |
| --- | --- |
| `mlperf_logger.py` | `MLPerfLogger` (rank-aware `mllog` wrapper, config extraction) and `ThroughputTimer`. |
| `mlperf_pre_training.py` | `MLPerfMegatronPretrainTrainer` — Primus pretrain trainer with MLPerf logging. |
| `mlperf_sft.py` | `MLPerfSFTLogger` — standalone lifecycle-hook logger for SFT / fine-tuning. |
| `warmup.py` | Synthetic-data warmup, FP8/FP4 state reset, and optimizer/scheduler save-restore. |

---

## License

Copyright (c) 2025, Advanced Micro Devices, Inc. Licensed under the
[Apache License 2.0](LICENSE).
