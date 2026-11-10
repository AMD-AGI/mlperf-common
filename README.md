# mlperf-common

MLPerf-compliant training logging utilities for **ROCm Primus**.

`mlperf-common` wraps the [MLCommons `mlperf_logging`](https://github.com/mlcommons/logging)
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

The package installer installs `mlperf-logging` and `numpy`. Provision the
training stack separately in the target image. Basic logger imports do not
require that stack, but runtime rank detection uses PyTorch, and the pretraining
trainer requires compatible Primus and Megatron installations.

## Installation

Install directly from GitHub (recommended for Dockerfiles — pin to a commit for
reproducible builds). Replace `<commit-with-package-rename>` with a commit
containing the Python package rename; older commits still install the old package:

```bash
pip install "git+https://github.com/AMD-AGI/mlperf-common.git@<commit-with-package-rename>"
```

In a Dockerfile:

```dockerfile
# git must be available in the image
RUN apt-get update && apt-get install -y --no-install-recommends git

RUN pip install "git+https://github.com/AMD-AGI/mlperf-common.git@<commit-with-package-rename>"
```

For local development, check out a revision containing the rename before
installing. While the rename PR is unmerged, use its branch rather than `main`:

```bash
git clone https://github.com/AMD-AGI/mlperf-common.git
cd mlperf-common
git checkout "<commit-with-package-rename>"
python3 -m pip install -e .
```

---

## Migrating from the old package

The distribution is now `mlperf-common` and the Python namespace is
`mlperf_common`. Update imports in trainers, launchers, and SFT scripts, then
uninstall the previous distribution (`pip uninstall primus-mllog`) and install
this version. There is no `primus_mllog` compatibility package. Installing the
new distribution does not automatically uninstall the old one.

Basic logger imports do not load Primus, Megatron, or PyTorch. The pretraining
trainer and warmup helpers load their training dependencies only when requested.
The trainer class and its logging/warmup behavior are unchanged. Primus consumers
must update their imports and package pins; this rename does not itself change
Primus stage dispatch or activate MLPerf logging for ordinary `pretrain`.

### Verify the installation

Run this with the same Python interpreter/container used for training, from
outside the source checkout so an uninstalled local package cannot mask an
installation problem:

```bash
python3 -m pip check
python3 - <<'PYTHON'
from importlib.metadata import version
import mlperf_common
from mlperf_common import MLPerfLogger, MLPerfSFTLogger

print("mlperf-common:", version("mlperf-common"))
print("imported from:", mlperf_common.__file__)
MLPerfLogger()
MLPerfSFTLogger(global_batch_size=8, micro_batch_size=1)
PYTHON
```

Expect version `0.2.0` for this change and a module path in the intended
environment (or your intended checkout for an editable install). This checks
package installation and logger construction, not training or GPU execution.

Inside the prepared Primus training environment, also check the optional
trainer import:

```bash
python3 -c 'from mlperf_common import MLPerfMegatronPretrainTrainer; print(MLPerfMegatronPretrainTrainer.__module__)'
```

Expect `mlperf_common.mlperf_pretrain_trainer`. This only verifies that the
trainer can be imported; the consumer must still select it as described below.

## Usage

### Pretraining (Primus Megatron)

`MLPerfMegatronPretrainTrainer` subclasses Primus's `MegatronPretrainTrainer`.
Primus constructs it with the normal trainer context and drives its lifecycle
(`setup -> init -> train -> cleanup`). Its `setup()` emits the initialization
markers; its `train()` wraps upstream Megatron's training/evaluation functions
and restores those functions when training exits.

The public import after this rename is:

```python
from mlperf_common import MLPerfMegatronPretrainTrainer
```

This import alone does not activate MLPerf logging. Primus must select this
trainer. It provides synthetic warmup before `RUN_START`, rank-aware logging,
evaluation events, and convergence-aware `RUN_STOP`. Warmup is controlled by
`SYNTH_WARMUP_STEPS`; see the configuration tables below for all settings.

#### GPT-OSS reference: existing integration

The following references are pinned to Primus commit
`d8202b102de22a9f60d266c8193ef13b1ecba9c3`, **before its consumer imports are
updated for this package rename**:

- [GPT-OSS launch guide](https://github.com/AMD-AGI/Primus/blob/d8202b102de22a9f60d266c8193ef13b1ecba9c3/examples/mlperf/gpt_oss_20b/README.md)
- [GPT-OSS MXFP4 recipe](https://github.com/AMD-AGI/Primus/blob/d8202b102de22a9f60d266c8193ef13b1ecba9c3/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml#L15)
- [Megatron trainer selection](https://github.com/AMD-AGI/Primus/blob/d8202b102de22a9f60d266c8193ef13b1ecba9c3/primus/backends/megatron/megatron_adapter.py#L44)

The existing flow is:

1. The GPT-OSS recipe sets `stage: ${PRIMUS_STAGE:mlperf_pretrain}` in its
   trainer overrides, defaulting to `mlperf_pretrain`.
2. The Megatron adapter selects `MLPerfMegatronPretrainTrainer` for that stage.
   At the pinned revision it imports the class from `primus_mllog`.
3. The trainer installs the logging and warmup behavior described above.

Llama 8B uses the same stage-based integration. This is a reference for the
existing activation path, not an example of MLPerf running through ordinary
`pretrain`.

#### Using the renamed package with Primus

Update the consumer and package pin together:

1. In Primus's Megatron adapter, replace the old trainer import with
   `from mlperf_common import MLPerfMegatronPretrainTrainer`; update the
   dependency error message as well. Update any other consumer imports of the
   old namespace, including direct logger or warmup imports.
2. Install a commit containing this rename and uninstall the old distribution,
   as described above. A Primus revision still importing `primus_mllog` is not
   compatible with this package version.
3. Keep the recipe's `mlperf_pretrain` stage while using this existing dispatch
   path. Validate the updated consumer in the target training image, including
   warmup, MLPerf event ordering, evaluation, and convergence stopping.

Keep existing submission Docker/code pins together if they still use the old
namespace. Merely changing the install URL or setting `PRIMUS_STAGE=pretrain`
does not preserve this MLPerf integration.

#### Removing the dedicated stage: follow-up integration work

The package rename does **not** implement an MLPerf enable flag or hook for
ordinary `pretrain`. Primus needs an explicit activation path that preserves
this trainer's logging and warmup behavior before the dedicated stage can be
retired. Those features can remain in `mlperf_common`; moving them into Primus
is not required by the rename. There is no normal-`pretrain` usage command to
recommend until that consumer integration exists and is validated.

### Supervised fine-tuning (Megatron-Bridge)

Use the standalone logger and call its lifecycle hooks from your custom training
loop. The following is an integration template, not a standalone executable:
your application supplies the model/data initialization, `training_loop`, step
metrics, evaluation scheduling, and `run_eval()`. It must run in the prepared
training environment. Ensure the parent directory of `MLLOG_OUTPUT_FILE` exists
and is writable, or set `MLLOG_SAVE_TO_FILE=0` for stdout-only logging.

```python
from mlperf_common import MLPerfSFTLogger

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

## Troubleshooting the migration

| Symptom | What to check |
| --- | --- |
| `No module named mlperf_common` | The installed revision may predate the rename, or installation used a different interpreter. Check the version and module path with the verification commands above. |
| `No module named primus_mllog` after upgrading | A consumer still imports the old namespace. Update its imports and package pin together, or retain the previous matching consumer/package versions. |
| Importing the trainer fails on `primus`, `megatron`, or another training dependency | Run inside the prepared training image and verify compatible runtime dependencies. Installing this package alone does not install the training stack. |
| Training runs with no MLPerf events or synthetic warmup | Check that Primus selects `MLPerfMegatronPretrainTrainer`; importing the package or using ordinary `pretrain` does not activate it. Check `SYNTH_WARMUP_STEPS` if logging works but warmup is disabled. |
| Opening the log file fails | Create a writable parent directory for `MLLOG_OUTPUT_FILE`, or set `MLLOG_SAVE_TO_FILE=0`. |

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
| `MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR` | `""` | Lowest linear-layer precision. |
| `MLLOG_LOWEST_NUMERICAL_PRECISION_IN_ATTN` | `""` | Lowest attention precision. |
| `MLLOG_LOWEST_NUMERICAL_PRECISION_IN_COMM` | `""` | Lowest communication precision. |

`MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR` remains supported as a deprecated
alias for `MLLOG_LOWEST_NUMERICAL_PRECISION_IN_LINEAR`.

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
| `mlperf_pretrain_trainer.py` | `MLPerfMegatronPretrainTrainer` — Primus pretrain trainer with MLPerf logging. |
| `mlperf_sft.py` | `MLPerfSFTLogger` — standalone lifecycle-hook logger for SFT / fine-tuning. |
| `warmup.py` | Synthetic-data warmup, FP8/FP4 state reset, and optimizer/scheduler save-restore. |

---

## License

Copyright (c) 2025, Advanced Micro Devices, Inc. Licensed under the
[Apache License 2.0](LICENSE).
