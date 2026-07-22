###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MLPERF-enabled MegatronPretrainTrainer."""

import os
import time
from typing import Any

from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

from .mlperf_logger import MLPerfLogger, ThroughputTimer

try:
    from rpdTracerControl import rpdTracerControl
    print("[MLPerf Train] rpdTracerControl loaded successfully, RPD profiling enabled.")
except ImportError:
    print("[MLPerf Train] rpdTracerControl not available, RPD profiling will be disabled.")
    rpdTracerControl = None 


class MLPerfMegatronPretrainTrainer(MegatronPretrainTrainer):
    """MegatronPretrainTrainer with MLPERF logging."""
    
    def __init__(self, *args, **kwargs):
        """Initialize trainer with MLPERF logging components."""
        super().__init__(*args, **kwargs)
        
        self.mllogger = MLPerfLogger()
        self.throughput_timer = None
        self.train_start_time = None
        self.train_stop_time = None
        self.last_validation_loss = None
        self.train_loss_log_freq = int(os.getenv("MLLOG_TRAIN_LOSS_LOG_FREQ", "1"))
        self.block_tput_log = os.getenv("MLLOG_BLOCK_TPUT_LOG", "0") == "1"
        self.target_eval_loss = float(os.getenv("MLLOG_TARGET_EVAL_LOSS", "0.0"))
        self.is_target_reached = False
        
        from mlperf_logging.mllog import constants
        self.mllogger.log_event_all_ranks(key=constants.CACHE_CLEAR, value=True)
        self.mllogger.log_event_all_ranks(key=constants.INIT_START, value=None)
        profiler = os.getenv("PROFILER", '')
        self.profile_segment = os.getenv("PROFILE_SEGMENT", 'train')
        self.rpd = None
        if profiler == 'rpd' and rpdTracerControl is not None:
            rpdTracerControl.setFilename(name=f"trace.rpd", append=True)
            self.rpd = rpdTracerControl()
            enable_python_trace = os.getenv("ENABLE_PYTHON_TRACE", "false").strip().lower() in {"1", "true", "yes", "on"}
            self.rpd.setPythonTrace(doTrace=enable_python_trace)
    
    def init(self):
        """Configure MLLOG and log init events."""
        from megatron.training import get_args
        from mlperf_logging.mllog import constants
        
        result = super().init()
        
        args = get_args()
        output_file = os.getenv("MLLOG_OUTPUT_FILE", "/results/mlperf_output.log")
        configs = self.mllogger.configure(output_file, args)
        
        for key, value in configs.items():
            self.mllogger.log_event(key=key, value=value)
        
        self.mllogger.log_end(key=constants.INIT_STOP)
        
        return result
    
    def run(self, *args, **kwargs):
        from megatron.training import get_args
        from mlperf_logging.mllog import constants

        megatron_args = get_args()
        self.throughput_timer = ThroughputTimer(megatron_args.global_batch_size)
        
        result = super().run(*args, **kwargs)
        
        final_args = get_args()
        consumed_samples = final_args.consumed_train_samples
        
        duration = self.train_stop_time - self.train_start_time
        duration_minutes = duration / 60.0
        overall_throughput = consumed_samples / duration if duration > 0 else 0.0
        self.mllogger.log_event(
            key="run_duration",
            value=f"{round(duration, 2)}s -> {round(duration_minutes, 2)} minutes",
            metadata={'samples': consumed_samples}
        )
        self.mllogger.log_event(
            key="overall_throughput",
            value=round(overall_throughput, 2),
            metadata={'samples': consumed_samples}
        )
        
        return result
    
    def train(self, forward_step_func, model, optimizer, opt_param_scheduler,
              train_data_iterator, valid_data_iterator, process_non_loss_data_func,
              config, checkpointing_context, non_loss_data_func):
        """Inject MLLOG evaluation events via monkey-patching."""
        from megatron.training import get_args, training as megatron_training_module
        from mlperf_logging.mllog import constants
        from primus.modules.trainer.megatron import trainer as trainer_module
        
        megatron_args = get_args()
        
        original_eval_and_print_func = trainer_module.evaluate_and_print_results
        original_evaluate_func = megatron_training_module.evaluate
        
        def wrapped_evaluate(*args, **kwargs):
            if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == 'eval':
                self.rpd.start()

            result = original_evaluate_func(*args, **kwargs)

            if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == 'eval':
                self.rpd.stop()
                self.rpd = None
            total_loss_dict, collected_non_loss_data, should_exit = result
            
            if total_loss_dict and 'lm loss' in total_loss_dict:
                val_loss = total_loss_dict['lm loss']
                if hasattr(val_loss, 'item'):
                    self.last_validation_loss = val_loss.item()
                else:
                    self.last_validation_loss = float(val_loss)
            
            return result
        
        def wrapped_evaluate_and_print_results(*args, **kwargs):
            eval_args = get_args()
            consumed_samples = eval_args.consumed_train_samples
            
            self._log_tracked_stats(eval_args.iteration, consumed_samples)
            
            self.mllogger.log_end(key=constants.BLOCK_STOP, 
                                 metadata={constants.SAMPLES_COUNT: consumed_samples})
            self.mllogger.log_start(key=constants.EVAL_START, 
                                   metadata={constants.SAMPLES_COUNT: consumed_samples})
            
            if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == 'eval':
                self.rpd.start()

            result = original_eval_and_print_func(*args, **kwargs)

            if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == 'eval':
                self.rpd.stop()
                self.rpd = None
            
            validation_loss = self._get_validation_loss()
            
            if validation_loss is not None:
                self.mllogger.log_event(key=constants.EVAL_ACCURACY, 
                                       value=validation_loss,
                                       metadata={constants.SAMPLES_COUNT: consumed_samples})
                
                if self.target_eval_loss > 0.0 and validation_loss <= self.target_eval_loss:
                    if not self.is_target_reached:
                        self.is_target_reached = True
                        self.train_stop_time = time.time()
                        if self.mllogger._get_rank() == 0:
                            print(f"[MLPERF] Target eval loss {self.target_eval_loss} reached with validation loss {validation_loss}")
                        eval_args.train_iters = eval_args.iteration
                        eval_args.do_valid = False
                        eval_args.do_test = False
            
            self.mllogger.log_end(key=constants.EVAL_STOP, 
                                 metadata={constants.SAMPLES_COUNT: consumed_samples})
            
            if not self.is_target_reached:
                self.mllogger.log_start(key=constants.BLOCK_START, 
                                       metadata={constants.SAMPLES_COUNT: consumed_samples})
            
            
            return result
        
        trainer_module.evaluate_and_print_results = wrapped_evaluate_and_print_results
        megatron_training_module.evaluate = wrapped_evaluate

        from .warmup import run_synthetic_warmup
        self._warmup_active = True
        try:
            run_synthetic_warmup(
                self, forward_step_func, model, optimizer, opt_param_scheduler,
                config, megatron_args,
            )
        finally:
            self._warmup_active = False

        self.throughput_timer.get_throughput()

        self.mllogger.log_start(key=constants.RUN_START)
        self.train_start_time = time.time()

        self.mllogger.log_start(key=constants.EPOCH_START,
                               metadata={constants.SAMPLES_COUNT: 0})
        self.mllogger.log_start(key=constants.BLOCK_START,
                               metadata={constants.SAMPLES_COUNT: 0})

        if self.rpd and self.profile_segment == 'train':
            self.rpd.start()

        try:
            result = super().train(
                forward_step_func, model, optimizer, opt_param_scheduler,
                train_data_iterator, valid_data_iterator, process_non_loss_data_func,
                config, checkpointing_context, non_loss_data_func
            )
        finally:
            trainer_module.evaluate_and_print_results = original_eval_and_print_func
            megatron_training_module.evaluate = original_evaluate_func

        if self.rpd and self.profile_segment == 'train':
            self.rpd.stop()
            self.rpd = None

        final_args = get_args()
        consumed_samples = final_args.consumed_train_samples
        self.mllogger.log_end(key=constants.EPOCH_STOP,
                             metadata={constants.SAMPLES_COUNT: consumed_samples})

        if self.is_target_reached:
            self.mllogger.log_end(
                key=constants.RUN_STOP,
                metadata={
                    constants.SAMPLES_COUNT: consumed_samples,
                    constants.STATUS: constants.SUCCESS,
                }
            )
        else:
            self.train_stop_time = time.time()
            self.mllogger.log_end(
                key=constants.RUN_STOP,
                metadata={
                    constants.SAMPLES_COUNT: consumed_samples,
                    constants.STATUS: constants.ABORTED,
                }
            )
        
        return result
    
    def train_step(self, forward_step_func, data_iterator, model, optimizer,
                   opt_param_scheduler, config, no_optimizer_post_validation=False):
        if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == 'train_step':
            self.rpd.start()
        self.throughput_timer.step_start()
        result = super().train_step(
            forward_step_func, data_iterator, model, optimizer,
            opt_param_scheduler, config, no_optimizer_post_validation
        )
        self.throughput_timer.step_stop()
        
        loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad = result
        
        if not skipped_iter and loss_dict and not getattr(self, '_warmup_active', False):
            self._on_train_step_end(loss_dict)
        if self.rpd and self.mllogger._get_rank() == 0 and self.profile_segment == 'train_step':
            self.rpd.stop()
            self.rpd = None
        
        return result
    
    def _on_train_step_end(self, loss_dict):
        from megatron.training import get_args
        from mlperf_logging.mllog import constants
        
        args = get_args()
        iteration = args.iteration
        consumed_samples = args.consumed_train_samples
        
        if self.train_loss_log_freq <= 0 or iteration % self.train_loss_log_freq != 0:
            return
        
        loss_value = loss_dict.get('lm loss')
        if loss_value is None:
            return
        
        if isinstance(loss_value, (tuple, list)):
            loss_value = loss_value[0]
        if hasattr(loss_value, 'item'):
            loss_value = loss_value.item()
        
        learning_rate = None
        for param_group in self.optimizer.param_groups:
            if not param_group.get("is_decoupled_lr", False):
                learning_rate = param_group["lr"]
                break
        
        self.mllogger.log_event(
            key="train_loss",
            value=loss_value,
            metadata={
                constants.SAMPLES_COUNT: consumed_samples,
                'lr': learning_rate
            }
        )
    
    def _log_tracked_stats(self, iteration, consumed_samples):
        throughput = self.throughput_timer.get_throughput()
        if throughput is not None and self.block_tput_log:
            self.mllogger.log_event(
                key="tracked_stats",
                value={'throughput': throughput},
                metadata={'step': consumed_samples}
            )
    
    def _get_validation_loss(self):
        return getattr(self, 'last_validation_loss', None)
