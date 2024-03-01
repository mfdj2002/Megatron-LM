#!/bin/bash

cd $WORKDIR

# Runs the "345M" parameter model
#export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 #?????
# if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
#     if args.sequence_parallel:
#         raise RuntimeError(
#             "Using sequence parallelism requires setting the environment variable "
#             "CUDA_DEVICE_MAX_CONNECTIONS to 1")
#     if args.async_tensor_model_parallel_allreduce:
#         raise RuntimeError(
#             "Using async gradient all reduce requires setting the environment "
#             "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

if [ "$USE_NSYS" -eq 1 ]; then
    $NSYS_CMD torchrun $TORCHRUN_ARGS pretrain_gpt.py \
        $FIXED_ARGS \
        $SEARCH_ARGS \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        >$LOGDIR/$RUN_UID/torchrun-rank${NODE_RANK}.log 2>&1
else
    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw,pstate,pcie.link.gen.max,pcie.link.gen.current --format=csv -l 5 >"$LOGDIR/$RUN_UID/nvidia-smi-rank${NODE_RANK}.csv" &
    nvidia_smi_pid=$!
    dool --more --output "$LOGDIR/$RUN_UID/dool-rank${NODE_RANK}.csv" 5 &
    dool_pid=$!

    torchrun $TORCHRUN_ARGS pretrain_gpt.py \
        $FIXED_ARGS \
        $SEARCH_ARGS \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        >$LOGDIR/$RUN_UID/torchrun-rank${NODE_RANK}.log 2>&1
    # --load $CHECKPOINT_PATH
    # --save $CHECKPOINT_PATH

    kill $nvidia_smi_pid $dool_pid 2>/dev/null || true
fi

exit 0
