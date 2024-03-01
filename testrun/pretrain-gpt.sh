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
    timeout "${MAX_RUNTIME_PER_EXPERIMENT}m" $NSYS_CMD torchrun $TORCHRUN_ARGS pretrain_gpt.py \
        $FIXED_ARGS \
        $SEARCH_ARGS \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        >$LOGDIR/$RUNNAME/torchrun-rank${NODE_RANK}.log 2>&1
else
    timeout "${MAX_RUNTIME_PER_EXPERIMENT}m" torchrun $TORCHRUN_ARGS pretrain_gpt.py \
        $FIXED_ARGS \
        $SEARCH_ARGS \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        >$LOGDIR/$RUNNAME/torchrun-rank${NODE_RANK}.log 2>&1
    # --load $CHECKPOINT_PATH
    # --save $CHECKPOINT_PATH

fi

exit 0
