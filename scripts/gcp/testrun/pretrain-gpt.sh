#!/bin/bash

cd $WORKDIR
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=INFO
# Runs the "345M" parameter model
# export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
#     if args.sequence_parallel:
#         raise RuntimeError(
#             "Using sequence parallelism requires setting the environment variable "
#             "CUDA_DEVICE_MAX_CONNECTIONS to 1")
#     if args.async_tensor_model_parallel_allreduce:
#         raise RuntimeError(
#             "Using async gradient all reduce requires setting the environment "
#             "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

#TODO: add retry logic in USE_NSYS = 1

if [ "$USE_NSYS" -eq 1 ]; then
    timeout "${MAX_RUNTIME_PER_EXPERIMENT}m" $NSYS_CMD torchrun $TORCHRUN_ARGS pretrain_gpt.py \
        $FIXED_ARGS \
        $SEARCH_ARGS \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        >>$LOGDIR/$RUNNAME/torchrun-rank${NODE_RANK}.log 2>&1
    ret=$?
    retry_counter=1
    # echo "nsys exit code: $ret"
    while [[ $ret -ne 0 && $ret -ne 124 && $retry_counter -lt 5 ]]; do
        # echo "nsys failed. Retrying in 60 seconds..."
        echo "nsys returned: $ret, retrying in 60 seconds..." >>$LOGDIR/$RUNNAME/torchrun-rank${NODE_RANK}.log
        sleep 60
        timeout "${MAX_RUNTIME_PER_EXPERIMENT}m" $NSYS_CMD torchrun $TORCHRUN_ARGS pretrain_gpt.py \
            $FIXED_ARGS \
            $SEARCH_ARGS \
            $GPT_ARGS \
            $DATA_ARGS \
            $OUTPUT_ARGS \
            >>$LOGDIR/$RUNNAME/torchrun-rank${NODE_RANK}.log 2>&1
        ret=$?
        retry_counter=$((retry_counter + 1))
    done

    # while [ $? -ne 0 ]; do
    #     echo "nsys failed. Retrying in 60 seconds..."
    #     sleep 60
    #     timeout "${MAX_RUNTIME_PER_EXPERIMENT}m" $NSYS_CMD torchrun $TORCHRUN_ARGS pretrain_gpt.py \
    #         $FIXED_ARGS \
    #         $SEARCH_ARGS \
    #         $GPT_ARGS \
    #         $DATA_ARGS \
    #         $OUTPUT_ARGS \
    #         >$LOGDIR/$RUNNAME/torchrun-rank${NODE_RANK}.log 2>&1
    #     echo "nsys exit code: $?"
    #     retry_counter=$((retry_counter + 1))
    # done
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
