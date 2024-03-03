Usage: bash start-search.sh




args known to increase training efficiency but might or might not impact stability: 

--recompute-activations # action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.' #activation checkpointing
--recompute-granularity=full #['full', 'selective'] let's start with full

--distribute-saved-activations #action='store_true',
                       help='If set, distribute recomputed activations '
                       'across model parallel group.'

--recompute-method=uniform #type=str, default=None, #let's do uniform
                       choices=['uniform', 'block'],
                       help='1) uniform: uniformly divide the total number of '
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'
                       'default) do not apply activations recompute to any layers'

--recompute-num-layers=uniform, type=int, default=None,
                       help='1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.'

group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')

--use-flash-attn #action='store_true',help='use FlashAttention implementation of attention. ''https://arxiv.org/abs/2205.14135'


--use-mcore-models #action='store_true', help='Use the implementation from megatron core'

--use-distributed-optimizer #action='store_true', help='Use distributed optimizer.'


--sequence-parallel # action='store_true',
                       help='Enable sequence parallel optimization.')

--overlap-grad-reduce', action='store_true',
                       default=False, help='If set, overlap DDP grad reduce.')

--no-delay-grad-reduce', action='store_false',
                       help='If not set, delay / synchronize grad reductions in all but first PP stage.',
                       dest='delay_grad_reduce')
--overlap-param-gather', action='store_true',
                       default=False, help='If set, overlap param all-gather in distributed optimizer.')
    
group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None, help='If set, affine parallel weights '
                       'initialization uses CPU' )
    


SEARCH_ARGS=

--no-clone-scatter-output-in-embedding, action='store_false',
                       help='If not set, clone the output of the scatter in embedding layer to GC original tensor.',
                       dest='clone_scatter_output_in_embedding'

--tp-comm-overlap # action='store_true', help = 'Enables the '
                       ' overlap of Tensor parallel communication and GEMM kernels.')
--tp-comm-overlap-cfg # type=str, default=None, 
                       help = 'Config file when tp_comm_overlap is enabled.')
--tensor-model-parallel-size # type=int, default=1,
                       help='Degree of tensor model parallelism.')

--pipeline-model-parallel-size # type=int, default=1,
                       help='Degree of pipeline model parallelism.')

--num-layers-per-virtual-pipeline-stage # type=int, default=None,
                       help='Number of layers per virtual pipeline stage'
    

group.add_argument('--empty-unused-memory-level', default=0, type=int,
                       choices=[0, 1, 2],
                       help='Call torch.cuda.empty_cache() each iteration '
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')

    group.add_argument('--context-parallel-size', type=int, default=1,
                       help='Degree of context parallelism.')



PROFILING_ARGS=

'--profile', action='store_true',
                       help='Enable nsys profiling. When using this option, nsys '
                       'options should be specified in commandline. An example '
                       'nsys commandline is `nsys profile -s none -t nvtx,cuda '
                       '-o <path/to/output_file> --force-overwrite true '
                       '--capture-range=cudaProfilerApi '
                       '--capture-range-end=stop`.' 


group.add_argument('--profile-step-start', type=int, default=10,
                       help='Global step to start profiling.')
    group.add_argument('--profile-step-end', type=int, default=12,
                       help='Global step to stop profiling.')
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[0],
                       help='Global ranks to profile.')





    




    
    
    





ADVANCED_ARGS=

    group.add_argument('--manual-gc', action='store_true',
                       help='Disable the threshold-based default garbage '
                       'collector and trigger the garbage collection manually. '
                       'Manual garbage collection helps to align the timing of '
                       'the collection across ranks which mitigates the impact '
                       'of CPU-associated jitters. When the manual gc is enabled, '
                       'garbage collection is performed only at the start and the '
                       'end of the validation routine by default.')
    group.add_argument('--manual-gc-interval', type=int, default=0,
                       help='Training step interval to trigger manual garbage '
                       'collection. When the value is set to 0, garbage '
                       'collection is not triggered between training steps.')
    group.add_argument('--no-manual-gc-eval', action='store_false',
                       help='When using manual garbage collection, disable '
                       'garbage collection at the start and the end of each '
                       'evaluation run.', dest='manual_gc_eval')

    group.add_argument('--use-ring-exchange-p2p', action='store_true',
                       default=False, help='If set, use custom-built ring exchange '
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')






args known to reduce efficiency: 
 group.add_argument('--disable-tp-comm-split-ag', action='store_false', 
                       help = 'Disables the All-Gather overlap with fprop GEMM.',
                       dest='tp_comm_split_ag')
    group.add_argument('--disable-tp-comm-split-rs', action='store_false', 
                       help = 'Disables the Reduce-Scatter overlap with fprop GEMM.',
                       dest='tp_comm_split_rs')
    group.add_argument('--disable-tp-comm-bulk-dgrad', action='store_false', 
                       help = 'Disables the All-Gather overlap with bprop activation gradient GEMM.',
                       dest='tp_comm_bulk_dgrad')
    group.add_argument('--disable-tp-comm-bulk-wgrad', action='store_false', 
                       help = 'Disables the Reduce-Scatter overlap with bprop weight gradient GEMM.',
                       dest='tp_comm_bulk_wgrad')

    group.add_argument('--no-async-tensor-model-parallel-allreduce',
                       action='store_false',
                       help='Disable asynchronous execution of '
                       'tensor-model-parallel all-reduce with weight '
                       'gradient compuation of a column-linear layer.',
                       dest='async_tensor_model_parallel_allreduce')
    group.add_argument('--no-persist-layer-norm', action='store_true',
                       help='Disable using persistent fused layer norm kernel. '
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')


    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false',
                       help='Disable fusing gradient accumulation to weight '
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')

    group.add_argument('--no-overlap-p2p-communication', action='store_false',
                       help='overlap pipeline parallel communication with forward and backward chunks',
                       dest='overlap_p2p_comm')


    group.add_argument('--delay-param-gather', action='store_true',
                       default=False, help='If set, delay / synchronize param all-gathers in all but first PP stage.')

    --no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='If not set, use scatter/gather to optimize communication of tensors in pipeline.',
                       dest='scatter_gather_tensors_in_pipeline')
        


irrelevant for gpt: group.add_argument('--pipeline-model-parallel-split-rank',
                       type=int, default=None,
                       help='Rank where encoder and decoder should be split.')







from megatron/core/parallel_state.py:


        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.


    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.