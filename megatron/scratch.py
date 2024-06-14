Traceback (most recent call last):
  File "pretrain_gpt.py", line 207, in <module>
    pretrain(train_valid_test_datasets_provider,
  File "/workspace/Megatron-LM/megatron/training.py", line 257, in pretrain
    iteration, num_floating_point_operations_so_far = train(
  File "/workspace/Megatron-LM/megatron/training.py", line 964, in train
    train_step(forward_step_func,
  File "/workspace/Megatron-LM/megatron/training.py", line 529, in train_step
    losses_reduced = forward_backward_func(
  File "/workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 1248, in forward_backward_pipelining_without_interleaving
    output_tensor = forward_step(
  File "/workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 200, in forward_step
    output_tensor = loss_func(output_tensor)
  File "pretrain_gpt.py", line 112, in loss_func
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
RuntimeError: The size of tensor a (1048576) must match the size of tensor b (1024) at non-singleton dimension 0