# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """
    TODO
    """
    if eval_mode:
        model.eval().requires_grad_(False)

    # FSDP lazy init expects all original floating parameters in a shard group
    # to have one dtype. Some checkpoints keep small params in fp32 even when
    # loaded with torch_dtype, so normalize before fully_shard wraps modules.
    model.to(device=device, dtype=param_dtype)

    if dist.is_initialized():
        dist.barrier()
        model = shard_fn(model, param_dtype=param_dtype)

    return model


def init_distributed(world_size, local_rank, rank):
    # if world_size > 1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            rank=rank,
                            world_size=world_size)

def dist_mean(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
