import torch
import torch.nn
import torch.distributed as dist

dist.init_process_group('gloo')
rank = dist.get_rank()

if rank == 0:
    print("hello!")
if rank == 1:
    print("dooshbag")