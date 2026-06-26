import torch
import torch.distributed as dist
import torch.nn as nn
from buffer import *

dist.init_process_group('gloo')
rank = dist.get_rank()

def scheduler(step):
    f0 = step % 4
    f1 = step % 4 + 1
    f2 = step % 4 + 2
    f3 = step % 4 + 3
    return f0, f1, f2, f3


class MicroBatchContent:
    def __init__(self):
        self.x = None
        self.y = None
        

class PipelineStage:
    def __init__(
            self, 
            module:nn.Module, 
            input_size,
            output_size,
            prev_node, 
            next_node
            ):
        self.module = module
        self.prev_node = prev_node
        self.next_node = next_node
        self.input_size = input_size
        self.output_size = output_size
        self.ctxs = {}


    def forward_recv(self, mb_id)->torch.Tensor:
        X = torch.empty(self.input_size)
        dist.recv(
            tensor=X,
            src=self.prev_node,
            tag=0
        )
        return X
    
    def forward_send(self, mb_id, Y)->None:
        dist.send(
            tensor= Y,
            dst=self.next_node,
            tag=0
        )
    

        
    def forward_step(self, mb_id, X):
        
        
        

        
        




        

# class TestFunc1(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, X):
#         return X
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_output = grad_output.clone()
#         dist.all_reduce(grad_output)
#         return grad_output
    




# class TestNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         rand_seed.fix_seed(seed+rank)
#         self.fc1 = nn.Linear(10, 5) # should be initialized differently later on.
#         self.fc2 = nn.Linear(5, 5, bias=False) 
#         rand_seed.fix_seed(seed)
#         self.bias = nn.Parameter(torch.randn(size=(5,)))


#     def forward(self, X):
#         X = TestFunc1.apply(X)
#         h1 = self.fc1(X)
#         h2 = self.fc2(h1)
#         out = TestFunc2.apply(h2)
#         return out + self.bias
