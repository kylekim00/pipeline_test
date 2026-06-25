import torch
import torch.distributed as dist
import torch.nn as nn

dist.init_process_group('gloo')
rank = dist.get_rank()

class RecvFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, rank):
        dist.recv()
        ctx.save_for_backward(rank)
        return X
    
    @staticmethod
    def backward(ctx, grad_output):
        (rank, ) = ctx.saved_tensors
        dist.send(grad_output)
        return grad_output


class SendFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, rank):
        dist.send(X, dst=rank)
        ctx.save_for_backward(rank)
        return X
    
    @staticmethod
    def backward(ctx, grad_output):
        (rank, ) = ctx.saved_tensors
        dist.send(grad_output)
        return grad_output




class DistLinear(nn.Module):
    def __init__(
            self, 
            in_features, 
            out_features, 
            batch_features=1, 
            rank=0
            ):
        super().__init__()
        self.model = nn.Linear(in_features, out_features)
        self.recv_buf = torch.empty([batch_features, in_features])


    


    def forward(self, X):
        pass



        

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
