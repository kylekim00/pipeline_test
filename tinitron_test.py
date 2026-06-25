import torch
import torch.nn as nn
import torch.distributed as dist
import rand_seed

# torch.manual_seed(1234)
seed = 1234


dist.init_process_group('gloo')
rank = dist.get_rank()

class TestFunc1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        return X
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clone()
        dist.all_reduce(grad_output)
        return grad_output
    

class TestFunc2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        X = X.clone()
        dist.all_reduce(X, op=dist.ReduceOp.SUM)
        return X
    
    @staticmethod
    def backward(ctx, grad_output):
        # dist.all_reduce(grad_output)
        return grad_output

class TestNN(nn.Module):
    def __init__(self):
        super().__init__()
        rand_seed.fix_seed(seed+rank)
        self.fc1 = nn.Linear(10, 5) # should be initialized differently later on.
        self.fc2 = nn.Linear(5, 5, bias=False) 
        rand_seed.fix_seed(seed)
        self.bias = nn.Parameter(torch.randn(size=(5,)))


    def forward(self, X):
        X = TestFunc1.apply(X)
        h1 = self.fc1(X)
        h2 = self.fc2(h1)
        out = TestFunc2.apply(h2)
        return out + self.bias




model = TestNN()
X = torch.randn([4, 10])
out = model(X)
print(out)