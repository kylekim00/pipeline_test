import torch
import torch.nn as nn
import torch.distributed as dist

dist.init_process_group('gloo')
rank = dist.get_rank()

class TestFunc1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        return X
    
    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output)
        return grad_output
    

class TestFunc2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        dist.all_gather()
        return X
    
    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output)
        return grad_output

class TestNN(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(10, 10)
        if rank== 0:
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(10, 5, bias=False)            
        elif rank == 1:
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(10, 5, bias= False)



    def forward(self, X):
        X = TestFunc1.apply(X)
        h1 = self.fc1(X)
        h2 = self.fc2(h1)
        







        return 






if rank == 0:
    print("hello!")
if rank == 1:
    print("dooshbag")