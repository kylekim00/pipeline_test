import torch
import torch.distributed as dist
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum, auto

from buffer import *

dist.init_process_group('gloo')
rank = dist.get_rank()



class Action(Enum):
    F = auto()
    B = auto()

@dataclass
class Ctx:
    X:torch.Tensor
    Y:torch.Tensor

@dataclass
class MicroBatch:
    action:Action
    mb_id:int

    def __repr__(self):
        return f"<{self.action.name},{self.mb_id}>\n"
    def __iter__(self):
        yield self.action
        yield self.mb_id
        





class MicroBatchSchedular:
    def __init__(
        self, 
        world_size:int, 
        world_rank:int,
        num_microbatches:int = 8,
        max_activation_size:int = 4,

        ):
        self.world_size = world_size
        self.world_rank = world_rank
        self.num_microbatches = num_microbatches
        self.max_activation_size = max_activation_size
        if num_microbatches < max_activation_size:
            raise AttributeError(f"num_microbatches({num_microbatches}) can not be lesser than max_activation_size({max_activation_size})")
        
    def build(self):
        schedulers = list()
        forward_lists = list()
        backward_lists = list()



        for i in range(self.world_size):
            schedulers.append(list())
            forward_lists.append(list())
            backward_lists.append(list())

        forward_lists[0] = list(range(self.num_microbatches))

        backward_round = [False] * self.world_size
        activation_count = [0] * self.world_size
        while True:
            no_action = 0
            for i in range(self.world_size):
                if backward_round[i]:
                    if len(backward_lists[i]) != 0:
                        back_id = backward_lists[i].pop(0)
                        schedulers[i].append(MicroBatch(Action.B,back_id))
                        activation_count[i] -= 1
                        if i != 0:
                            backward_lists[i-1].append(back_id)
                        if len(forward_lists[i]) != 0:
                            backward_round[i] = False
                    elif len(forward_lists[i]) != 0: #not likely to come in.
                        backward_round[i] = False
                    else:
                        no_action += 1

                else:
                    if len(forward_lists[i]) != 0 and activation_count[i] < self.max_activation_size:
                        for_id = forward_lists[i].pop(0)
                        schedulers[i].append(MicroBatch(Action.F, for_id))
                        activation_count[i] += 1
                        if i == self.world_size - 1:
                            backward_lists[i].append(for_id)
                        else:
                            forward_lists[i+1].append(for_id)
                        if len(backward_lists[i]) != 0:
                            backward_round[i] = True
                    elif len(backward_lists[i]) != 0:
                        back_id = backward_lists[i].pop(0)
                        schedulers[i].append(MicroBatch(Action.B, back_id))
                        activation_count[i] -= 1
                        if i != 0:
                            backward_lists[i-1].append(back_id)
                        
                    else:
                        no_action += 1
            # print(activation_count)
            if no_action == self.world_size:
                break
        return schedulers

        

class PipelineStage:
    def __init__(
            self, 
            module:nn.Module, 
            input_size,
            output_size,
            prev_node, 
            next_node,
            # data_loader:torch.utils.data.dataloader.DataLoader = None
            ):
        self.module = module
        self.prev_node = prev_node
        self.next_node = next_node
        self.input_size = input_size
        self.output_size = output_size
        self.ctxs = {}
        # if prev_node == None:
        #     self.dataloaderiter = iter(data_loader)


    def forward_recv(self)->torch.Tensor:
        if self.dataloader:
            pass
        X = torch.empty(self.input_size)
        dist.recv(
            tensor=X,
            src=self.prev_node,
            tag=0
        )

        return X
    


    def forward_step(self, mb_id, X):
        out = self.module(X)
        self.ctxs[mb_id] = Ctx(X, out)
        # return out
    

    def forward_send(self,mb_id)->None:
        dist.send(
            tensor= self.ctxs[mb_id].Y,
            dst=self.next_node,
            tag=0
        )

        

    def backward_recv(self)->torch.Tensor:
        output_grad = torch.empty(self.output_size)
        dist.recv(
            tensor=output_grad,
            src=self.next_node,
            tag=1
        )
        return output_grad
    
    def backward_step(self, mb_id, output_grad)->None:
        self.ctxs[mb_id].Y.backward(output_grad)
    
    def backward_send(self, mb_id)->None:
        ctx = self.ctxs.pop(mb_id)
        dist.send(
            tensor= ctx.X,
            dst = self.prev_node,
            tag=1
        )
        

class Pipeline_Node:
    def __init__(
            self, 
            node_num:int,
            world_size:int,
            world_rank:int,

            module:nn.Module,
            input_size,
            output_size,
            data_loader:torch.utils.data.dataloader.DataLoader = None,
            ):
        self.node_num = node_num
        self.world_size = world_size
        self.world_rank = world_rank
        self.module = module
        self.input_size = input_size
        self.output_size = output_size
        
        self.scheduler = MicroBatchSchedular(
            world_size=world_size,
            world_rank=world_rank,
            num_microbatches=8,
            max_activation_size=4
        )

        if node_num == 0:
            self.data_loader = data_loader

        self.operation = PipelineStage(
            module=module,
            input_size= input_size,
            output_size=output_size,
            prev_node= node_num - 1 if node_num != 0 else None,
            next_node= node_num + 1 if node_num != world_rank - 1 else None
        )


    def one_batch(self, X=None, label=None):
        schedule = self.scheduler.build()[self.world_rank]#build schedules
        

        for a, mb_id in schedule:
            if a == Action.F:
                #TODO make an operation that splits operations corresponding to mb_id for node 0

                X = self.operation.forward_recv()
                self.operation.forward_step(mb_id, X)
                self.operation.forward_send(mb_id)

            elif a == Action.B:
                output_grad = self.operation.backward_recv()
                self.operation.backward_step(mb_id, output_grad)
                self.operation.backward_send(mb_id)
        
        pass

    def run(self, epoch):
        for i in range(epoch): #epoch

        #TODO get the batch from the dataloader.
        #TODO send the labels to the end node.
        pass
        
        
#first we will start with the run code. 
#the run code will then start the loop for the whole epoch.
# each step of the loop computes the whole batch that dataloader gives us. 
# So the schedule will split the batches into microbatches.
# After splitting the batches into microbatches, we will send each 
# 
        
        
