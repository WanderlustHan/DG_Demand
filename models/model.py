from typing import Tuple
from .GMSDR import GMSDRNet, DynamicGraphNet
import torch
from torch import nn, Tensor


def create_model(loss, conf, device,  support=None):
    x = torch.tensor(support,device=device)
    support = x.float()
    model = GMSDRNet(**conf['GMSDR'],support=support, device=device)
    return model, MetricNNTrainer(model, loss)




class Trainer:
    def __init__(self, model: nn.Module, loss):
        self.model = model
        self.loss = loss

    def train(self, inputs: Tensor, targets: Tensor, supports,  phase: str) -> Tuple[Tensor, Tensor]:
        raise ValueError('Not implemented.')


class MetricNNTrainer(Trainer):
    def __init__(self, model, loss):
        super(MetricNNTrainer, self).__init__(model, loss)

    def train(self, inputs: Tensor, targets: Tensor, supports, phase: str):
        outputs, graph = self.model(inputs,supports)
        loss = self.loss(outputs, targets, graph)
        return outputs, loss

