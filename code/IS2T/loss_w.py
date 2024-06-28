import torch
from torch import nn
from torch.nn import functional as F


class Loss_weight(nn.Module):
    def __init__(self, args):
        super(Loss_weight, self).__init__()
        self.args = args
        self.vars = nn.ParameterList()

        self.vars.append(nn.Parameter(torch.zeros(1) + 0.001))
        self.vars.append(nn.Parameter(torch.zeros(1) + 0.001))
        self.vars.append(nn.Parameter(torch.zeros(1) + 0.001))
        self.vars.append(nn.Parameter(torch.zeros(1) + 0.001))

    def forward(self, l_c, l_theta, l_h, l_emb):
        vars = self.vars
        loss = torch.sigmoid(vars[0]) * l_c + torch.sigmoid(vars[1]) * l_theta +\
               torch.sigmoid(vars[2]) * l_h + torch.sigmoid(vars[3]) * l_emb
        return loss

    def parameters(self):
        return self.vars
