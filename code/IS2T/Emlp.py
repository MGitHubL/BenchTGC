import torch
from torch import nn
from torch.nn import functional as F


class EMLP(nn.Module):
    def __init__(self, args):
        super(EMLP, self).__init__()
        self.vars = nn.ParameterList()
        w1 = nn.Parameter(torch.ones(*[1, args.out_dim]))  # [1,d]
        torch.nn.init.kaiming_normal_(w1)
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(1)))

    def forward(self, x, vars=None):
        if vars == None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1])

        return x

    def parameters(self):
        return self.vars
