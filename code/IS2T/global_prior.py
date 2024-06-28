import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class Global_emb(nn.Module):
    def __init__(self, args):
        super(Global_emb, self).__init__()
        self.vars = nn.ParameterList()
        w = nn.Parameter(torch.ones(*[args.out_dim, args.out_dim]))  # [d,d]
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.out_dim)))
        self.g_emb = Variable(torch.ones(*[1, args.out_dim]).cuda(), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.g_emb)

    def forward(self, s_node_emb, t_node_emb, global_s, global_t):
        vars = self.vars
        g_emb = self.g_emb
        g_emb_update = torch.tanh((
                g_emb +
                torch.sum(F.linear(s_node_emb, vars[0], vars[1]), dim=0, keepdim=True) +
                torch.sum(F.linear(t_node_emb, vars[0], vars[1]), dim=0, keepdim=True)))
        self.g_emb = g_emb_update.data  # [1,d]

        return g_emb

    def parameters(self):
        return self.vars


class Global_w(nn.Module):
    def __init__(self, args):
        super(Global_w, self).__init__()
        self.vars = nn.ParameterList()
        self.args = args
        delta_global = nn.Parameter(torch.ones(*[1]))
        self.vars.append(delta_global)
        delta_node = nn.Parameter(torch.ones(*[1]))
        self.vars.append(delta_node)

    def forward(self, edge):
        vars = self.vars
        global_update = torch.tanh(vars[0] * edge)
        node_update = torch.tanh(vars[1] * (1 / edge))

        return global_update, node_update

    def parameters(self):
        return self.vars
