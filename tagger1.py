import torch as tr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as opt
import torch.utils.data as utdata
import numpy as np

VSIZE = 3
ESIZE = 5


# The Nueral Net
class Net(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim):
        super(Net, self).__init__()
        self.E = nn.Embedding(VSIZE, ESIZE)
        self.lin = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.lin2(F.tanh(self.lin(self.E(x).view((1, -1)))))


if __name__ == "__main__":
    pass