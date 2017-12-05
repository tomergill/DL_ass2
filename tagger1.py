import torch as tr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as opt
import torch.utils.data as utdata
import utils1 as ut


# The Nueral Net
class Net(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_size, hid_dim, out_dim):
        super(Net, self).__init__()
        self.E = nn.Embedding(vocab_size, embedding_size)
        self.lin = nn.Linear(embedding_size * context_size, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.lin2(F.tanh(self.lin(self.E(x).view((1, -1)))))


def make_data_loader(examples, batch_size= 100, shuffle=True):
    x, y = zip(*examples)  # makes lists of windows and tags
    x, y = tr.FloatTensor(x), tr.LongTensor(y)
    train = utdata.TensorDataset(x, y)
    return utdata.DataLoader(train, batch_size, shuffle)


def train_net(net, data_loader, iter_num, optimizer, criterion):
    print "it loss"
    for i in range(iter_num):
        cum_loss = 0.0
        for _, (inputs, labels) in enumerate(data_loader, 0):
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            cum_loss += loss.data[0]
        print i, cum_loss / len(data_loader)


if __name__ == "__main__":
    EMBED_SIZE = 50
    WIN_SIZE = 5
    epcohes = 15
    learning_rate = 0.01
    batch_size = 64

    net = Net(len(ut.W2I), EMBED_SIZE, WIN_SIZE, 100, len(ut.T2I))
    criterion = nn.CrossEntropyLoss()
    optimizer = opt.SGD(net.parameters(), learning_rate)

    print "Learning on test:"
    train_net(net, make_data_loader(ut.TRAIN, batch_size), epcohes, optimizer, criterion)

