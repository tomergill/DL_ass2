import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as opt
import torch.utils.data as utdata
import utils1 as ut
import numpy as np
import time


# The Nueral Net
class Net(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_size, hid_dim, out_dim, batch_size):
        super(Net, self).__init__()
        self.E = nn.Embedding(vocab_size, embedding_size)
        self.after_embed_size = embedding_size * context_size
        self.lin = nn.Linear(self.after_embed_size, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.bsize = batch_size

    def forward(self, x):
        out = self.E(x).view(-1, self.after_embed_size)
        out = self.lin(out)
        out = F.tanh(out)
        out = self.lin2(out)
        return out


def make_data_loader(examples, batch_size=100, shuffle=True):
    x, y = zip(*examples)  # makes lists of windows and tags
    x, y = tr.from_numpy(np.array(x)), tr.from_numpy(np.array(y))
    x, y = x.type(tr.LongTensor), y.type(tr.LongTensor)
    train = utdata.TensorDataset(x, y)
    return utdata.DataLoader(train, batch_size, shuffle)


def train_net(net, data_loader, iter_num, optimizer, criterion, acc_loader):
    print "+----+--------+----------+----------+---------+"
    print "| it |  loss  | time (s) | dev_loss | dev_acc |"
    print "+----+--------+----------+----------+---------+"
    for i in range(iter_num):
        cum_loss = 0.0
        start_time = time.time()
        for _, (inputs, labels) in enumerate(data_loader, 0):
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            cum_loss += loss.data[0]
        acc, loss = accuracy_and_loss_on(net, acc_loader, criterion)  # compute accuracy in each iteration
        print "| %-2d | %1.4f | %8.5f | %f | %6.3f%% |" % (
            i, cum_loss / len(data_loader), time.time() - start_time, loss, acc * 100)
    print "+----+--------+----------+----------+---------+"


def accuracy_and_loss_on(net, data_loader, criterion):
    good = bad = cum_loss = 0.0
    for data in data_loader:
        feats, labels = data
        ################################
        # print feats
        # print labels
        # for j in range(0, 1000, 100):
        #     temp = []
        #     for i in range(5):
        #         temp.append(ut.I2W[feats[j, i]])
        #     print temp
        #     print ut.I2T[labels[j]]
        # input()
        ################################
        outputs = net(Variable(feats))
        _, predicted = tr.max(outputs.data, 1)
        comp = (predicted == labels)
        bad += (predicted != labels).sum()
        if ut.dir_name == "ner":
            comp = [0 if labels[i] == ut.T2I["O"] else score for i, score in enumerate(comp)]
        good += sum(comp)
        loss = criterion(outputs, Variable(labels))
        cum_loss += loss.data[0]
    return good / (good + bad), (cum_loss / len(data_loader))


def predict_by_windows(net, windows):
    prediction, inputs = []
    for input in windows:
        prediction += list(tr.max(net(Variable(input)).data, 1))
        inputs += list(input[:, 2])
    return prediction, inputs


if __name__ == "__main__":
    train = True
    save_model = True
    test = True
    model_args_path = "~/Desktop/Deep Learning/ass2"
    load_model = True

    EMBED_SIZE = 50
    WIN_SIZE = 5
    epcohes = 15
    learning_rate = 0.001
    batch_size = 1000
    hidden_dim = 100

    net = Net(len(ut.W2I), EMBED_SIZE, WIN_SIZE, hidden_dim, len(ut.T2I), batch_size)

    if load_model:
        net.load_state_dict(tr.load(model_args_path))

    if train:
        criterion = nn.CrossEntropyLoss()
        optimizer = opt.Adam(net.parameters(), learning_rate)

        print "######################################################"
        print "Run parameters:"
        print "*\tEmbedding layer size: %d" % EMBED_SIZE
        print "*\tWindow size: %d" % WIN_SIZE
        print "*\tNumber of iterations: %d" % epcohes
        print "*\tLearning rate: %f" % learning_rate
        print "*\tBatch size: %d" % batch_size
        print "*\tHidden dimension size: %d" % hidden_dim
        print "######################################################"
        print "\nLearning on test, with accuracy on dev each iteration:"
        train_net(net, make_data_loader(ut.TRAIN, batch_size), epcohes, optimizer, criterion,
                  make_data_loader(ut.DEV, batch_size))
        if save_model:  # should save the net
            tr.save(net.state_dict(), model_args_path)

    if test:
        predictions, inputs = predict_by_windows(net, ut.TEST)
        pred_file = open("~/Desktop/Deep Learning/ass2/test1." + ut.dir_name, "w")
        for i, pred in enumerate(predictions):
            pred_file.write(ut.I2W[inputs[i]] + " " + ut.I2T[pred] + "\n")
        pred_file.close()

