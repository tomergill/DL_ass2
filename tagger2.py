import torch as tr
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import torch.optim as opt
import torch.utils.data as utdata
import utils2 as ut
import numpy as np
import time


STUDENT = {'name': 'Tomer Gill',
           'ID': '318459450'}


class Net(nn.Module):
    """
    Class for  the neural net model, an MLP model with one hidden layer and an embedding matrix.
    """

    def __init__(self, context_size, hid_dim, out_dim, init_embed_matrix):
        """
        Initialize the parametrs of the neural net.
        :param context_size: How many words are in a single input
        :param hid_dim: Size of the first layer's output vector
        :param out_dim: Size of output vector
        :param init_embed_matrix: A numpy matrix of pre-trained embedding vectors
        """
        super(Net, self).__init__()

        # Embedding matrix
        self.E = nn.Embedding(init_embed_matrix.shape[0], init_embed_matrix.shape[1])
        self.E.weight.data.copy_(tr.from_numpy(init_embed_matrix))
        self.after_embed_size = init_embed_matrix.shape[1] * context_size

        self.lin = nn.Linear(self.after_embed_size, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        """
        Runs the input through the neural network and returns the output.
        :param x: A torch.Autograd.Variable of a "context_size" vector (context size is defined in the constructor)
        :return: An "out_dim" vector of outputs, each value is a score of how likely the tag should be this index
        """
        out = self.E(x).view(-1, self.after_embed_size)  # concating the embedding vectors
        out = self.lin(out)  # first linear layer
        out = f.tanh(out)  # non-linear function
        out = self.lin2(out)  # second linear layer
        return out


def make_data_loader(examples, batch_size=100, shuffle=True):
    """
    Makes a torch.utils.data.DataLoader for training and accuracy checks
    :param examples: a list of tuples: (window, tag), where window is list of indexes of words, and tag is index for the
    correct tag of the window's center
    :param batch_size: How many examples should be in one batch
    :param shuffle: Should shuffle the examples?
    :return: The DataLoader
    """
    x, y = zip(*examples)  # makes lists of windows and tags
    x, y = tr.from_numpy(np.array(x)), tr.from_numpy(np.array(y))
    x, y = x.type(tr.LongTensor), y.type(tr.LongTensor)  # convert lists to tensors
    train = utdata.TensorDataset(x, y)
    return utdata.DataLoader(train, batch_size, shuffle)


def train_net(net, data_loader, iter_num, optimizer, criterion, acc_loader):
    """
    Trains the net and checks it's accuracy and loss on the dev data after each iteration of training on all the train
    data. Prints it in a little pretty table.
    :param net: The neural net to train
    :param data_loader: The torch.utils.data.DataLoader to train on
    :param iter_num: The number of desired iteration
    :param optimizer: The optimizer for the learning process
    :param criterion: The loss and grads calculator
    :param acc_loader: torch.utils.data.DataLoader of the dev data (check accuracy and loss on it)
    """
    print "+----+--------+----------+----------+---------+"
    print "| it |  loss  | time (s) | dev_loss | dev_acc |"
    print "+----+--------+----------+----------+---------+"
    for i in range(iter_num):
        cum_loss = 0.0
        start_time = time.time()

        for _, (inputs, labels) in enumerate(data_loader, 0):  # go over all examples
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)  # compute output of net
            loss = criterion(outputs, labels)  # compute loss
            loss.backward()  # compute grads
            optimizer.step()  # update parameters
            cum_loss += loss.data[0]
        acc, loss = accuracy_and_loss_on(net, acc_loader, criterion)  # compute accuracy in each iteration
        print "| %-2d | %1.4f | %8.5f | %f | %5.2f %% |" % (
            i, cum_loss / len(data_loader), time.time() - start_time, loss, acc * 100)
    print "+----+--------+----------+----------+---------+"


def accuracy_and_loss_on(net, data_loader, criterion):
    """
    Predicts the tag of each example, and computes the loss and accuracy on the whole data,
    :param net: Neural net to predict on
    :param data_loader: torch.utils.data.DataLoader to predict on
    :param criterion: The loss calculator
    :return: a tuple: (accuracy, avg. loss) on the whole data
    """
    good = bad = cum_loss = 0.0
    for data in data_loader:
        feats, labels = data
        outputs = net(Variable(feats))
        _, predicted = tr.max(outputs.data, 1)
        comp = (predicted == labels)
        bad += (predicted != labels).sum()
        if ut.dir_name == "ner":  # if on ner ignore correct O tagging
            comp = [0 if labels[i] == ut.T2I["O"] else score for i, score in enumerate(comp)]
        good += sum(comp)
        loss = criterion(outputs, Variable(labels))
        cum_loss += loss.data[0]
    return good / (good + bad), (cum_loss / len(data_loader))


def predict_by_windows(net, windows):
    """
    Predicts on each window the correct tag of the window's center
    :param net: Neural net to predict on
    :param windows: A list of lists, which are windows (list of indexes of words from the same sentence)
    :return: A list of predictions and a list of the index of the words that were predicted, respectively
    """
    prediction, inputs = [], []
    for input in windows:
        _, pred = tr.max(net(Variable(tr.LongTensor(input))).data, 1)
        prediction.append(pred)
        inputs.append(input[2])
    return prediction, inputs


if __name__ == "__main__":
    train = True  # should the network train
    save_model = False  # should save the network parameters to a file?
    test = False  # should predict using the network
    model_args_path = "trained_model_" + ut.dir_name  # path to load/save model
    load_model = False  # should the model be loaded from a file

    EMBED_SIZE = 50
    WIN_SIZE = 5
    epcohes = 5  # number of iterations
    learning_rate = 0.01
    batch_size = 1000
    hidden_dim = 100

    net = Net(WIN_SIZE, hidden_dim, len(ut.T2I), ut.words_vecs)

    if load_model:
        net.load_state_dict(tr.load(model_args_path))

    if train and net is not None:
        criterion = nn.CrossEntropyLoss()
        optimizer = opt.Adam(net.parameters(), learning_rate)

        print "######################################################"
        print "Run parameters:"
        print "*\tEmbedding layer size: %d" % EMBED_SIZE
        print "*\tWindow size: %d" % WIN_SIZE
        print "*\tNumber of iterations: %d" % epcohes
        print "*\tLearning rate: %g" % learning_rate
        print "*\tBatch size: %d" % batch_size
        print "*\tHidden dimension size: %d" % hidden_dim
        print "######################################################"
        print "\nLearning on test, with accuracy on dev each iteration:"
        train_net(net, make_data_loader(ut.TRAIN, batch_size), epcohes, optimizer, criterion,
                  make_data_loader(ut.DEV, batch_size))
        if save_model:  # should save the net
            tr.save(net.state_dict(), model_args_path)

    if test and net is not None:  # writes the predictions to a test2 file
        predictions, inputs = predict_by_windows(net, ut.TEST)
        pred_file = open("test3." + ut.dir_name, "w")
        diff = 0  # difference in rows made by \n chars
        for j, line in enumerate(file(ut.dir_name + "/test")):
            i = j - diff
            if line != "\n":
                pred_file.write(line[:-1] + " " + ut.I2T[predictions[i][0]] + "\n")
            else:
                pred_file.write("\n")
                diff += 1
        pred_file.close()

