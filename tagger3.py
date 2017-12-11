import torch as tr
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import torch.optim as opt
import torch.utils.data as utdata
import numpy as np
import time

STUDENT = {'name': 'Tomer Gill',
           'ID': '318459450'}

# Decide whether to use the pre-trained vectors
ans = str(raw_input("Use pre-trained embedding vectors?[Y/n]"))
if ans == "Y" or ans == "y" or ans == "yes" or ans == "Yes" or ans == "YES":
    ans = True
    import utils2 as ut
else:
    ans = False
    import utils1 as ut

prefix_size = 3
suffix_size = 3
prefixes = [ut.I2W[i][ : prefix_size ] for i in xrange(len(ut.I2W))]
suffixes = [ut.I2W[i][-suffix_size : ] for i in xrange(len(ut.I2W))]
P2I = {pre: i for i, pre in enumerate(sorted(set(prefixes)))}
S2I = {suf: i for i, suf in enumerate(sorted(set(suffixes)))}


def get_prefix_index_by_word_index(index):
    return P2I[prefixes[index]]


def get_suffix_index_by_word_index(index):
    return S2I[suffixes[index]]


class Net(nn.Module):
    """
    Class for  the neural net model, an MLP model with one hidden layer and an embedding matrix.
    """

    def __init__(self, vocab_size, context_size, prefix_num, suffix_num, hid_dim, out_dim,
                 init_embed_matrix=None, embedding_size=50):
        """
        Initialize the parameters of the neural net.
        :param context_size: How many words are in a single input
        :param hid_dim: Size of the first layer's output vector
        :param out_dim: Size of output vector
        :param init_embed_matrix: A numpy matrix of pre-trained embedding vectors
        """
        super(Net, self).__init__()

        # Embedding matrix
        if init_embed_matrix is not None:  # load from pre-trained embedding matrix
            self.E = nn.Embedding(init_embed_matrix.shape[0], init_embed_matrix.shape[1])
            self.E.weight.data.copy_(tr.from_numpy(init_embed_matrix))
            embedding_size = init_embed_matrix.shape[1]
        else:  # create a new one
            self.E = nn.Embedding(vocab_size, embedding_size)
        self.after_embed_size = embedding_size * context_size

        # prefix / suffix embedding
        self.prefix_E = nn.Embedding(prefix_num, embedding_size)
        self.suffix_E = nn.Embedding(suffix_num, embedding_size)

        self.lin = nn.Linear(self.after_embed_size, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        """
        Runs the input through the neural network and returns the output.
        :param x: A torch.Autograd.Variable of a "context_size" vector (context size is defined in the constructor)
        :return: An "out_dim" vector of outputs, each value is a score of how likely the tag should be this index
        """
        # out = self.E(x).view(-1, self.after_embed_size)  # concating the embedding vectors

        # get prefix and suffix indexes for each word's pre/suffix
        prefix_vectors, suffix_vectors = x.data.numpy().copy(), x.data.numpy().copy()
        prefix_vectors = prefix_vectors.reshape(-1)  # make one long vector
        suffix_vectors = suffix_vectors.reshape(-1)

        for i, pre in enumerate(prefix_vectors):  # replace each word index with it's prefix index
            prefix_vectors[i] = get_prefix_index_by_word_index(pre)
        for i, suf in enumerate(suffix_vectors):  # replace each word index with it's suffix index
            suffix_vectors[i] = get_suffix_index_by_word_index(suf)

        # return to the shape of x and make variables
        suffix_vectors = tr.from_numpy(suffix_vectors.reshape(x.data.shape))
        prefix_vectors = tr.from_numpy(prefix_vectors.reshape(x.data.shape))
        prefix_vectors, suffix_vectors = prefix_vectors.type(tr.LongTensor), suffix_vectors.type(tr.LongTensor)
        prefix_vectors, suffix_vectors = Variable(prefix_vectors), Variable(suffix_vectors)

        out = (self.E(x) + self.prefix_E(prefix_vectors) + self.suffix_E(suffix_vectors)).view(-1, self.after_embed_size)  # sum embedding vectors of word
        # out.view(-1, self.after_embed_size)  # concat each word's sum of vectors

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
    epcohes = 15  # number of iterations
    learning_rate = 0.01
    batch_size = 1000
    hidden_dim = 150
    k = 3

    if ans:
        net = Net(len(ut.W2I), WIN_SIZE, len(P2I), len(S2I), hidden_dim, len(ut.T2I), init_embed_matrix=ut.words_vecs)
    else:
        net = Net(len(ut.W2I), WIN_SIZE, len(P2I), len(S2I), hidden_dim, len(ut.T2I), embedding_size=EMBED_SIZE)

    if load_model:
        net.load_state_dict(tr.load(model_args_path))

    if train and net is not None:
        criterion = nn.CrossEntropyLoss()
        optimizer = opt.Adam(net.parameters(), learning_rate)

        print "######################################################"
        print "Run parameters:"
        print "*\tUsing Pre-trained Embedding Vectors: %s" % "Yes" if ans else "No"
        print "*\tData: " + ut.dir_name.upper()
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
        pred_file = open("test4." + ut.dir_name, "w")
        diff = 0  # difference in rows made by \n chars
        for j, line in enumerate(file(ut.dir_name + "/test")):
            i = j - diff
            if line != "\n":
                pred_file.write(line[:-1] + " " + ut.I2T[predictions[i][0]] + "\n")
            else:
                pred_file.write("\n")
                diff += 1
        pred_file.close()
