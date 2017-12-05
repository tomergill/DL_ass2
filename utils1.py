dir_name = "pos"

TAGS = set()
WORDS = set()

START = 0  # start of sequence token
END = -1  # end of sequence token


# TRAIN
def read_data(fname):
    """
    reads the file, and creates lists from each sentence holding tuples (word, tag)
    :param fname: file to read from
    :return: a list of lists (sentences) each holds tuples as discussed above
    """
    sentences_and_tags = []
    current = []
    for line in file(fname):
        if line != "\n":
            word, tag = line[:-1].split(' ')
            current.append((word, tag))
            WORDS.add(word)
            TAGS.add(tag)
        else:
            sentences_and_tags.append(current)
            current = []
    return sentences_and_tags


train_raw = read_data(dir_name + "/train")
dev_raw = read_data(dir_name + "/dev")
T2I = {tag: i for i, tag in enumerate(list(sorted(TAGS)))}
I2T = {i: tag for tag, i in T2I.iteritems()}
W2I = {word: i+1 for i, word in enumerate(list(sorted(WORDS)))}
W2I[START] = 0
W2I[END] = -1
I2W = {i: word for word, i in W2I.iteritems()}


def make_5_windows(sentences_and_tags):
    """
    Takes a list of sentences and each word's tag and create from them 5 word windows
    :param sentences_and_tags: list of lists (sentences), each is a list of (word, tag) tuples
    :return: a list of tuples in the form (list of the 5-word window's indexes by W2I when the center is the
    current word, index of tag by T2I)
    """
    windows = []
    for sentence in sentences_and_tags:
        sentences_and_tags = [START, START] + sentences_and_tags + [END, END]  # adds starts/end tags for start/last win

        for i, (word, tag) in enumerate(sentence[2:-2]):  # from first word, up to n
            windows.append(([W2I[sentence[i-2][0]], W2I[sentence[i-1][0]], W2I[word],
                                        W2I[sentence[i+1][0]], W2I[sentence[i+2][0]]], T2I[tag]))
    return windows


TRAIN = make_5_windows(train_raw)
DEV = make_5_windows(dev_raw)
# TEST = make_5_windows(test_raw)

