import random

dir_name = "pos"

TAGS = set()
WORDS = set()

START = 0  # start of sequence token
END = 1  # end of sequence token
UNKNOWN_WORD = "*UNKNOWN*"


# TRAIN
def read_data(fname, unknown_words=False):
    """
    reads the file, and creates lists from each sentence holding tuples (word, tag)
    :param fname: file to read from
    :return: a list of lists (sentences) each holds tuples as discussed above
    """
    sentences_and_tags = []
    current = []
    for line in file(fname):
        if line != "\n":
            word, tag = line[:-1].split()
            if unknown_words is True and word not in WORDS:
                word = UNKNOWN_WORD
            current.append((word, tag))
            WORDS.add(word)
            TAGS.add(tag)
        else:
            sentences_and_tags.append(current)
            current = []
    return sentences_and_tags


train_raw = read_data(dir_name + "/train")
dev_raw = read_data(dir_name + "/dev", unknown_words=True)
T2I = {tag: i for i, tag in enumerate(list(sorted(TAGS)))}
I2T = {i: tag for tag, i in T2I.iteritems()}
W2I = {word: i+2 for i, word in enumerate(list(sorted(WORDS)))}
W2I[START] = 0
W2I[END] = 1
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
        sentence = [(START, START), (START, START)] + sentence + [(END, END), (END, END)]  # adds starts/end tags for start/last win

        for j, (word, tag) in enumerate(sentence[2:-2]):  # from first word, up to n
            i = j + 2
            windows.append(([W2I[sentence[i-2][0]], W2I[sentence[i-1][0]], W2I[word],
                                        W2I[sentence[i+1][0]], W2I[sentence[i+2][0]]], T2I[tag]))
    return windows


TRAIN = make_5_windows(train_raw)
DEV = make_5_windows(dev_raw)
# TEST = make_5_windows(test_raw)

# #########################
# print DEV[:10]
# DEV = [([random.randint(0, 10) for _ in range(5)], i % 10) for i in xrange(0, 10000)]
# print DEV[:10]
# #########################

if __name__ == "__main__":
    print "{"
    for k, v in W2I.iteritems():
        print " ", k, ":", v
    print "}"
    print "{"
    for k, v in T2I.iteritems():
        print " ", k, ":", v
    print "}"


