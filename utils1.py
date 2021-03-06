

STUDENT = {'name': 'Tomer Gill',
           'ID': '318459450'}

dir_name = "pos"

TAGS = set()
WORDS = set()

START = "*START*"  # start of sequence token
END = "*END*"  # end of sequence token
UNKNOWN_WORD = "*UNKNOWN*"

WORDS.add(UNKNOWN_WORD)


def read_data(fname, unknown_words=False):
    """
    reads the file, and creates lists from each sentence holding tuples (word, tag)
    :param fname: file to read from
    :param unknown_words If True and found a new word (that wasn't seen before) replaces it with UNKNOWN_WORD
    :return: a list of lists (sentences) each holds tuples as discussed above
    """
    sentences_and_tags = []
    current = []
    for line in file(fname):
        if line != "\n":
            word, tag = line[:-1].split()
            if unknown_words is True and word not in WORDS:
                word = UNKNOWN_WORD
            elif unknown_words is False:
                WORDS.add(word)
            current.append((word, tag))
            TAGS.add(tag)
        else:
            sentences_and_tags.append(current)
            current = []
    return sentences_and_tags


train_raw = read_data(dir_name + "/train")
dev_raw = read_data(dir_name + "/dev", unknown_words=True)
T2I = {tag: i for i, tag in enumerate(list(sorted(TAGS)))}
I2T = {i: tag for tag, i in T2I.iteritems()}
W2I = {word: i + 2 for i, word in enumerate(list(sorted(WORDS)))}
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
        sentence = [(START, START), (START, START)] + sentence + [(END, END),
                                                                  (END, END)]  # adds starts/end tags for start/last win

        for j, (word, tag) in enumerate(sentence[2:-2]):  # from first word, up to n
            i = j + 2
            windows.append(([W2I[sentence[i - 2][0]], W2I[sentence[i - 1][0]], W2I[word],
                             W2I[sentence[i + 1][0]], W2I[sentence[i + 2][0]]], T2I[tag]))
    return windows


TRAIN = make_5_windows(train_raw)
DEV = make_5_windows(dev_raw)


# FOR TESTS

def read_test(fname):
    """
    Reads a test file for prediction, which is just a list of words with \n seperating sentences from each other.
    :param fname: Name of file
    :return: A list of sentences (lists of words)
    """
    sentences = []
    current = []
    for line in file(fname):
        if line != "\n":
            word = line[:-1]  # without \n
            if word not in WORDS:
                word = UNKNOWN_WORD
            current.append(word)
        else:
            sentences.append(current)
            current = []
    return sentences


def make_5_windows_without_tags(sentences):
    """
    Creates a 5 word windows from each sentence.
    For word i, it's window is [word i-2, word i-1, word i, word i+1, word i+ 2],
    where each word comes from the same sentences.
    For index < 0 there is a special START word, and likewise and END word for index >= len(sentence)
    :param sentences: A list of sentences (lists of words)
    :return: A list of all the windows from all the sentences
    """
    windows = []
    for sentence in sentences:
        sentence = [START, START] + sentence + [END, END]
        for j, word in enumerate(sentence[2:-2]):
            i = j + 2
            windows.append([
                W2I[sentence[i - 2]], W2I[sentence[i - 1]], W2I[word], W2I[sentence[i + 1]], W2I[sentence[i + 2]]
            ])
    return windows


test_raw = read_test(dir_name + "/test")
TEST = make_5_windows_without_tags(test_raw)

if __name__ == "__main__":
    print "test_raw:\n["
    for sentence in test_raw:
        print " " + str(sentence)
