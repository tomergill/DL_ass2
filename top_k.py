import numpy as np

words_vecs = np.loadtxt("wordVectors.txt")
words = [line[:-1] for line in file("vocab.txt")]


def vec_dist(u, v):
    """
    Computes the cosine distance between the vectors
    :param u: Numpy vector
    :param v: Numpy vector
    :return: (u dot v) / (sqrt(u dot u) * sqrt(v dot v))
    """
    dot_prodct = np.dot(u, v)
    u2_sqrt = np.sqrt(np.dot(u, u))
    v2_sqrt = np.sqrt(np.dot(v, v))
    return dot_prodct / (u2_sqrt * v2_sqrt)


def most_similar(word, k):
    """
    Finds the most similar words using cosine distance.
    :param word: The word to compare to (find most similar to it)
    :param k: How many similar words to find
    :return: The k most similar words to word by cosine distance, or None if word isn't found in words
    """
    try:
        index = words.index(word)
        vec = words_vecs[index]
        distances = np.array([vec_dist(vec, other_vec) if np.any(other_vec != vec) else 0 for other_vec in words_vecs])
        max_ind = np.argpartition(distances, -k)[-k:]
        similar = [(words[index], distances[index]) for index in max_ind]
        return similar
    except ValueError:  # word isn't in words
        return None


if __name__ == "__main__":
    words_to_check = ["dog", "england", "john", "explode", "office"]
    k = 5
    for word in words_to_check:
        similar = most_similar(word, k)
        if similar is not None:
            print "The %d most similar words to %s are:" % (k, word)
            for w, dist in similar:
                print "*\t%s (%f)" % (w, dist)
        else:
            print "Couldn't find a the word %s in words file." % word
        print "####################################\n"
