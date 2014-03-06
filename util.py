from collections import Counter
from numpy import array, zeros, linspace, diag, dot
from numpy.linalg import norm
from scipy.linalg import svd
from scipy.spatial.distance import cosine


def draw_word_senses(sense_vectors, context_vectors, labels, cluster_num=2):
    """ Utility function that draws sense-vectors as o in different
    colours and context vectors as black x.
    """

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if len(sense_vectors[0]) > 2:
        sense_vectors = svd_reduced_original(sense_vectors, 2)
    if len(context_vectors[0]) > 2:
        context_vectors = svd_reduced_original(context_vectors, 2)

    col = cm.rainbow(linspace(0, 1, cluster_num))
    plt.figure()

    plt.scatter(sense_vectors[:, 0], sense_vectors[:, 1], marker='o',
                c=col, s=100)

    for i in range(len(sense_vectors)):
        cluster_i = [vector for vector, label in
                     zip(context_vectors, labels) if label == i]
        for x, y in cluster_i:
            plt.plot(x, y, marker='x', color=col[i])

    plt.show()


def sized_context(word_index, window_radius, corpus):
    """ Returns a list containing the window_size amount of words to the left
    and to the right of word_index, not including the word at word_index.
    """

    max_length = len(corpus)

    left_border = word_index - window_radius
    left_border = 0 if word_index - window_radius < 0 else left_border

    right_border = word_index + 1 + window_radius
    right_border = max_length if right_border > max_length else right_border

    return corpus[left_border:word_index] + corpus[word_index+1: right_border]


def word_vector_from_context(context, dimensions):
    """ Takes a list of words as context and a list of words as dimensions
    and returns a vector denoting how often each dimension word appeared
    in the context. Order of dimensions in return is presereved.
    """

    word_vector = []
    # slow! Manual: context.count(word) for each word would be faster
    counts = Counter(context)

    for word in dimensions:
        # slow! if/else would be faster
        count = counts.get(word, 0)
        word_vector.append(count)

    return word_vector


def context_vector_from_context(context, word_vectors):
    """ Takes a list of words as context and all the word vectors.
    Returns a context vector that is the centroid of the word vectors
    that occured in the context. Thus gives us second-order co-occurence
    with the dimension words.
    """

    dim_num = len(word_vectors.values()[0])
    centroid = zeros(dim_num, int)

    for word in context:
        if word in word_vectors:
            centroid += array(word_vectors[word])

    return centroid


def svd_reduced_eigenvectors(matrix, dim):
    """ Takes a matrix of size N x M and returns it's
    left eigenvector reduced to dim dimensions.
    """

    U, _, _ = svd(matrix, True)
    return U[:, :dim]


def svd_reduced_original(matrix, dim):
    """ Takes a matrix of size N x M and
    returns the best approximation of dimension
    N x dim (in a least square sense).
    """

    U, s, V = svd(matrix, True)

    # reduce dimensionality
    S = zeros((len(U), dim), float)
    S[:dim, :dim] = diag(s)[:dim, :dim]
    return dot(U, dot(S, V[:dim, :dim]))


def assign_sense(context_vector, sense_vectors):
    " Returns the index of the sense vector most similar to context_vector."

    cos_similarities = []
    for sense in sense_vectors:
        if norm(sense) == 0:
            continue

        cos_similarities.append(cosine(context_vector, sense))

    return cos_similarities.index(min(cos_similarities))
