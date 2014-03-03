import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk import ConcordanceIndex
from numpy import array, zeros, vstack, linspace, diag, dot
from numpy import sum as npsum
from numpy.linalg import svd
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cosine

# TODO: lemmatization?
# TODO: normalization of word-vectors in context-vector creation?

# constants
feat_num = 7  # number of words to build word-vectors for
dim_num = 2  # number of dimensions of word space
svd_dim_num = 2  # number of dimensions in svd-space
window_radius = 3  # how many words to include on each side of occurance
cluster_num = 2
ambiguous_words = ['mad']



train_corpus = ['"', 'But', 'I', "don't", 'want', 'to', 'go', 'among', 'mad', 'people', ',', '"', 'Alice', 'remarked', '.', 'Oh', ',',
'you', "can't", 'help', 'that', ',', 'said', 'the', 'Cat', ':', "'we're", 'all', 'mad', 'here', '.', "I'm", 'mad', '.', "You're", 'mad', '.',
'How', 'do', 'you', 'know', "I'm", 'mad', '?', 'said', 'Alice', '.', 'You', 'must', 'be', ',', 'said', 'the', 'Cat', ',', 'or', 'you',
"wouldn't", 'have', 'come', 'here', '.']

test_corpus = train_corpus


def draw_word_senses(sense_vectors, context_vectors, labels):
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

    plt.scatter(sense_vectors[:, 0], sense_vectors[:, 1], marker='o', c=col, s=100)

    for i in range(len(sense_vectors)):
        cluster_i = [vector for vector, label in\
                     zip(context_vectors, labels) if label == i]
        for x,y in cluster_i:
            plt.plot(x,y, marker='x', color=col[i])

    plt.show()


def cleanse_corpus(corpus):
    """ Returns lowercase copy of list with removed stop words and punctuation
    marks.
    """

    return [word.lower() for word in corpus if
            word.lower() not in stopwords.words('english') and
            word not in string.punctuation]


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

    if matrix.shape[0] < matrix.shape[1]:
        raise ValueError("Matrix has more columns (dimensions) than rows (feature vectors).")
    if len(matrix[0]) < dim:
        raise ValueError('Reduction to more dimensions than contained in matrix not possible.')

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
        cos_similarities.append(cosine(context_vector, sense))

    return cos_similarities.index(min(cos_similarities))


def train_sec_order(corpus, ambigous_words):
    # containers
    word_vectors = {}  # maps feature-words to lists containing their vectors
    sense_vectors = {}  # maps ambiguous words to ndarray of sense vectors

    # remove stop words and signs
    filtered = cleanse_corpus(corpus)

    # find dimensions and features
    words_desc = FreqDist(filtered).keys()
    dimensions = words_desc[:dim_num]
    features = words_desc[:feat_num]

    # create word vectors for features
    offset_index = ConcordanceIndex(filtered, key=lambda s: s.lower())
    for word in features:
        context = []
        # get word positions in text
        offsets = offset_index.offsets(word)

        # collect ALL contexts for word vector
        for offset in offsets:
            context += sized_context(offset, window_radius, filtered)

        word_vector = word_vector_from_context(context, dimensions)
        word_vectors[word] = word_vector

    for word in ambigous_words:
        # create context vectors for ambigous words
        vectors = []
        offsets = offset_index.offsets(word)
        for offset in offsets:
            context = sized_context(offset, window_radius, filtered)
            vectors.append(context_vector_from_context(context, word_vectors))

        # perform svd and dimension reduction
        context_matrix = vstack(vectors)
        svd_matrix = svd_reduced_eigenvectors(context_matrix, svd_dim_num)

        # create sense vectors for ambigous context vectors
        svd_centroids, labels = kmeans2(svd_matrix, cluster_num)

        centroids = []
        for i in range(cluster_num):
            cluster_i = [vector for vector, label in\
                         zip(vectors, labels) if label == i]
            centroids.append(npsum(vstack(cluster_i), 0))

        sense_vectors[word] = centroids

        #draw_word_senses(svd_centroids, svd_matrix, labels)
        #draw_word_senses(vstack(centroids), context_matrix, labels)

    return sense_vectors, word_vectors


def test_sec_order(corpus, ambiguous_words, sense_vectors, word_vectors):
    # remove stop words and signs
    filtered = cleanse_corpus(corpus)

    # get word positions in text
    offset_index = ConcordanceIndex(filtered, key=lambda s: s.lower())

    word_context_labels = {}
    for ambiguous_word in ambiguous_words:
        offsets = offset_index.offsets(ambiguous_word)

        # find all contexts for the word and assign them to a sense
        context_labels = []
        for offset in offsets:
            context = sized_context(offset, window_radius, filtered)
            context_vector = context_vector_from_context(context, word_vectors)
            label = assign_sense(context_vector, sense_vectors[ambiguous_word])
            context_labels.append(label)

        word_context_labels[ambiguous_word] = context_labels

    return word_context_labels

senses, words = train_sec_order(train_corpus, ambiguous_words)
labels = test_sec_order(test_corpus, ambiguous_words, senses, words)
