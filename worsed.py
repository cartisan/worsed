import logging
from collections import Counter
from nltk.probability import FreqDist
from nltk import ConcordanceIndex
from nltk.stem.porter import PorterStemmer
from numpy import array, zeros, vstack, linspace, diag, dot
from numpy import sum as npsum
from numpy.linalg import svd, norm
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cosine

from senseval_adapter import split_corpus
from util import cleanse_corpus


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(message)s')

# TODO: normalization of word-vectors in context-vector creation?

# constants
feat_num = 7  # number of words to build word-vectors for
dim_num = 2  # number of dimensions of word space
svd_dim_num = 2  # number of dimensions in svd-space
window_radius = 3  # how many words to include on each side of occurance
cluster_num = 2
ambiguous_words = ['hard', 'line', 'serve']



#train_corpus = ['"', 'But', 'I', "don't", 'want', 'to', 'go', 'among', 'mad', 'people', ',', '"', 'Alice', 'remarked', '.', 'Oh', ',',
#'you', "can't", 'help', 'that', ',', 'said', 'the', 'Cat', ':', "'we're", 'all', 'mad', 'here', '.', "I'm", 'mad', '.', "You're", 'mad', '.',
#'How', 'do', 'you', 'know', "I'm", 'mad', '?', 'said', 'Alice', '.', 'You', 'must', 'be', ',', 'said', 'the', 'Cat', ',', 'or', 'you',
#"wouldn't", 'have', 'come', 'here', '.']
#
#test_corpus = train_corpus


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

#    if matrix.shape[0] < matrix.shape[1]:
#        raise ValueError("Matrix has more columns (dimensions) than rows (feature vectors).")
#    if len(matrix[0]) < dim:
#        raise ValueError('Reduction to more dimensions than contained in matrix not possible.')

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


def compute_precision(computed, correct):
    logging.info("calculating precison")
    for word, labels in computed.items():
        translater_comp_corr = {}  # maps from cluster num to sense label
        translater_corr_comp = {}  # maps from sense label to cluster num
        hit = 0
        size = 0

        corr_labels = correct[word]
        for comp, corr in zip(labels, corr_labels):
            baseline = Counter(corr_labels).most_common()[0][1]  # count of most common

            size += 1
            if not comp in translater_comp_corr and\
                    not corr in translater_corr_comp:
                # first encounter of noth labels, assign them to each
                # other, its a hit by definition
                translater_comp_corr[comp] = corr
                translater_corr_comp[corr] = comp
                hit += 1

            elif not comp in translater_comp_corr and\
                    corr in translater_corr_comp:
                # first encounter of computed label, but correct
                # label is assigned to another compted label -->
                # thats a wrong classification
                pass

            elif comp in translater_comp_corr and\
                    corr in translater_corr_comp:
                # know both labels, check if they match
                if translater_comp_corr[comp] == corr:
                    hit += 1

        logging.info("{}-translater: {}".format(word, translater_comp_corr))
        print "{}: {}/{} correct, {}/{} baseline".format(word,
                                                         hit,
                                                         size,
                                                         baseline,
                                                         size)


def train_sec_order(corpus, ambigous_words):
    logging.info("Start train second order co-occurence")
    stemmer = PorterStemmer()

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
        offsets = offset_index.offsets(stemmer.stem(word))
        for offset in offsets:
            context = sized_context(offset, window_radius, filtered)
            vectors.append(context_vector_from_context(context, word_vectors))

        # perform svd and dimension reduction
        context_matrix = vstack(vectors)
        svd_matrix = svd_reduced_eigenvectors(context_matrix, svd_dim_num)

        # create sense vectors for ambigous context vectors
        svd_centroids, labels = kmeans2(svd_matrix, cluster_num)

        #draw_word_senses(svd_centroids, svd_matrix, labels)

        # labels tell which context belongs to which cluster in svd
        # space. Compute centroids in word space according to that
        centroids = []
        for i in range(cluster_num):
            cluster_i = [vector for vector, label in\
                         zip(vectors, labels) if label == i]
            try:
                centroids.append(npsum(vstack(cluster_i), 0))
            except ValueError:
                logging.warning("Empty sense vector")
                centroids.append(zeros(dim_num))

        sense_vectors[word] = centroids

        #draw_word_senses(svd_centroids, svd_matrix, labels)
        #draw_word_senses(vstack(centroids), context_matrix, labels)

    logging.info("  sense vectors:{}".format(sense_vectors))
    logging.info("  word vectors: {}".format(word_vectors.items()))
    logging.info("end train")
    return sense_vectors, word_vectors


def test_sec_order(corpus, ambiguous_words, sense_vectors, word_vectors, all_offsets):
    logging.info("Start test second order co-occurence")
    stemmer = PorterStemmer()
    word_context_labels = {}

    for ambiguous_word in ambiguous_words:
        logging.info("   test for: {}".format(ambiguous_word))

        # no need to cleanse, done in split_corpus already
        # filtered = cleanse_corpus(corpus[ambiguous_word])
        filtered = corpus[ambiguous_word]

        offsets = all_offsets[ambiguous_word]

        # find all contexts for the word and assign them to a sense
        context_labels = []
        for offset in offsets:
            if filtered[offset] != stemmer.stem(ambiguous_word):
                raise ValueError("Word at offset {} is {}: not an ambiguous word".
                                 format(offset, filtered[offset]))

            context = sized_context(offset, window_radius, filtered)
            context_vector = context_vector_from_context(context, word_vectors)
            label = assign_sense(context_vector, sense_vectors[ambiguous_word])
            context_labels.append(label)

        word_context_labels[ambiguous_word] = context_labels

    logging.info("labels(h/l/s): {}/{}/{}".
                 format(
                     len(word_context_labels['hard']),
                     len(word_context_labels['line']),
                     len(word_context_labels['serve'])))

    logging.info("end test")
    return word_context_labels

train_corpus, test_corpus, correct_labels, offsets = split_corpus()

senses, words = train_sec_order(train_corpus, ambiguous_words)
labels = test_sec_order(test_corpus, ambiguous_words, senses, words, offsets)

print 'labels ', labels
print 'correct labels ', correct_labels

compute_precision(labels, correct_labels)
