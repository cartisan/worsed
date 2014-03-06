import logging
from collections import Counter
from nltk.probability import FreqDist
from nltk import ConcordanceIndex
from nltk.stem.porter import PorterStemmer
from numpy import zeros, vstack, mean
from numpy import sum as npsum
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans

from senseval_adapter import split_corpus
from cleanser import cleanse_corpus
from util import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

# constants
feat_num = 4000  # number of words to build word-vectors for
dim_num = 400  # number of dimensions of word space
svd_dim_num = 100  # number of dimensions in svd-space
window_radius = 25  # how many words to include on each side of occurance
cluster_num = 2
ambiguous_words = ['hard', 'line', 'serve']


def compute_precision(computed, correct):
    logging.info("calculating precison")
    precision = []
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

        precision.append(float(hit)/size)
        logging.info("  {}-translater: {}".format(word, translater_comp_corr))
        print "{}: {}/{} correct, {}/{} baseline".format(word,
                                                         hit,
                                                         size,
                                                         baseline,
                                                         size)
    print "avg-precision: {}".format(mean(precision))


def train_fir_order(corpus, ambigous_words):
    logging.info("Start train first order co-occurence")
    stemmer = PorterStemmer()

    # containers
    sense_vectors = {}  # maps ambiguous words to ndarray of sense vectors
    estimators = {}

    # remove stop words and signs
    logging.info("  Start stemming and cleansing corpus")
    filtered = cleanse_corpus(corpus)
    logging.info("  {} different words after cleansing".format(
                 len(set(filtered))))

    # find dimensions
    logging.info("  Start finding dimensions")
    words_desc = FreqDist(filtered).keys()
    dimensions = words_desc[:dim_num]
    offset_index = ConcordanceIndex(filtered, key=lambda s: s.lower())

    for word in ambigous_words:
        logging.info("  Start train: {}".format(word))
        estimator = KMeans(cluster_num, "k-means++", n_init=20)

        # create context vectors for ambigous words
        logging.info("    Start creating sense vectors")
        vectors = []
        offsets = offset_index.offsets(stemmer.stem(word))
        for offset in offsets:
            context = sized_context(offset, window_radius, filtered)
            vectors.append(word_vector_from_context(context, dimensions))

        # perform svd and dimension reduction
        logging.info("    Start svd reduction")
        context_matrix = vstack(vectors)
        svd_matrix = svd_reduced_eigenvectors(context_matrix, svd_dim_num)

        # create sense vectors for ambigous context vectors
        logging.info("    Start clustering")

        # +++++++++ SVD switch here +++++++++++
        #estimator.fit(context_matrix)
        estimator.fit(svd_matrix)

        labels = estimator.labels_
        estimators[word] = estimator

        # labels tell which context belongs to which cluster in svd
        # space. Compute centroids in word space according to that
        logging.info("    Start centroid computation")
        centroids = []
        for i in range(cluster_num):
            cluster_i = [vector for vector, label in\
                         zip(vectors, labels) if label == i]
            try:
                centroids.append(npsum(vstack(cluster_i), 0))
            except ValueError:
                logging.warning("CRITICAL: Empty sense vector")
                centroids.append(zeros(dim_num))

        sense_vectors[word] = centroids

        #draw_word_senses(svd_centroids, svd_matrix, labels)
        #draw_word_senses(vstack(centroids), context_matrix, labels)

    logging.info("  sense vectors:{}".format(
        len(sense_vectors['line'])))
    logging.info("end train")
    return sense_vectors, dimensions, estimators


def train_sec_order(corpus, ambigous_words):
    logging.info("Start train second order co-occurence")
    stemmer = PorterStemmer()

    # containers
    word_vectors = {}  # maps feature-words to lists containing their vectors
    sense_vectors = {}  # maps ambiguous words to ndarray of sense vectors
    estimators = {}

    # remove stop words and signs
    logging.info("  Start stemming and cleansing corpus")
    filtered = cleanse_corpus(corpus)
    logging.info("  {} different words after cleansing".format(
                 len(set(filtered))))

    # find dimensions and features
    logging.info("  Start finding features and dimensions")
    words_desc = FreqDist(filtered).keys()
    dimensions = words_desc[:dim_num]
    features = words_desc[:feat_num]

    # create word vectors for features
    logging.info("  Start creating word vectors")
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
        logging.info("  Start train: {}".format(word))
        estimator = KMeans(cluster_num, "k-means++", n_init=20)

        # create context vectors for ambigous words
        logging.info("    Start creating sense vectors")
        vectors = []
        offsets = offset_index.offsets(stemmer.stem(word))
        for offset in offsets:
            context = sized_context(offset, window_radius, filtered)
            vectors.append(context_vector_from_context(context, word_vectors))

        # perform svd and dimension reduction
        logging.info("    Start svd reduction")
        context_matrix = vstack(vectors)
        svd_matrix = svd_reduced_eigenvectors(context_matrix, svd_dim_num)

        # create sense vectors for ambigous context vectors
        logging.info("    Start clustering")

        # +++++++++ SVD switch here +++++++++++
        #estimator.fit(context_matrix)
        estimator.fit(svd_matrix)

        labels = estimator.labels_
        estimators[word] = estimator

        # labels tell which context belongs to which cluster in svd
        # space. Compute centroids in word space according to that
        logging.info("    Start centroid computation")
        centroids = []
        for i in range(cluster_num):
            cluster_i = [vector for vector, label in
                         zip(vectors, labels) if label == i]
            try:
                centroids.append(npsum(vstack(cluster_i), 0))
            except ValueError:
                logging.warning("CRITICAL: Empty sense vector")
                centroids.append(zeros(dim_num))

        sense_vectors[word] = centroids

        #draw_word_senses(svd_centroids, svd_matrix, labels)
        #draw_word_senses(vstack(centroids), context_matrix, labels)

    logging.info("  sense vectors:{}".format(
        len(sense_vectors['line'])))
    logging.info("  word vectors: {}".format(
        len(word_vectors.items())))
    logging.info("end train")
    return sense_vectors, word_vectors, estimators


def test_fir_order(corpus, ambiguous_words, estimators,
                   all_offsets, dimensions):
    logging.info("Start test second order co-occurence")
    stemmer = PorterStemmer()
    labels = {}

    for ambiguous_word in ambiguous_words:
        logging.info("  test for: {}".format(ambiguous_word))

        # no need to cleanse, done in split_corpus already
        # filtered = cleanse_corpus(corpus[ambiguous_word])
        filtered = corpus[ambiguous_word]

        offsets = all_offsets[ambiguous_word]

        # find all contexts for the word and assign them to a sense
        context_vectors = []
        for offset in offsets:
            if filtered[offset] != stemmer.stem(ambiguous_word):
                raise ValueError("Word at offset {} is {}: not an ambiguous word".
                                 format(offset, filtered[offset]))

            context = sized_context(offset, window_radius, filtered)
            context_vectors.append(word_vector_from_context(context,
                                                            dimensions))

        context_matrix = vstack(context_vectors)
        svd_matrix = svd_reduced_eigenvectors(context_matrix,
                                              svd_dim_num)

        # +++++++++ SVD switch here +++++++++++
        labels[ambiguous_word] = estimators[ambiguous_word].predict(svd_matrix)
        #labels[ambiguous_word] = estimators[ambiguous_word].predict(context_matrix))

    for k, v in labels.items():
        logging.info("  {} label counts: {}".
                     format(k, Counter(v).most_common()))
    logging.info("labels(h/l/s): {}/{}/{}".
                 format(
                     len(labels['hard']),
                     len(labels['line']),
                     len(labels['serve'])))
    logging.info("end test")

    return labels


def test_sec_order(corpus, ambiguous_words, estimators,
                   all_offsets, word_vectors):
    logging.info("Start test second order co-occurence")
    stemmer = PorterStemmer()
    labels = {}

    for ambiguous_word in ambiguous_words:
        logging.info("  test for: {}".format(ambiguous_word))

        # no need to cleanse, done in split_corpus already
        # filtered = cleanse_corpus(corpus[ambiguous_word])
        filtered = corpus[ambiguous_word]

        offsets = all_offsets[ambiguous_word]

        # find all contexts for the word and assign them to a sense
        context_vectors = []
        for offset in offsets:
            if filtered[offset] != stemmer.stem(ambiguous_word):
                raise ValueError("Word at offset {} is {}: not an ambiguous word".
                                 format(offset, filtered[offset]))

            context = sized_context(offset, window_radius, filtered)
            context_vectors.append(context_vector_from_context(context,
                                                               word_vectors))

        context_matrix = vstack(context_vectors)
        svd_matrix = svd_reduced_eigenvectors(context_matrix,
                                              svd_dim_num)

        # +++++++++ SVD switch here +++++++++++
        labels[ambiguous_word] = estimators[ambiguous_word].predict(svd_matrix)
        #labels[ambiguous_word] = estimators[ambiguous_word].predict(context_matrix))

    for k, v in labels.items():
        logging.info("  {} label counts: {}".format(k,
                                                    Counter(v).most_common()))
    logging.info("labels(h/l/s): {}/{}/{}".
                 format(
                     len(labels['hard']),
                     len(labels['line']),
                     len(labels['serve'])))
    logging.info("end test")

    return labels


logging.disable(logging.CRITICAL)
train_corpus, test_corpus, correct_labels, offsets = split_corpus()

#for feat_num in [100, 1000, 4000]:
#    for dim_num in [10, 100, 400]:
#        print feat_num, dim_num

for i in range(1):
    print "1st order"
    senses, dimensions, estim = train_fir_order(train_corpus, ambiguous_words)
    labels = test_fir_order(test_corpus, ambiguous_words, estim, offsets, dimensions)
    compute_precision(labels, correct_labels)

    print "2nd order"
    senses, words, estim = train_sec_order(train_corpus, ambiguous_words)
    labels = test_sec_order(test_corpus, ambiguous_words, estim, offsets, words)
    compute_precision(labels, correct_labels)

    print "#######################"
