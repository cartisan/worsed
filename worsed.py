import string
from collections import Counter
from nltk.corpus import stopwords, brown
from nltk.probability import FreqDist
from nltk import ConcordanceIndex
from numpy import array, zeros


# TODO: lemmatization?
# TODO: normalization of word-vectors in context-vector creation?

# constants
dim_num = 2
feat_num = 7
window_radius = 3  # how many words to include at each side of occurance
ambigous_words = ['cat']

# containers
word_vectors = {}  # maps from feature-words to lists containing their vectors
context_vectors = {}  # maps from ambiguous words to lists of context vectors

test_corpus = ['"', 'But', 'I', "don't", 'want', 'to', 'go', 'among', 'mad', 'people', ',', '"', 'Alice', 'remarked', '.', 'Oh', ',',
'you', "can't", 'help', 'that', ',', 'said', 'the', 'Cat', ':', "'we're", 'all', 'mad', 'here', '.', "I'm", 'mad', '.', "You're", 'mad', '.',
'How', 'do', 'you', 'know', "I'm", 'mad', '?', 'said', 'Alice', '.', 'You', 'must', 'be', ',', 'said', 'the', 'Cat', ',', 'or', 'you',
"wouldn't", 'have', 'come', 'here', '.']


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


def train_sec_order(corpus):
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

    # create context vectors for ambigous words
    for word in ambigous_words:
        vectors = []
        offsets = offset_index.offsets(word)
        for offset in offsets:
            context = sized_context(offset, window_radius, filtered)
            vectors.append(context_vector_from_context(context, word_vectors))
        context_vectors[word] = vectors

    # create sense vectors for ambigous context vectors


train_sec_order(test_corpus)
