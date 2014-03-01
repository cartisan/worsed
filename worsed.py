import pdb
import string
from nltk.corpus import stopwords, brown
from nltk.probability import FreqDist
from nltk import ConcordanceIndex
from numpy import array

# constants
dim_num = 2
feat_num = 7
window_radius = 7  # how many words to include at each side of occurance

# containers
word_vectors = {}  # maps from features words to lists containing their vectors

test_corpus = ['"', 'But', 'I', "don't", 'want', 'to', 'go', 'among', 'mad', 'people', ',', '"', 'Alice', 'remarked', '.', 'Oh', ',',
'you', "can't", 'help', 'that', ',', 'said', 'the', 'Cat', ':', "'we're", 'all', 'mad', 'here', '.', "I'm", 'mad', '.', "You're", 'mad', '.',
'How', 'do', 'you', 'know', "I'm", 'mad', '?', 'said', 'Alice', '.', 'You', 'must', 'be', ',', 'said', 'the', 'Cat', ',', 'or', 'you',
"wouldn't", 'have', 'come', 'here', '.']


def cleanse_corpus(corpus):
    " Returns copy of list with removed stop words and punctuation marks"

    return [word for word in corpus if
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


def word_vector_from_context(contet, dimensions):
    pass


def train(corpus):
    # remove stop words and signs
    filtered = cleanse_corpus(corpus)

    # find dimensions and features
    words_desc = FreqDist(filtered).keys()
    dimensions = words_desc[:dim_num]
    features = words_desc[:feat_num]


    # create word vectors for features
    offset_index = ConcordanceIndex(filtered, key=lambda s: s.lower())
    for word in features:
        # get word positions in text
        offsets = offset_index.offsets(word)

        for offset in offsets:
            context = sized_context(offset, window_radius, filtered)
            pdb.set_trace() ############################## Breakpoint ##############################
            #word_vector = word_vector_from_context(context, dimensions)
            #word_vectors[word] = word_vector
            # update word_vector for context

    # create context vectors for ambigous words
    # create sense vectors for ambigous context vectors


train(test_corpus)
