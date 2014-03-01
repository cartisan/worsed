import string
from nltk.corpus import stopwords, brown
from nltk.probability import FreqDist

dim_num = 2
feat_num = 7
window_size = 6

test_corpus = ['"', 'But', 'I', "don't", 'want', 'to', 'go', 'among', 'mad', 'people', ',', '"', 'Alice', 'remarked', '.', 'Oh', ',',
'you', "can't", 'help', 'that', ',', 'said', 'the', 'Cat', ':', "'we're", 'all', 'mad', 'here', '.', "I'm", 'mad', '.', "You're", 'mad', '.',
'How', 'do', 'you', 'know', "I'm", 'mad', '?', 'said', 'Alice', '.', 'You', 'must', 'be', ',', 'said', 'the', 'Cat', ',', 'or', 'you',
"wouldn't", 'have', 'come', 'here', '.']


def cleanse_corpus(corpus):
    " Returns copy of list with removed stop words and punctuation marks"
    return [word for word in corpus if
            word.lower() not in stopwords.words('english') and
            word not in string.punctuation]

def train(corpus):
    # remove stop words and signs
    filtered = cleanse_corpus(corpus)

    # find dimensions and features
    words_desc = FreqDist(filtered).keys()
    dimensions = words_desc[:dim_num]
    features = words_desc[:feat_num]

    # create word vectors for features
    # create context vectors for ambigous words
    # create sense vectors for ambigous context vectors


train(test_corpus)
