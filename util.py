from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string


def _word_acceptable(word):
    return (word.lower() not in stopwords.words('english') and
            word not in string.punctuation and
            word != "FRASL" and  # foreign language word marker?
            len(word) > 2)


def cleanse_corpus(corpus):
    """ Returns lowercased and stemmed copy of list with removed stop words
    and punctuation marks.
    """

    stemmer = PorterStemmer()
    return [stemmer.stem(word.lower()) for word in corpus if
            word.lower() not in stopwords.words('english') and
            word not in string.punctuation and
            len(word) > 2]


def cleanse_corpus_pos_aware(corpus, old_pos):
    """ Returns lowercased and stemmed copy of list with removed stop words
    and punctuation marks.
    """

    stemmer = PorterStemmer()
    new_corpus = []
    new_pos = -1

    for i, word in enumerate(corpus):
        if(_word_acceptable(word)):
            if i != old_pos:
                new_corpus.append(stemmer.stem(word.lower()))
            else:
                # mark word at position by making it upper case
                new_corpus.append(stemmer.stem(word.lower()).upper())

    # find marked word, save it's position and unmark it
    for i, word in enumerate(new_corpus):
        if word.isupper():
            new_pos = i
            new_corpus[i] = new_corpus[i].lower()
            break

    #assert new_pos > -1, "marked word was deleted during cleanseing"
    if new_pos < 0:
        print "++++++++++++++++++++++"
        print corpus
        print old_pos
        print new_corpus
        print new_pos
        raise ValueError("marked word was deleted during cleanseing")
    return new_corpus, new_pos
