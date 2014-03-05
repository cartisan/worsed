from nltk.corpus import senseval as seval
from random import sample
import logging

from util import cleanse_corpus_pos_aware

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(message)s')


def sense_anal(word):
    print "word: ", word
    senses = []
    for inst in seval.instances(word):
        senses += inst.senses
    print 'senses: ', set(senses)
    print "sentences: ", len(senses)

    borders = [(0, senses[0])]
    for i in range(1, len(senses)):
        if senses[i] != senses[i-1]:
            borders.append((i, senses[i]))
    print "borders: ", borders

# Using sense_anal we find the following:
# borders:  [(0, 'HARD1'), (3455, 'HARD2'), (3957, 'HARD3')]
# h1: 3455, h2: 502 --> 3957  [0:3957]
# big dispartity --> 3002 [0:2500] + [3455:3957]

# borders:  [(0, 'cord'), (373, 'division'), (747, 'formation'), (1096, 'phone'), (1525, 'product'), (3742, 'text')]
# phone: 429, product: 2217 --> 2646 [1096:3742]

# borders:  [(0, 'SERVE10'), (1814, 'SERVE12'), (3086, 'SERVE2'), (3939, 'SERVE6')]
# serve10: 1814, serve12: 1272 --> 3086 [0:3086]


def split_corpus():

    logging.info("start corpus")
    logging.info("  restriction starts")

    # restrict corpora to 2 most common senses
    hard = seval.instances("hard.pos")[0:600]
    hard += seval.instances("hard.pos")[3455:3957]
    line = seval.instances("line.pos")[1096:2200]
    serve = seval.instances("serve.pos")[0:600]
    serve = seval.instances("serve.pos")[2486:3086]

    logging.info("  value setting starts")

    # smallest corpus has 2646 entries, for simplicity we restrict the
    # num of samples in all corpora to that
    #sample_range = 2646
    sample_range = 1100
    sample_num = 1100

    train_p, test_p = 0.8, 0.2
    train = []
    test = {'hard': [], 'line': [], 'serve': []}
    labels = {'hard': [], 'line': [], 'serve': []}
    offsets = {'hard': [], 'line': [], 'serve': []}
    corpora = [hard, line, serve]
    samples = sample(range(sample_range), sample_num)  # random order for sentences
    border = int(sample_num * train_p)

    logging.info("  training samples start")

    # ambiguous words alterning to prevent skew
    for i in samples[:border]:
        for corp in corpora:
            inst = corp[i]
            train += [w[0] for w in inst.context if isinstance(w, tuple)]

    logging.info("  test samples start")

    lengths = {'hard': 0, 'line': 0, 'serve': 0}
    for i in samples[border:]:
        for corp in corpora:
            inst = corp[i]
            word = inst.word.split('-')[0]

            # corpus.context somtimes contains non-tuple entries:
            new_sentence_dirty = []
            for w in inst.context:
                if isinstance(w, tuple):
                    new_sentence_dirty.append(w[0])
                else:
                    new_sentence_dirty.append(w)

            # need to perform cleansing here and not in worsed for offsets to
            # be aligned
            new_sentence, new_pos = cleanse_corpus_pos_aware(new_sentence_dirty,
                                                             inst.position)
            test[word] += new_sentence
            labels[word] += inst.senses
            offsets[word].append(new_pos + lengths[word])
            lengths[word] += len(new_sentence)

    logging.info("end corpus")
    logging.info("length train: {}, length test: {}".
                 format(len(train), len(test['hard']) + len(test['line'])
                 + len(test['serve'])))
    logging.info("labels(h/l/s): {}/{}/{}".
                 format(
                     len(labels['hard']),
                     len(labels['line']),
                     len(labels['serve'])))

    return train, test, labels, offsets
