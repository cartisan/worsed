from nltk.corpus import senseval as seval


def sense_anal(word):
    print "word: ", word
    senses = []
    for inst in seval.instances(word):
        senses += inst.senses
    print 'senses: ', set(senses)
    print "sentences: ", len(senses)

    borders = [0]
    for i in range(1,len(senses)):
        if senses[i] != senses[i-1]:
            borders.append(i)
    print "borders: ", borders


def split_corpus():
    train_p, test_p = 0.8, 0.2
    train, test = [], {'hard': [], 'line': [], 'serve': []}
    labels = {'hard': [], 'line': [], 'serve': []}

    corpora = ['hard.pos', 'line.pos', 'serve.pos']
    #sen_count = min([len(seval.instances(corp)) for corp in corpora])  # 4146
    #border = int(sen_count * train_p)

    sen_count = 6
    border = 3

    # train is 80% of corpus, ambiguous words alterning
    for i in range(border):
        for corp in corpora:
            inst = seval.instances(corp)[i]
            train += [w for (w, _) in inst.context]

    for i in range(border, sen_count):
        for corp in corpora:
            word = corp.split('.')[0]
            inst = seval.instances(corp)[i]
            test[word] += [w for (w, _) in inst.context]
            labels[word] += inst.senses

    return train, test, labels

tr, te, la = split_corpus()

#for f in seval.fileids():
#    border = int(len(seval.instances(f)) * train_p)
#    for inst in seval.instances(f)[:border]:
#        train += [w for (w, _) in inst.context]
#
#    word = seval.instances[0].word.split('-')[0]
#    test[word] = []
#    for inst in seval.instances(f)[border:]:
#        test[word] += [w for (w, _) in inst.context]
