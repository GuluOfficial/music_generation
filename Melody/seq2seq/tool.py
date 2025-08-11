import os
from config import root_path
import pickle

from utils.profile import Lang


def build_vocab(file, n_limit = 1):
    vocab = Lang()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '': continue
            vocab.addSentence(line.split(' '))

    new_vocab = Lang()
    for i in range(4, vocab.n_items):
        term = vocab.index2item[i]
        freq = vocab.item2count[term]
        if freq < n_limit: continue
        new_vocab.addItem(term)
        new_vocab.item2count[term] = freq
        
    return new_vocab


if __name__ == '__main__':
    en_file = os.path.join(root_path, 'Melody/note2duration/en_train.txt')
    de_file = os.path.join(root_path, 'Melody/note2duration/de_train.txt')

    en_vocab = build_vocab(en_file)
    de_vocab = build_vocab(de_file)

    with open(os.path.join(root_path, 'Melody/checkpoints/seq2seq/note2duration/en_vocab.pkl'), 'wb') as fd:
        pickle.dump(en_vocab, fd)
    with open(os.path.join(root_path, 'Melody/checkpoints/seq2seq/note2duration/de_vocab.pkl'), 'wb') as fd:
        pickle.dump(de_vocab, fd)