'''
@Author: xiaoyao jiang
@Date: 2020-04-09 17:01:08
@LastEditTime: 2020-07-17 16:40:29
@LastEditors: xiaoyao jiang
@Description: build dictionary
@FilePath: /bookClassification/src/data/dictionary.py
'''

from random import shuffle


class Dictionary(object):
    def __init__(self,  min_count=1, start_end_tokens=True, wordvec_mode=None, types = 'lyric'):
        # 定义所需要参数
        self.min_count = min_count
        self.start_end_tokens = start_end_tokens
        self.wordvec_mode = wordvec_mode
        self.PAD_TOKEN = '<PAD>'
        self.type = types

    def build_dictionary(self, data):
        # 构建词典主方法， 使用_build_dictionary构建
        if self.type == 'lyric':
            self.voacab_words, self.word2idx, self.idx2word = self._build_dictionary(data)
        elif self.type == 'note':
            self.voacab_words, self.word2idx, self.idx2word = self.build_notes(data)
        self.vocabulary_size = len(self.voacab_words)
    
    def build_notes(self, data):
        vocab_words = [self.PAD_TOKEN, '<UNK>']
        vocab_size = 2 
        if self.start_end_tokens:
            vocab_words += ['<BOS>', '<EOS>']
            vocab_size += 2
        wl = list(range(49, 82))
        shuffle(wl)
        vocab_words = vocab_words + wl
        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words
        return vocab_words, word2idx, idx2word


    def indexer(self, word):
        # 根据词获取到对应的id
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<UNK>']

    def decode_ids(self, indexs):
        result = []
        for index in indexs:
            word = self.decode_id(index)
            result.append(word)
        
        return result
    
    def decode_id(self, index):
        try:
            return self.idx2word[index]
        except:
            return '<UNK>'
    
    def encode(self, sent):
      '''对相应的片段分词获取相应的编码id
      sent:list
      '''
      ids = []
      for w in sent:
        index = self.indexer(w)
        ids.append(index)

      return ids
    
    def get_all_words(self, data):
        words = []
        for sent in data:
            sent_list = sent.split(' ')
            for w in sent_list:
                if w not in words:
                    words.append(w)
        
        shuffle(words)

        return words


    def _build_dictionary(self, data):
        # 加入UNK标示， 按照需要加入EOS 或者EOS
        vocab_words = [self.PAD_TOKEN, '<UNK>']
        vocab_size = 2 
        if self.start_end_tokens:
            vocab_words += ['<BOS>', '<EOS>']
            vocab_size += 2

        words = self.get_all_words(data)
        vocab_words = vocab_words + words
        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words
        return vocab_words, word2idx, idx2word

    
