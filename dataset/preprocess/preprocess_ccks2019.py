
# coding: utf-8

import os
import math
import random
import numpy as np

from collections import Counter

class CCKS2019NER(object):

    def __init__(self, configs, vocab_size = None, min_freq = 1, random_seed = 1301):

        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.random_seed = random_seed
        self.trainfile = configs['label_file']
        self.configs = configs

        self.txtloader()
        self.label_dict = configs['label']['label2id']
        self.class_dict = configs['label']['class2label']
        self.txtnerlabel()
        self.nervocab()
        self.get_raw_data()
        self.get_train_data()

        self.data_count = len(self.data)
        self.data_ids = list(range(self.data_count))
        self.train_test_split()

    def txtloader(self):
        self.originalText = {}
        self.entities = {}
        with open(os.path.join(self.configs['txt_path'], 'subtask1_training_part1.txt'), 'r', encoding = 'utf-8') as f:
            i = 0
            for line in f:
                self.originalText[i] = eval(line)['originalText']
                self.entities[i] = eval(line)['entities']
                i += 1
        f.close()
        with open(os.path.join(self.configs['txt_path'], 'subtask1_training_part2.txt'), 'r', encoding = 'utf-8') as f:
            for line in f:
                self.originalText[i] = eval(line)['originalText']
                self.entities[i] = eval(line)['entities']
                i += 1
        f.close()

    def txtnerlabel(self):
        if not os.path.exists(self.trainfile):
            with open(self.trainfile, 'w', encoding = 'utf-8') as f:
                for i in range(len(self.originalText)):
                    text = self.originalText[i]
                    res_dict = {}
                    for e in self.entities[i]:
                        start = e['start_pos']
                        end = e['end_pos']
                        label = self.configs['label']['class2label'][e['label_type']]
                        for i in range(start, end):
                            if i == start:
                                label_cate = 'B-' + label
                            else:
                                label_cate = 'I-' + label
                            res_dict[i] = label_cate
                    for indx, char in enumerate(text):
                        char_label = res_dict.get(indx, 'O')
                        f.write(char + '\t' + char_label + '\n')
                        # 保证每条文本末尾都以中文句号结尾
                        if indx == len(text)-1 and char not in ['。','?','!','！','？']:
                            f.write('。' + '\t' + 'O' + '\n')
            f.close()

    def nervocab(self):
        """
        获得NER所需要的字特征
        """
        if not os.path.exists(self.trainfile):
            self.txtnerlabel()
        words = []
        counter = Counter()
        with open(self.trainfile, 'r', encoding = 'utf-8') as f:
            for line in f:
                words.append(line[0])
        f.close()
        for word in words:
            counter[word] += 1

        # 将词token按词频freq排序，方便用vocab_size限制词表大小
        self.token_freqs = sorted(counter.items(), key = lambda tup: tup[0])
        self.token_freqs.sort(key = lambda tup: tup[1], reverse = True) 

        self.itos = []

        # 剔除低频词
        for tok, freq in self.token_freqs:
            if freq < self.min_freq or len(self.itos) == self.vocab_size:
                break
            self.itos.append(tok)

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def get_raw_data(self):
        if not os.path.exists(self.trainfile):
            self.txtnerlabel()
        self.raw_data = []
        with open(self.trainfile, 'r', encoding = 'utf-8') as f:
            chars = []
            labels = []
            for line in f:
                line = line.rstrip().split('\t')
                if not line:
                    continue
                char = line[0]
                if not char:
                    continue
                label = line[-1]
                chars.append(char)
                labels.append(label)
                if char in ['。','?','!','！','？']:
                    self.raw_data.append([chars, labels])
                    chars = []
                    labels = []
        f.close()

    def get_train_data(self):
        label2id = self.configs['label']['label2id']
        self.data = []
        for i, item in enumerate(self.raw_data):
            sentence = item[0]
            label = item[1]
            s2id = [self.stoi[tok] for tok in sentence]
            l2id = [label2id[la] for la in label]
            self.data.append([s2id, l2id])

    def shuffle(self):
        random.seed(self.random_seed)
        random.shuffle(self.data_ids)

    def train_test_split(self, validation = 0.3, random_seed = 1301):
        self.shuffle()
        train_ids = self.data_ids[:math.ceil(self.data_count * (1 - validation))]
        test_ids = self.data_ids[math.ceil(self.data_count * (1 - validation)):]
        self.train_data = np.array(self.data)[train_ids]
        self.test_data = np.array(self.data)[test_ids]