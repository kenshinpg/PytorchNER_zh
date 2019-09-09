
# coding: utf-8

import os
import math
import random
import numpy as np

def get_conll_data(data_file):
    data = []
    with open(data_file, 'r', encoding = 'utf-8') as f:
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
                data.append([chars, labels])
                chars = []
                labels = []
    return data

def shuffle(data_ids, random_seed = 1301):
    random.seed(random_seed)
    random.shuffle(data_ids)

def conll_to_train_test_dev(data_file, output_dir, random_seed = 1301, validation = 0.2):
    data = get_conll_data(data_file)
    data_count = len(data)
    data_ids = list(range(data_count))

    dev_div = validation
    test_div = validation
    train_div = 1 - dev_div - test_div

    # 因为数据集可能会出现按顺序分布不均的情况，所以需要对data_ids进行shuffle
    # 比如ccks2019数据集就有病史特点，出院情况，一般项目，诊疗经过
    shuffle(data_ids, random_seed)

    train_ids = data_ids[:math.ceil(data_count * train_div)]
    dev_ids = data_ids[math.ceil(data_count * train_div):math.ceil(data_count * (1 - test_div))]
    test_ids = data_ids[math.ceil(data_count * (1 - test_div)):]
    train_data = np.array(data)[train_ids]
    dev_data = np.array(data)[dev_ids]
    test_data = np.array(data)[test_ids]
    
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding = 'utf-8') as f:
        for index, (chars, labels) in enumerate(train_data):
            for char, label in zip(chars, labels):
                f.write(char + '\t' + label + '\n')
    f.close()
    with open(os.path.join(output_dir, 'dev.txt'), 'w', encoding = 'utf-8') as f:
        for index, (chars, labels) in enumerate(dev_data):
            for char, label in zip(chars, labels):
                f.write(char + '\t' + label + '\n')
    f.close()
    with open(os.path.join(output_dir, 'test.txt'), 'w', encoding = 'utf-8') as f:
        for index, (chars, labels) in enumerate(test_data):
            for char, label in zip(chars, labels):
                f.write(char + '\t' + label + '\n')
    f.close()

def oov_to_vocab(tok, tokenizer):
    if tok in tokenizer.vocab:
        return tok
    elif tok.lower() in tokenizer.vocab:
        return tok.lower()
    elif tok == '“':
        return '"'
    elif tok == '”':
        return '"'
    else:
        return '[UNK]'