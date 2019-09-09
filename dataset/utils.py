
# coding: utf-8

import numpy as np
import logging

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.output_mask = output_mask

def convert_examples_to_features(examples, 
                                 max_seq_length, 
                                 tokenizer,
                                 label_list,
                                 pad_type = 'head+tail'):
    # 标签转换为数字
    label_map = {label: i for i, label in enumerate(label_list)}

    # load sub_vocab
    # sub_vocab = {}
    # with open(vocab_file, 'r', encoding = 'utf-8') as fr:
    #     for line in fr:
    #         _line = line.strip('\n')
    #         if "##" in _line and sub_vocab.get(_line) is None:
    #             sub_vocab[_line] = 1

    features = []
    for ex_index, example in enumerate(examples):
        tokens = example.tokens
        labels = example.labels
        
        if len(tokens)==0 or len(labels)==0:
            continue
            
        if len(tokens) > max_seq_length - 2:
            if pad_type == 'head-only':
                tokens = tokens[:(max_seq_length-2)]
                labels = labels[:(max_seq_length-2)]
            elif pad_type == 'tail-only':
                tokens = tokens[(max_seq_length-2):]
                labels = labels[(max_seq_length-2):]
            elif pad_type == 'head+tail':
                tokens = tokens[:round((max_seq_length-2)/4)] + tokens[-round(3 * (max_seq_length-2)/4):]
                labels = labels[:round((max_seq_length-2)/4)] + labels[-round(3 * (max_seq_length-2)/4):]                                
            else:
                raise ValueError('Unknown `pad_type`: ' + str(pad_type))
        # ----------------处理source--------------
        ## 句子首尾加入标示符
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        # segment_ids = [sequence_a_segment_id] * len(tokens)
        # sequence_a_segment_id = 0, sequence_b_segment_id = 1
        segment_ids = [0] * len(tokens)
        ## 词转换成数字
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # ---------------处理target----------------
        ## Notes: label_id中不包括[CLS]和[SEP]
        label_ids = [label_map[l] for l in labels]
        label_ids = [0] + label_ids + [0]
        label_padding = [0] * (max_seq_length-len(label_ids))
        label_ids += label_padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        # label_ids = to_categorical(label_ids.cpu(), num_classes = len(label_list))

        ## output_mask用来过滤bert输出中sub_word的输出,只保留单词的第一个输出(As recommended by jocob in his paper)
        ## 此外，也是为了适应crf
        # output_mask = [0 if sub_vocab.get(t) is not None else 1 for t in tokens]
        # output_mask = [0] + output_mask + [0]
        # output_mask += padding

        # ----------------处理后结果-------------------------
        # for example, in the case of max_seq_length=10:
        # raw_data:          春 秋 忽 代 谢le
        # token:       [CLS] 春 秋 忽 代 谢 ##le [SEP]
        # input_ids:     101 2  12 13 16 14 15   102   0 0 0
        # input_mask:      1 1  1  1  1  1   1     1   0 0 0
        # label_id:          T  T  O  O  O
        # output_mask:     0 1  1  1  1  1   0     0   0 0 0
        # --------------看结果是否合理------------------------


        feature = InputFeatures(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               # output_mask=output_mask,
                               label_ids=label_ids)
        features.append(feature)

    return features

def convert_features_to_dataloader(features, batch_size):

    """
        input_ids: size=(batch_size, max_seq_length)
        input_mask: size=(batch_size, max_seq_length)
        segment_ids: size=(batch_size, max_seq_length)
        label_ids: size=(batch_size, max_seq_length)
    """
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    # output_mask = torch.tensor([f.output_mask for f in features], dtype=torch.long)

    data = TensorDataset(input_ids, segment_ids, input_mask, label_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler = sampler, batch_size= batch_size)
    return dataloader