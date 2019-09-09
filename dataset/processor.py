
# coding: utf-8

import os
import codecs
from dataset.conll import oov_to_vocab

class InputExample(object):

    def __init__(self, uid, tokens, labels):
        self.uid = uid
        self.tokens = tokens
        self.labels = labels

class DataProcessor(object):

	def get_train_examples(self, data_dir, tokenizer = None):
		raise NotImplementedError

	def get_dev_examples(self, data_dir, tokenizer = None):
		raise NotImplementedError

	def get_test_examples(self, data_dir, tokenizer = None):
		raise NotImplementedError	

	@staticmethod
	def create_examples_from_conll_format_file(data_file, set_type, tokenizer = None, all_Os = False):
		"""
		input file format 为conll类型
		all_Os: 是否把全是标签'O'的句子当做训练集
		"""
		examples = []
		index = 0
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
		        if tokenizer:
		        	chars.append(oov_to_vocab(char, tokenizer))		        	
		        else:
		        	chars.append(char)
		        labels.append(label)
		        if char in ['。','?','!','！','？']:
		            uid = "%s-%d" %(set_type, index)
		            # 如果单句标签全为O则不作为训练样本
		            if labels != ['O'] * len(labels) or all_Os: 
		            	examples.append(InputExample(uid = uid, tokens = chars, labels = labels))
		            chars = []
		            labels = []
		return examples

	@staticmethod
	def create_examples_from_zhsentence(sentence, tokenizer = None):
		"""
		对单句做chars-examples的转化
		"""
		examples = []
		chars = []
		for char in sentence:
			if tokenizer:
				chars.append(oov_to_vocab(char, tokenizer))
			else:
				chars.append(char)
			if char in ['。','?','!','！','？']:
				uid = None
				labels = ['O'] * len(chars)
				examples.append(InputExample(uid = uid, tokens = chars, labels = labels))
				chars = []
				labels = []
		return examples

	@staticmethod
	def get_labels():
		raise NotImplementedError()

	@staticmethod
	def get_labels_to_entities():
		raise NotImplementedError()

class CCKS2019Processor(DataProcessor):
	def get_train_examples(self, data_dir, tokenizer = None):
		return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, 'train.txt'), 'train', tokenizer = tokenizer, all_Os = True)

	def get_dev_examples(self, data_dir, tokenizer = None):
	    return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, 'dev.txt'), 'dev', tokenizer = tokenizer, all_Os = True)

	def get_test_examples(self, data_dir, tokenizer = None):
	    return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, 'test.txt'), 'test', tokenizer = tokenizer, all_Os = True)

	@staticmethod
	def get_labels():
		label_type = ['O', 'B-LABCHECK', 'I-LABCHECK','B-PICCHECK','I-PICCHECK','B-SURGERY','I-SURGERY',
							'B-DISEASE','I-DISEASE','B-DRUGS','I-DRUGS','B-ANABODY','I-ANABODY']
		return label_type

	@staticmethod
	def get_labels_to_entities():
		label_entities_map = {
	      'LABCHECK': '实验室检验',
	      'PICCHECK': '影像检查',
	      'SURGERY': '手术',
	      'DISEASE': '疾病和诊断',
	      'DRUGS': '药物',
	      'ANABODY': '解剖部位'
		}
		return label_entities_map