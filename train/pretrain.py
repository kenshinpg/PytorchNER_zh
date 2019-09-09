
# coding: utf-8

import os
import torch
import re

from utils.datautils import check_dir
from model.bert import Bert, BertCRF, BertBiLSTMCRF
from model.bilstm import BiLSTM, BiLSTMCRF 
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers.tokenization_bert import BertTokenizer
from dataset.embedding import build_word_embed

def load_pretrain(configs, model_class, fine_tune_dir, processor, eval = False):
	"""
	configs: 配置文件
	model_class: 模型名称
	fine_tune_dir: 微调模型保存路径
	processor: DataProcessor
	eval: 是否验证
	"""

	model_class_map = {
			'Bert': Bert,
			'BertCRF': BertCRF,
			'BertBiLSTMCRF': BertBiLSTMCRF,
			'BiLSTM': BiLSTM,
			'BiLSTMCRF': BiLSTMCRF
		}
	model_class_ = model_class_map[model_class]
	label_list = processor.get_labels()

	check_dir(fine_tune_dir)
	if eval:
		model_pretrained_path = fine_tune_dir
	else:
		model_pretrained_path = configs['pretrained_model_dir']
	tokenizer = BertTokenizer.from_pretrained(model_pretrained_path, do_lower_case = configs['lower_case'])

	if model_class in ['Bert', 'BertCRF', 'BertBiLSTMCRF']:
		bert_config = BertConfig.from_pretrained(model_pretrained_path, 
											num_labels = len(label_list),
											finetuning_task="ner")
		model = model_class_.from_pretrained(model_pretrained_path, config = bert_config)

	elif model_class in ['BiLSTM', 'BiLSTMCRF']:
		configs['num_labels'] = len(label_list)
		if configs['use_pretrained_embedding']:
			pretrained_word_embed = build_word_embed(tokenizer,
				pretrain_embed_file = configs['pretrain_embed_file'],
				pretrain_embed_pkl =  configs['pretrain_embed_pkl'])
			configs['word_vocab_size'] = pretrained_word_embed.shape[0]
			configs['word_embedding_dim'] = pretrained_word_embed.shape[1]
		else:
			pretrained_word_embed = None
		if eval:
			model_pretrained_path = fine_tune_dir
			model = model_class_.from_pretrained(model_pretrained_path, pretrained_word_embed)
		else:
			model = model_class_(configs, pretrained_word_embed)
	else:
		raise ValueError("Invalid Model Class")
	return model, tokenizer