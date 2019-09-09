
# coding: utf-8

import os
import torch
from pytorch_transformers.tokenization_bert import BertTokenizer
from model.bert import BertBiLSTMCRF
from train.pretrain import load_pretrain
from dataset.utils import convert_examples_to_features, convert_features_to_dataloader
from train.device import check_cuda
from sklearn.metrics import f1_score, classification_report
import numpy as np
from tqdm import tqdm

# import torchsnooper

class Predictor(object):

	def __init__(self, configs, model_class, processor):

		self.configs = configs
		self.processor = processor
		self.label_list = self.processor.get_labels()
		self.label2entities = self.processor.get_labels_to_entities()
		self.device, self.use_cuda = check_cuda(configs)
		self.fine_tune_dir = os.path.join(self.configs['finetune_model_dir'], model_class)
		self.model, self.tokenizer = load_pretrain(configs, model_class, self.fine_tune_dir, processor, eval = True)

		if self.use_cuda:
			self.model = self.model.cuda()

		self.entities_list = set(label.split('-')[1] for label in self.label_list if label != 'O')
	
	@staticmethod
	def class_report(y_pred, y_true):
		y_true = y_true.numpy()
		y_pred = y_pred.numpy()
		classify_report = classification_report(y_true, y_pred)
		print('\n\nclassify_report:\n', classify_report)

	def eval(self):

		test_examples = self.processor.get_test_examples(self.configs['data_dir'], tokenizer = self.tokenizer)
		test_features = convert_examples_to_features(examples = test_examples, 
													max_seq_length = self.configs['max_seq_length'], 
													tokenizer = self.tokenizer, 
													label_list = self.label_list)
		self.test_dataloader = convert_features_to_dataloader(test_features, batch_size = self.configs['batch_size'])

		self.model.eval()
		count = 0
		y_preds, y_labels = [], []

		# with torchsnooper.snoop():
		with torch.no_grad():
			for batch in tqdm(self.test_dataloader, ncols=75):
				input_ids, segment_ids, input_mask, label_ids = tuple(t.to(self.device) for t in batch)
				feats = self.model(input_ids, segment_ids, input_mask)
				predicts = self.model.predict(feats, input_mask)
				y_preds.append(predicts)
				y_labels.append(label_ids)

		self.y_preds = y_preds
		self.y_labels = y_labels

		eval_predict = torch.cat(y_preds, dim = 0).view(-1).cpu()
		eval_label = torch.cat(y_labels, dim = 0).view(-1).cpu()
		self.class_report(eval_predict, eval_label)

	@staticmethod
	def convert_ids_to_labels(label_ids, label_list):
		id_label_map = {i: label for i, label in enumerate(label_list)}
		return [id_label_map[i] for i in label_ids]

	@staticmethod
	def get_entity(tag_seq, char_seq, entity):
		length = len(char_seq)
		entities = []
		begin_tag = 'B-%s' %entity
		inter_tag = 'I-%s' %entity
		entity_ = []
		for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
			if tag == begin_tag or tag == inter_tag:
				entity_.append(char)
				if tag_seq[i+1] == 'O' or (i+1 == length):
				    entities.append(''.join(entity_))
				    entity_= []
			else:
			    continue
		return entities

	def predict_one(self, sentence, special_tokens = ['[CLS]', '[PAD]', '[SEP]']):

		predict_examples = self.processor.create_examples_from_zhsentence(sentence, self.tokenizer)
		predict_features = convert_examples_to_features(examples = predict_examples, 
													max_seq_length = self.configs['max_seq_length'], 
													tokenizer = self.tokenizer, 
													label_list = self.label_list)
		input_ids = torch.tensor([f.input_ids for f in predict_features], dtype=torch.long)
		segment_ids = torch.tensor([f.segment_ids for f in predict_features], dtype=torch.long)
		input_mask = torch.tensor([f.input_mask for f in predict_features], dtype=torch.long)
		label_ids = torch.tensor([f.label_ids for f in predict_features], dtype=torch.long)
		if self.use_cuda:
			input_ids = input_ids.cuda()
			segment_ids = segment_ids.cuda()
			input_mask = input_mask.cuda()
			label_ids = label_ids.cuda()
		with torch.no_grad():
			feats = self.model(input_ids, segment_ids, input_mask)
			predicts = self.model.predict(feats, input_mask)

			tokens = self.tokenizer.convert_ids_to_tokens(input_ids.contiguous().view(-1).cpu().numpy())
			preds = self.convert_ids_to_labels(predicts.contiguous().view(-1).cpu().numpy(), self.label_list)

		result = [(tok, pred) for (tok, pred) in zip(tokens, preds) if tok not in special_tokens]

		entities = {}
		for tag in self.entities_list:
			entities[self.label2entities[tag]] = self.get_entity(preds, tokens, tag)
		return entities, result