
# coding: utf-8

import os
import sys
import torch

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from train.device import check_cuda
from train.pretrain import load_pretrain
from train.optimizer import load_optimizer
from train.plot import eval_plot
from dataset.utils import convert_examples_to_features, convert_features_to_dataloader
from utils.datautils import check_dir

# import torchsnooper

class Trainer(object):

	def __init__(self, configs, model_class, processor):
		"""
		configs: 配置
		model_clas: str, 模型名称, 'BertBiLSTMCRF'
		processor: DataProcessor
		"""
		self.model_class = model_class
		self.device, use_gpu = check_cuda(configs)
		self.configs = configs
		self.fine_tune_dir = os.path.join(self.configs['finetune_model_dir'], model_class)
		self.model, self.tokenizer = load_pretrain(configs, model_class, self.fine_tune_dir, processor, eval = False)
		self.model.to(self.device)

		self.configs = configs
		self.batch_size = configs['batch_size']
		self.nb_epoch = configs['nb_epoch']
		self.max_seq_length = configs['max_seq_length']
		# 设置随机数
		self.random_seed = configs['random_seed']
		self.set_seed(use_gpu)
		train_examples = processor.get_train_examples(configs['data_dir'], tokenizer = self.tokenizer)
		dev_examples = processor.get_dev_examples(configs['data_dir'], tokenizer = self.tokenizer)

		self.configs['num_train_steps'] = int(len(train_examples)/self.batch_size) * self.nb_epoch
		self.optimizer, self.scheduler = load_optimizer(self.configs, self.model)
		self.max_grad_norm = configs['max_grad_norm']
		self.label_list = processor.get_labels()

		train_features = convert_examples_to_features(examples = train_examples, 
														max_seq_length = self.max_seq_length, 
														tokenizer = self.tokenizer, 
														label_list = self.label_list)
		dev_features = convert_examples_to_features(examples = dev_examples, 
														max_seq_length = self.max_seq_length, 
														tokenizer = self.tokenizer, 
														label_list = self.label_list)
		self.train_count = len(train_examples)
		self.train_dataloader = convert_features_to_dataloader(train_features, batch_size = configs['batch_size'])
		self.dev_dataloader = convert_features_to_dataloader(dev_features, batch_size = configs['batch_size'])
		self.max_patience = configs['max_patience']

	def set_seed(self, use_gpu):
		random.seed(self.random_seed)
		np.random.seed(self.random_seed)
		torch.manual_seed(self.random_seed)
		if use_gpu:
			torch.cuda.manual_seed(self.random_seed)
			torch.cuda.manual_seed_all(self.random_seed)

	def train(self):

		best_dev_loss = 1.e8
		current_patience = 0
		for epoch in range(self.nb_epoch):

			torch.cuda.empty_cache()

			train_loss, dev_loss = 0., 0.
			train_loss_log, dev_loss_log = [], []
			self.model.train()
			iter_variable = 0
			# with torchsnooper.snoop():
			for i, train_batch in enumerate(self.train_dataloader):

				torch.cuda.empty_cache()

				self.optimizer.zero_grad()
				input_ids, segment_ids, input_mask, label_ids= tuple(t.to(self.device) for t in train_batch)
				feats = self.model(input_ids, segment_ids, input_mask)
				loss = self.model.loss_fn(feats, input_mask, label_ids)
				loss.backward()
				with torch.no_grad():
					train_loss += float(loss.item())
					train_loss_log.append(float(loss.item()))
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
				self.optimizer.step()
				self.scheduler.step()
				iter_variable += self.batch_size
				if iter_variable > self.train_count:
					iter_variable = self.train_count
				sys.stdout.write('Epoch {0}/{1}: {2}/{3}\r'.format(epoch+1, self.nb_epoch, 
							iter_variable, self.train_count))
			# early stopping
			self.model.eval()
			with torch.no_grad():
				for dev_batch in self.dev_dataloader:

					torch.cuda.empty_cache()

					input_ids, segment_ids, input_mask, label_ids = tuple(t.to(self.device) for t in dev_batch)
					feats = self.model(input_ids, segment_ids, input_mask)
					loss = self.model.loss_fn(feats, input_mask, label_ids)				
					dev_loss += float(loss.item())
					dev_loss_log.append(float(loss.item()))
		
			print('\ttrain loss: {0}, dev loss: {1}'.format(train_loss, dev_loss))
			# early stopping
			if dev_loss < best_dev_loss:
				current_patience = 0
				best_dev_loss = dev_loss
				self.save_model()
			else:
				current_patience += 1
				if self.max_patience <= current_patience:
					print('finished training! (early stopping, max_patience: {0})'.format(self.max_patience))
					return

		# eval_plot(self.configs, train_loss_log, dev_loss_log)
		print('finished training!')

	def save_model(self):
		self.model.save_pretrained(self.fine_tune_dir)
		self.tokenizer.save_pretrained(self.fine_tune_dir)