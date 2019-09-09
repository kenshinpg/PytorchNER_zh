
# coding: utf-8

import os
from configs.confighelper import config_loader, args_parser
from dataset.preprocess import CCKS2019NER
from dataset.conll import conll_to_train_test_dev
from dataset.processor import CCKS2019Processor
from train.trainer import Trainer
from train.eval import Predictor
from utils.datautils import check_dir

dataset_name_to_class = {
	'CCKS2019': (CCKS2019NER, CCKS2019Processor, './configs/ccks2019.yml')
}

def main():

	args = args_parser()
	if args.task == 'train':
		# conll process
		data_vocab_class, processor_class, conll_config_path = dataset_name_to_class[args.dataset]
		conll_configs = config_loader(conll_config_path)
		if not os.path.exists(os.path.join(conll_configs['data_path'], 'train.txt')):
			data_vocab = data_vocab_class(conll_configs)
			conll_to_train_test_dev(conll_configs['label_file'], conll_configs['data_path'])
		
		# config
		configs = config_loader(args.config_path)
		configs['data_dir'] = os.path.join(configs['data_dir'], args.dataset.lower())
		configs['finetune_model_dir'] = os.path.join(configs['finetune_model_dir'], args.dataset.lower())
		configs['output_dir'] = os.path.join(configs['output_dir'], args.dataset.lower())
		check_dir(configs['data_dir'])
		check_dir(configs['finetune_model_dir'])
		check_dir(configs['output_dir'])
		
		# train
		processor = processor_class()
		for model_class in configs['model_class']:
			print('Begin Training %s Model on corpus %s' %(model_class, args.dataset))
			trainer = Trainer(configs, model_class, processor)
			trainer.train()

	if args.task == 'eval':
		data_vocab_class, processor_class, conll_config_path = dataset_name_to_class[args.dataset]
		conll_configs = config_loader(conll_config_path)
		if not os.path.exists(os.path.join(conll_configs['data_path'], 'test.txt')):
			data_vocab = data_vocab_class(conll_configs)
			conll_to_train_test_dev(conll_configs['label_file'], conll_configs['data_path'])

		configs = config_loader(args.config_path)
		configs['data_dir'] = os.path.join(configs['data_dir'], args.dataset.lower())
		configs['finetune_model_dir'] = os.path.join(configs['finetune_model_dir'], args.dataset.lower())
		configs['output_dir'] = os.path.join(configs['output_dir'], args.dataset.lower())
		check_dir(configs['data_dir'])
		check_dir(configs['finetune_model_dir'])
		check_dir(configs['output_dir'])

		processor = processor_class()
		for model_class in configs['model_class']:
			print('Begin Evaluate %s Model on corpus %s' %(model_class, args.dataset))
			predicter = Predictor(configs, model_class, processor)
			predicter.eval()

if __name__ == '__main__':
	main()