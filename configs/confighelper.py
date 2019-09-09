
# coding: utf-8

import os
import yaml
import codecs
import argparse

def read_yml(yml_file):
	return yaml.load(codecs.open(yml_file, encoding = 'utf-8'))

def config_loader(config_file = './configs/cner.yml'):
	return read_yml(config_file)

# def config_update(args, configs):



def args_parser():
    # start parser 
    parser = argparse.ArgumentParser()
    # required parameters 
    parser.add_argument("--config_path", default='./configs/config.yml', type=str)
    parser.add_argument("--dataset", default='CCKS2019', type=str, help="dataset name")
    parser.add_argument("--task", default='eval', type=str, help = "task type, train/eval/conll")
    # parser.add_argument("--output_dir", default=None, 
    #     type=str, required=True, help="the outptu directory where the model predictions and checkpoints will")

    # # other parameters 
    # parser.add_argument("--use_cuda", type=bool, default=True)
    # parser.add_argument("--max_len_limit", default=100, 
    #     type=int, help="the maximum total input sequence length after ")

    # parser.add_argument("--hidden_dim", default=100, type=int)
    # parser.add_argument("--num_rnn_layers", default=1, type=int)
    # parser.add_argument("--bi_flag", default = True, type = bool)

    # parser.add_argument("--batch_size", default=32, type=int)
    # parser.add_argument("--average_batch", default=False, type=bool)
    # parser.add_argument("--optimizer", default='sgd', type=str)
    # parser.add_argument("--use_pretrained_embedding", default=True, type=bool)
    # parser.add_argument("--max_patience", default=50, type=int)

    # parser.add_argument("--test_batch_size", default=8, type=int)
    # parser.add_argument("--learning_rate", default=0.015, type=float)
    # parser.add_argument("--nb_epoch", default=1000, type=float)
    # parser.add_argument("--random_seed", type=int, default=1301)
    # parser.add_argument("--export_model", type=bool, default=True)
    # parser.add_argument("--output_dir", type=str, default='./output/')

    args = parser.parse_args()

    # os.makedirs(args.output_dir, exist_ok=True)

    return args
