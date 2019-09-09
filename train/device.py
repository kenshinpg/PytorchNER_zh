
# coding: utf-8

import torch

def check_cuda(configs):

	if configs['use_cuda'] and torch.cuda.is_available():
		device = torch.device("cuda", torch.cuda.current_device())
		use_gpu = True
	else:
		device = torch.device("cpu")
		use_gpu = False
	return device, use_gpu