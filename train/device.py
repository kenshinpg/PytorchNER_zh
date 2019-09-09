
# coding: utf-8

import torch
import pynvml

def check_cuda(configs):

	if configs['use_cuda'] and torch.cuda.is_available():
		device = torch.device("cuda", torch.cuda.current_device())
		use_gpu = True
	else:
		device = torch.device("cpu")
		use_gpu = False
	return device, use_gpu

def gpu_memory_occupancy(configs):

	total_memory = configs['gpu_memory']
	pynvml.nvmlInit()
	# 这里的0是GPU id
	handle = pynvml.nvmlDeviceGetHandleByIndex(0)
	meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
	return '%.3fMiB//%dMiB' %(float(meminfo.used/1024/1024), total_memory)