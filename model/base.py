
# coding: utf-8

import os

import torch
import torch.nn as nn

from utils.datautils import json_write, json_load

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
TF_WEIGHTS_NAME = 'model.ckpt'

class BaseModel(nn.Module):

	def save_pretrained(self, save_directory):

		assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

		# Only save the model it-self if we are using distributed training
		model_to_save = self.module if hasattr(self, 'module') else self
		# Save configuration file
		json_write(model_to_save.configs, os.path.join(save_directory, CONFIG_NAME))
		# If we save using the predefined names, we can load using `from_pretrained`
		output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
		torch.save(model_to_save.state_dict(), output_model_file)

	@classmethod
	def from_pretrained(cls, pretrained_model_path, pretrained_word_embed = None):

		configs = json_load(os.path.join(pretrained_model_path, CONFIG_NAME))
		model = cls(configs, pretrained_word_embed = pretrained_word_embed)
		archive_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
		state_dict = torch.load(archive_file, map_location='cpu')        
		# Load from a PyTorch state_dict
		missing_keys = []
		unexpected_keys = []
		error_msgs = []       
		# copy state_dict so _load_from_state_dict can modify it
		metadata = getattr(state_dict, '_metadata', None)
		state_dict = state_dict.copy()
		if metadata is not None:
		    state_dict._metadata = metadata
		def load(module, prefix=''):
		    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
		    module._load_from_state_dict(
		        state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
		    for name, child in module._modules.items():
		        if child is not None:
		            load(child, prefix + name + '.')
		load(model)
		# Set model in evaluation mode to desactivate DropOut modules by default
		model.eval()
		return model
