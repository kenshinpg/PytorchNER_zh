
# coding: utf-8

import os

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
# %matplotlib inline
sns.set_style("whitegrid")

def eval_plot(configs, train_loss_log, dev_loss_log):

	plt.figure(figsize=(12,7))
	epochs = list(range(len(train_loss_log)))

	plt.plot(epochs, train_loss_log, color = 'r')
	plt.plot(epochs, dev_loss_log, color = 'b')
	plt.xlabel('Epochs', fontsize = 15)
	plt.ylabel('Loss', fontsize = 15)
	plt.savefig(os.path.join(configs['test']['output_dir'], 'loss_eval.png'))