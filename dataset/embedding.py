
# coding: utf-8

import os
import numpy as np
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from configs.confighelper import config_loader
from utils.datautils import pickle_load, pickle_save

def glove_to_gensim(glove_embed, gensim_embed):
	"""
	将glove的embedding转化为用gensim可读取的格式
	glove_embed: stanford官网下载的glove.6B.txt
	new_word2vec_embed: 可用gensim读取的embedding_file
	"""
	# 输入文件
	glove_file = datapath(glove_embed)
	# 输出文件
	tmp_file = get_tmpfile(gensim_embed)
	# 开始转换
	glove2word2vec(glove_file, tmp_file)

	# # 加载转化后的文件
	# model = KeyedVectors.load_word2vec_format(tmp_file)

def load_embed_with_gensim(path_embed, binary = False):
	"""
	读取预训练的embedding
	binary = True 二进制embedding
	"""	
	return KeyedVectors.load_word2vec_format(path_embed, binary=binary)

def get_pretrained_embedding(pretrain_embed_file, pretrain_embed_pkl):
	if os.path.exists(pretrain_embed_pkl):
		word_vectors = pickle_load(pretrain_embed_pkl)
	else:
		word_vectors = load_embed_with_gensim(pretrain_embed_file)
		pickle_save(word_vectors, pretrain_embed_pkl)
	return word_vectors

# def get_stoi_from_tokenizer(tokenizer):
#     """
#     从BertTokenizer得到word2id_dict
#     """


#     return word2id_dict

def build_word_embed(tokenizer, pretrain_embed_file, pretrain_embed_pkl, seed=1301):
    """
    从预训练的文件中构建word embedding表
    Args:
        tokenizer: BertTokenizer
    Returns:
        word_embed_table: np.array, shape=[word_count, embed_dim]
        match_count: int, 匹配的词数
        unknown_count: int, 未匹配的词数
    """
    word_vectors = get_pretrained_embedding(pretrain_embed_file, pretrain_embed_pkl)
    word_dim = word_vectors.vector_size
    word_count = tokenizer.vocab_size  # 0 is for padding value
    np.random.seed(seed)
    scope = np.sqrt(3. / word_dim)
    word_embed_table = np.random.uniform(
        -scope, scope, size=(word_count, word_dim)).astype('float32')
    # match_count, unknown_count = 0, 0
    # for word in word2id_dict:
    #     if word in word_vectors.vocab:
    #         word_embed_table[word2id_dict[word]] = word_vectors[word]
    #         match_count += 1
    #     else:
    #         unknown_count += 1
    # total_count = match_count + unknown_count
    # print('\tmatch: {0} / {1}'.format(match_count, total_count))
    # print('\tOOV: {0} / {1}'.format(unknown_count, total_count))
    return word_embed_table
