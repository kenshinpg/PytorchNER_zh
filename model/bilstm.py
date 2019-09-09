
# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from model.crf import CRF
from model.base import BaseModel
# from pytorch_transformers.modeling_utils import PretrainedConfig

# class BiLSTMConfig(PretrainedConfig):

class BiLSTM(BaseModel):

    def __init__(self, configs, pretrained_word_embed = None):
        super(BiLSTM, self).__init__()

        self.configs = configs
        self.num_labels = configs['num_labels']
        self.max_seq_length = configs['max_seq_length']
        self.hidden_dim = configs['rnn_hidden']
        self.use_cuda = configs['use_cuda'] and torch.cuda.is_available()		

        # word embedding layer
        self.word_embedding = nn.Embedding(num_embeddings = configs['word_vocab_size'], 
        									embedding_dim = configs['word_embedding_dim'])
        if configs['use_pretrained_embedding']:
        	self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_word_embed))
        	self.word_embedding.weight.requires_grad = configs['requires_grad']
        # dropout layer
        self.dropout_embed = nn.Dropout(configs['dropout_rate'])
        self.dropout_rnn = nn.Dropout(configs['dropout_rate'])
        # rnn layer
        self.rnn_layers = configs['rnn_layers']
        self.lstm = nn.LSTM(input_size = configs['word_embedding_dim'], # bert embedding
                            hidden_size = self.hidden_dim,
                            num_layers = self.rnn_layers, 
                            batch_first = True,                             
                            bidirectional = True)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.num_labels)
        self.loss_function = CrossEntropyLoss()

    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        双向是2，单向是1
        """
        if self.use_cuda:
            return (torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim).cuda(), 
                torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim), 
                torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim))

    def get_lstm_outputs(self, input_ids):

        word_embeds = self.word_embedding(input_ids)
        word_embeds = self.dropout_embed(word_embeds)

        batch_size = input_ids.size(0)
        hidden = self.rand_init_hidden(batch_size)    	

        lstm_outputs, hidden = self.lstm(word_embeds, hidden)
        lstm_outputs = lstm_outputs.contiguous().view(-1, self.hidden_dim * 2)
        return lstm_outputs

    def forward(self, input_ids, segment_ids, input_mask):
        lstm_outputs = self.get_lstm_outputs(input_ids)        
        logits = self.hidden2label(lstm_outputs)
        return logits.view(-1, self.max_seq_length, self.num_labels)

    def loss_fn(self, feats, mask, labels):
        loss_value = self.loss_function(feats.view(-1, self.num_labels), labels.view(-1))
        return loss_value

    def predict(self, feats, mask = None):
        return feats.argmax(-1)


class BiLSTMCRF(BaseModel):

    def __init__(self, configs, pretrained_word_embed = None):
        super(BiLSTMCRF, self).__init__()

        self.configs = configs
        self.num_labels = configs['num_labels']
        self.max_seq_length = configs['max_seq_length']
        self.use_cuda = configs['use_cuda'] and torch.cuda.is_available()

        self.bilstm = BiLSTM(configs, pretrained_word_embed)
        self.crf = CRF(target_size = self.num_labels,
                       use_cuda = self.use_cuda,
                       average_batch = False)
        self.hidden2label = nn.Linear(self.bilstm.hidden_dim * 2, self.num_labels + 2)

    def forward(self, input_ids, segment_ids, input_mask):
        lstm_outputs = self.bilstm.get_lstm_outputs(input_ids)
        logits = self.hidden2label(lstm_outputs)
        return logits.view(-1, self.max_seq_length, self.num_labels + 2)

    def loss_fn(self, feats, mask, labels):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        batch_size = feats.size(0)
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, labels)/float(batch_size)
        return loss_value

    def predict(self, feats, mask):

        path_score, best_path = self.crf(feats, mask.byte())
        return best_path