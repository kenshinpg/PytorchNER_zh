
# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from model.crf import CRF
from configs.confighelper import config_loader

class Bert(BertPreTrainedModel):

    def __init__(self, config):
        super(Bert, self).__init__(config)
        configs = config_loader('configs/config.yml')
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.hidden_dim = config.hidden_size
        self.use_cuda = configs['use_cuda'] and torch.cuda.is_available()
        self.dropout = nn.Dropout(configs['dropout_rate'])
        self.hidden2label = nn.Linear(self.hidden_dim, self.num_labels)
        self.loss_function = CrossEntropyLoss()
        self.max_seq_length = configs['max_seq_length']

        self.apply(self.init_weights)

    def forward(self, input_ids, segment_ids, input_mask):

        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output = encoder_outputs[0]
        # pooled_output = pooler(sequence_output)
        outputs = self.bert(input_ids = input_ids, 
                            position_ids = None, 
                            token_type_ids = segment_ids,
                            attention_mask = input_mask, 
                            head_mask = None)
        # bert_embeds: shape = [batch_size, max_seq_length, bert_embedding]
        bert_embeds = outputs[0].contiguous().view(-1, self.hidden_dim)

        bert_embeds = self.dropout(bert_embeds)
        logits = self.hidden2label(bert_embeds)

        return logits.view(-1, self.max_seq_length, self.num_labels)

    def loss_fn(self, feats, mask, labels):
        """
        feats: size=(batch_size, max_seq_length, num_labels)
        mask: size=(batch_size, max_seq_length)
        labels: size=(batch_size, max_seq_length)
        """

        # Only keep active parts of the loss
        if mask is not None:
            active_loss = mask.view(-1) == 1 # size=(batch_size * max_seq_length) 
            active_logits = feats.view(-1, self.num_labels)[active_loss] # size=(batch_size * max_seq_length, num_labels)
            active_labels = labels.view(-1)[active_loss]
            loss_value = self.loss_function(active_logits, active_labels)
        else:
            loss_value = self.loss_function(feats.view(-1, self.num_labels), labels.view(-1))             
        return loss_value

    def predict(self, feats, mask = None):
        return feats.argmax(-1)

class BertCRF(BertPreTrainedModel):

    def __init__(self, config):
        super(BertCRF, self).__init__(config)
        configs = config_loader('configs/config.yml')
        self.num_labels = config.num_labels
        self.max_seq_length = configs['max_seq_length']
        self.bert = BertModel(config)
        self.use_cuda = configs['use_cuda'] and torch.cuda.is_available()
        self.crf = CRF(target_size = self.num_labels,
                       use_cuda = self.use_cuda,
                       average_batch = False)
        bert_embedding = config.hidden_size
        # hidden_dim即输出维度
        # lstm的hidden_dim和init_hidden的hidden_dim是一致的
        # 是输出层hidden_dim的1/2
        self.hidden_dim = config.hidden_size
        self.dropout = nn.Dropout(configs['dropout_rate'])
        self.hidden2label = nn.Linear(self.hidden_dim, self.num_labels + 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, segment_ids, input_mask):

        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output = encoder_outputs[0]
        # pooled_output = pooler(sequence_output)
        outputs = self.bert(input_ids = input_ids, 
                            position_ids = None, 
                            token_type_ids = segment_ids,
                            attention_mask = input_mask, 
                            head_mask = None)
        # bert_embeds: shape = [batch_size, max_seq_length, bert_embedding]
        bert_embeds = outputs[0].contiguous().view(-1, self.hidden_dim)
        bert_embeds = self.dropout(bert_embeds)
        logits = self.hidden2label(bert_embeds)

        return logits.view(-1, self.max_seq_length, self.num_labels +2)

    def loss_fn(self, feats, mask, labels):
        batch_size = feats.size(0)
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, labels)/float(batch_size)        
        return loss_value

    def predict(self, feats, mask):

        path_score, best_path = self.crf(feats, mask.byte())
        return best_path

class BertBiLSTMCRF(BertPreTrainedModel):

    def __init__(self, config):
        super(BertBiLSTMCRF, self).__init__(config)
        configs = config_loader('configs/config.yml')
        self.num_labels = config.num_labels
        self.max_seq_length = configs['max_seq_length']
        self.bert = BertModel(config)
        self.use_cuda = configs['use_cuda'] and torch.cuda.is_available()
        self.crf = CRF(target_size = self.num_labels,
                       use_cuda = self.use_cuda,
                       average_batch = False)
        bert_embedding = config.hidden_size
        # hidden_dim即输出维度
        # lstm的hidden_dim和init_hidden的hidden_dim是一致的
        # 是输出层hidden_dim的1/2
        self.hidden_dim = config.hidden_size
        self.rnn_layers = configs['rnn_layers']
        self.lstm = nn.LSTM(input_size = bert_embedding, # bert embedding
                            hidden_size = self.hidden_dim,
                            num_layers = self.rnn_layers, 
                            batch_first = True,                             
                            # dropout = configs['train']['dropout_rate'],
                            bidirectional = True)
        self.dropout = nn.Dropout(configs['dropout_rate'])
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.num_labels + 2)
        self.apply(self.init_weights)

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

    def forward(self, input_ids, segment_ids, input_mask):

        # outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output = encoder_outputs[0]
        # pooled_output = pooler(sequence_output)
        outputs = self.bert(input_ids = input_ids, 
                            position_ids = None, 
                            token_type_ids = segment_ids,
                            attention_mask = input_mask, 
                            head_mask = None)
        # bert_embeds: shape = [batch_size, max_seq_length, bert_embedding]
        bert_embeds = outputs[0]

        batch_size = input_ids.size(0)

        hidden = self.rand_init_hidden(batch_size)

        lstm_output, hidden = self.lstm(bert_embeds, hidden)
        lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim * 2)
        # lstm_output = self.dropout(lstm_output)
        logits = self.hidden2label(lstm_output)

        return logits.view(-1, self.max_seq_length, self.num_labels +2)

    def loss_fn(self, feats, mask, labels):
        batch_size = feats.size(0)
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, labels)/float(batch_size)        
        return loss_value 

    def predict(self, feats, mask):

        path_score, best_path = self.crf(feats, mask.byte())
        return best_path