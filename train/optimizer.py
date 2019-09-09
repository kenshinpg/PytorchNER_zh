
# coding: utf-8

from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import torch.optim as optim

def load_optimizer(configs, model):

    # Prepare optimizer
    optimizer_type = configs['optimizer_type']

    lr_decay = configs['lr_decay']
    learning_rate = configs['learning_rate']
    momentum = configs['momentum']
    l2_rate = configs['l2_rate']

    if optimizer_type.lower() == 'adamw':
    	param_optimizer = list(model.named_parameters())
    	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    	optimizer_grouped_parameters = [
    	    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': configs['weight_decay']},
    	    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    	]
    	optimizer = AdamW(optimizer_grouped_parameters, 
    						lr=configs['learning_rate'], 
    						correct_bias = False)
    else:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=l2_rate)
        elif optimizer_type.lower() == "adagrad":
            optimizer = optim.Adagrad(parameters, lr = learning_rate, weight_decay=l2_rate)
        elif optimizer_type.lower() == "adadelta":
            optimizer = optim.Adadelta(parameters, lr=learning_rate, weight_decay=l2_rate)
        elif optimizer_type.lower() == "rmsprop":
            optimizer = optim.RMSprop(parameters, lr=learning_rate, weight_decay=l2_rate)
        elif optimizer_type.lower() == "adam":
            optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=l2_rate)
        else:
            print('请选择正确的optimizer: {0}'.format(optimizer_type))

    # warmup_proportion：warm up 步数的比例，比如说总共学习100步，
    # warmup_proportion=0.1表示前10步用来warm up，warm up时以较低的学习率进行学习
    # (lr = global_step/num_warmup_steps * init_lr)，10步之后以正常(或衰减)的学习率来学习。
    schedular = WarmupLinearSchedule(optimizer, 
                            warmup_steps = int(configs['num_train_steps'] * configs['warmup_proportion']), 
    						t_total = configs['num_train_steps'])
    return optimizer, schedular