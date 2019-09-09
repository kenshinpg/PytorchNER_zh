
# coding: utf-8
import os
import sys

from django.http import HttpResponse
import json
from utils.datautils import json_dump
from configs.confighelper import config_loader, args_parser
from dataset.preprocess import CCKS2019NER
from dataset.conll import conll_to_train_test_dev
from dataset.processor import CCKS2019Processor
from train.trainer import Trainer
from train.eval import Predictor
from utils.datautils import check_dir

dataset_name_to_class = {
  'ccks2019': (CCKS2019NER, CCKS2019Processor, './configs/ccks2019.yml')
}

def get_NER_result(request): 
    """
    request['data']: json
    example:[{"sentence":"入院后完善相关辅助检查，给予口服活血止痛、调节血压药物及物理治疗，患者血脂异常，补充诊断：混合性高脂血症，给予调节血脂药物治疗；患者诉心慌、无力，急查心电图提示：心房颤动，ST段改变。急请内科会诊，考虑为：1.冠心病 不稳定型心绞痛 心律失常 室性期前收缩 房性期前收缩 心房颤动；2.高血压病3级 极高危组。给予处理：1.急查心肌酶学、离子，定期复查心电图；2.给予持续心电、血压、血氧监测3.给予吸氧、西地兰0.2mg加5%葡萄糖注射液15ml稀释后缓慢静推，给予硝酸甘油10mg加入5%葡萄糖注射液500ml以5~10ugmin缓慢静点，继续口服阿司匹林100mg日一次，辛伐他汀20mg日一次，硝酸异山梨酯10mg日三次口服，稳心颗粒1袋日三次，美托洛尔12.5mg日二次，非洛地平5mg日一次治疗，患者病情好转出院。","model_class":["BertBiLSTMCRF"],"dataset": "CCKS2019"}]
    """
    if request.method == 'POST':
        json_data = json.loads(request.POST['data'], encoding = 'utf-8');
        sentence = json_data[0]['sentence']
        model_class = json_data[0]['model_class']
        dataset = json_data[0]['dataset'].lower()

        data_vocab_class, processor_class, conll_config_path = dataset_name_to_class[dataset]

        configs = config_loader('./configs/config.yml')
        configs['finetune_model_dir'] = os.path.join(configs['finetune_model_dir'], dataset)
        configs['output_dir'] = os.path.join(configs['output_dir'], dataset)

        result = {}

        processor = processor_class()
        for model_class in model_class:
          print('%s Model Outputs:')
          predicter = Predictor(configs, model_class, processor)
          entities_, result_ = predicter.predict_one(sentence)
          print(entities_)

          result[model_class] = result_

        output = json_dump(result)

        return HttpResponse(json.dumps({
          "status": 200,
          "errMsg": "",
          "data": output 
        }))
    else:
        return HttpResponse(json.dumps({
          "status": 400,
          "errMsg": "ValueError",
          "data": "" 
        }))    