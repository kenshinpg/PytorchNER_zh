
# coding: utf-8

import os
import json
import numpy as np
import pickle

def check_dir(dir_path):
	if not os.path.exists(dir_path):
		os.makedirs(dir_path, exist_ok = True)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def json_dump(dict_):
	return json.dumps(dict_, ensure_ascii=False, cls=MyEncoder)

def json_load(file_path, encoding = 'utf-8'):
    with open(file_path, encoding = encoding) as json_file:
        return json.load(json_file)

def json_write(file, file_path, encoding = 'utf-8'):
    with open(file_path, 'w', encoding = encoding) as json_file:
        json_file.write(json.dumps(file))

def pickle_save(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

def pickle_load(file_path):
    return pickle.load(open(file_path, 'rb'))