# -*- encoding: utf-8 -*-

"""snli数据预处理"""
import argparse
import os
import time
import json
from tqdm import tqdm
import numpy as np

def timer(func):
    """ time-consuming decorator 
    """
    def wrapper(*args, **kwargs):
        ts = time.time()
        res = func(*args, **kwargs)
        te = time.time()
        print(f"function: `{func.__name__}` running time: {te - ts:.4f} secs")
        return res
    return wrapper

def cvt_data(raw_tokens, pos_head, pos_tail):
    # token -> index
    # tokens = ['[CLS]']
    tokens = []
    cur_pos = 0
    for token in raw_tokens:
        token = token.lower()
        if cur_pos == pos_head[0]:
            tokens.append('[unused0]')
        if cur_pos == pos_tail[0]:
            tokens.append('[unused1]')

        tokens.append(token)

        if cur_pos == pos_head[-1]:
            tokens.append('[unused2]')
        if cur_pos == pos_tail[-1]:
            tokens.append('[unused3]')
        cur_pos += 1

    return ' '.join(tokens)

@timer
def fewrel_preprocess(train_src_path, train_dst_path, val_src_path, val_dst_path, way):
    """处理原始的中文snli数据

    Args:
        src_path (str): 原始文件地址
        dst_path (str): 输出文件地址
    """

    pid2name = json.load(open('datasets/fewrel/pid2name.json'))
    # 组织数据
    with open(train_src_path, 'r', encoding='utf-8') as reader, open(train_dst_path, 'w', encoding='utf-8') as writer, \
         open(val_src_path, 'r', encoding='utf-8') as reader1, open(val_dst_path, 'w', encoding='utf-8') as writer1:
        json_data = json.load(reader)
        json_data = dict(json.load(reader1), **json_data)

        # random sample k unseen classes for test
        test_keys = np.random.choice(list(json_data.keys()), way, replace=False)
        test_values = [json_data[k] for k in test_keys]
        raw_test = dict(zip(test_keys, test_values))
        keys_else = set(json_data.keys()) - set(test_keys)
        values_else = [json_data[k] for k in keys_else]
        raw_train = dict(zip(keys_else, values_else))

        json_keys = list(raw_train.keys())
        for json_key in json_keys:
            for _ in range(200):
                item1, item2 = np.random.choice(raw_train[json_key], 2, replace=False)
                sent1 = cvt_data(item1['tokens'], item1['h'][2][0], item1['t'][2][0])
                sent_e1 = cvt_data(item2['tokens'], item2['h'][2][0], item2['t'][2][0])
                sent_e2 = ','.join(pid2name[json_key])
                cont_keys = np.random.choice(list(set(json_keys) - set([json_key])), 1, replace=False)
                for cont_key in cont_keys:
                    item1 = np.random.choice(raw_train[cont_key], 1, replace=False)[0]
                    snet_c1 = cvt_data(item1['tokens'], item1['h'][2][0], item1['t'][2][0])
                    snet_c2 = ','.join(pid2name[cont_key])
                    writer.write(json.dumps({'origin': sent1, 'entailment': sent_e1, 'contradiction': snet_c1}) + '\n')
                    writer.write(json.dumps({'origin': sent1, 'entailment': sent_e2, 'contradiction': snet_c2}) + '\n')
        
        for key in raw_test:
            if key in json_keys:
                print('ERROR!!!!!!!!!!!!!!!!')
        writer1.write(json.dumps(raw_test))

def set_seed(seed):
    np.random.seed(seed)

# 500 1 94
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--way', type=int, default=5)
    # 5245 
    parser.add_argument('--seed', type=int, default=5546)
    args = parser.parse_args()
    train_src, train_dst, val_src, val_dst = 'datasets/fewrel/train_wiki.json', 'datasets/fewrel/train.txt','datasets/fewrel/val_wiki.json', 'datasets/fewrel/val.txt'
    set_seed(args.seed)
    fewrel_preprocess(train_src, train_dst, val_src, val_dst, args.way)
 