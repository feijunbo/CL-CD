# -*- encoding: utf-8 -*-

import argparse
import random


import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
import os
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
from sklearn.cluster import KMeans
from scipy.stats import mode
# 基本参数
EPOCHS = 1
# SAMPLES = 10000
BATCH_SIZE = 32
LR = 1e-5
DROPOUT = 0.3
MAXLEN = 128
POOLING = 'cls'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# 预训练模型目录
BERT = '/home/feijunbo/bert-base-uncased'
# BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
# ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
model_path = BERT 

# 微调后参数存放位置
UNSUP_SAVE_PATH = './saved_model/zerorc_unsup.pt'
SUP_SAVE_PATH = './saved_model/zerorc_sup.pt'

# 数据目录
# BASE_PATH = './datasets/fewrel'
# DATA_NAME = 'wiki'
FEWREL_TRAIN = './datasets/fewrel/train.txt'
FEWREL_VAL = './datasets/fewrel/val.txt'
FEWREL_LABEL = './datasets/fewrel/pid2name.json'

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

def load_data(path):
    """根据名字加载不同的数据集"""
    sents = []

    if os.path.exists(path):
        json_data = json.load(open(path))
        for key in json_data:
            for item in json_data[key]:
                sent = cvt_data(item['tokens'], item['h'][2][0], item['t'][2][0])
                sents.append(sent)
            if os.path.exists(FEWREL_LABEL):
                pid2name = json.load(open(FEWREL_LABEL))
                # print(','.join(pid2name[key]))
                sents.append(','.join(pid2name[key]))

    assert len(sents) != 0

    return sents

def load_test(path):
    sents = []
    labels = []
    label_sents = []
    if os.path.exists(path):
        json_data = json.load(open(path))
        index = 0
        for key in list(json_data.keys()):
            for item in json_data[key]:
                sent = cvt_data(item['tokens'], item['h'][2][0], item['t'][2][0])
                sents.append(sent)
                labels.append(index)
            index += 1
            if os.path.exists(FEWREL_LABEL):
                pid2name = json.load(open(FEWREL_LABEL))
                label_sents.append(','.join(pid2name[key]))

    return sents, labels, label_sents

class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data):
        self.data = data
      
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text):
        # 添加自身两次, 经过bert编码之后, 互为正样本
        return tokenizer([text, text], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index):
        return self.text_2_id(self.data[index])


class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        da = self.data[index]
        la = self.labels[index]
        return self.text_2_id([da]), int(la)

class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""
    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)       
        config.attention_probs_dropout_prob = DROPOUT   # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT           
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        
        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 768]
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
    
    
def simcse_unsup_loss(y_pred):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss

# def cluster_pred(model, dataloader) -> float:
#     """模型评估函数 
#     批量预测, batch结果拼接, 一次性求spearman相关度
#     """
#     model.eval()
#     cluster_tensor = torch.tensor([], device=DEVICE)
#     label_array = np.array([])
#     with torch.no_grad():
#         for source, label in tqdm(dataloader):
#             # source        [batch, 1, seq_len] -> [batch, seq_len]
#             source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
#             source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
#             source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
#             source_tensor = model(source_input_ids, source_attention_mask, source_token_type_ids)

#             cluster_tensor = torch.cat((cluster_tensor, source_tensor), dim=0)
#             label_array = np.append(label_array, np.array(label))        
#     # corrcoef 
#     return cluster_tensor, label_array
def eval(model, dataloader, label_sents):
    """模型评估函数 
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    target = tokenizer(label_sents, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        for source, label in tqdm(dataloader):
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            source_pred = source_pred.unsqueeze(1)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(DEVICE)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(DEVICE)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)            
            label_array = np.append(label_array, np.array(label))
    model.train()
    pred_array = sim_tensor.argmax(-1).cpu().numpy()
    f1 = f1_score(label_array, pred_array, average='macro')
    p = precision_score(label_array, pred_array, average='macro')
    r = recall_score(label_array, pred_array, average='macro')
    return sim_tensor, f1, p, r

def kmeans(model, dataloader, label_sents):
    way = len(label_sents)
    embedding_array = []
    label_array = []
    model.eval()
    target = tokenizer(label_sents, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        for source, label in tqdm(dataloader):
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)

            embedding_array.append(source_pred.cpu().numpy())
            label_array.append(np.array(label))

        # target        [batch, 1, seq_len] -> [batch, seq_len]
        # target_input_ids = target.get('input_ids').squeeze(1).to(DEVICE)
        # target_attention_mask = target.get('attention_mask').squeeze(1).to(DEVICE)
        # target_token_type_ids = target.get('token_type_ids').squeeze(1).to(DEVICE)
        # target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
        # embedding_array.append(target_pred.cpu().numpy())
        # label_array.append(np.arange(100, 100+len(label_sents)))

        embedding_array = np.concatenate(embedding_array, 0)
        label_array = np.concatenate(label_array, 0)

    # model.train()
    kmeans = KMeans(n_clusters=way)
    kmeans.fit(embedding_array)
    y_kmeans = kmeans.predict(embedding_array)
    tsne = TSNE()
    X_embedded = tsne.fit_transform(embedding_array)
    palette = sns.color_palette("bright", len(np.unique(label_array)))
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_kmeans, legend='full', palette=palette)
    plt.savefig(f'tsne_kmeans_{way}.png')

    pred_array = np.zeros_like(y_kmeans)
    for i in range(way):
        #得到聚类结果第i类的 True Flase 类型的index矩阵
        mask = (y_kmeans == i)
        #根据index矩阵，找出这些target中的众数，作为真实的label
        pred_array[mask] = mode(label_array)[0]

    f1 = f1_score(label_array, pred_array, average='macro')
    p = precision_score(label_array, pred_array, average='macro')
    r = recall_score(label_array, pred_array, average='macro')
    return None, f1, p, r

def train(model, train_dl, dev_dl, optimizer, label_sents) -> None:
    """模型训练函数"""
    model.train()
    global best
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)

        out = model(input_ids, attention_mask, token_type_ids)        
        loss = simcse_unsup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx) % 100 == 0 or batch_idx == len(train_dl):    
            print(f'loss: {loss.item():.4f}')
            _, f1, p, r = kmeans(model, dev_dl, label_sents)
            model.train()
            if best < f1:
                best = f1
                torch.save(model.state_dict(), UNSUP_SAVE_PATH)
                print(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
       

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--seed', type=int, default=5546)
    
    args = parser.parse_args()
    set_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # load data
    train_data = load_data(FEWREL_TRAIN)
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE)
    sents, labels, label_sents = load_test(FEWREL_VAL)
    test_dataset = TestDataset(sents, labels)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # load model
    print(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # train
    if args.do_train:
        # model.load_state_dict(torch.load(SUP_SAVE_PATH))
        best = 0
        for epoch in range(EPOCHS):
            print(f'epoch: {epoch}')
            train(model, train_dataloader, test_dataloader, optimizer, label_sents)
    # eval
    if args.do_predict:
        model.load_state_dict(torch.load(UNSUP_SAVE_PATH))
        sim_tensor, f1, p, r = eval(model, test_dataloader, label_sents)
        print(f'dev_macroF1: {f1:.4f}')

        sim_tensor, f1, p, r  = kmeans(model, test_dataloader, label_sents)
        print(f'kmeans test_macroF1: {f1:.4f}, p: {p}, r: {r}')