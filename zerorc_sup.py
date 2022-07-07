# -*- encoding: utf-8 -*-

import argparse
import random
import json
from typing import List
import os
from weakref import getweakrefcount
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
from sklearn.cluster import KMeans
from scipy.stats import mode

# 基本参数
EPOCHS = 2
# SAMPLES = 10000
BATCH_SIZE = 32
LR = 1e-5
DROPOUT = 0.3
MAXLEN = 128
POOLING = 'cls'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') 

# 预训练模型目录
BERT = '/home/feijunbo/bert-base-uncased'
# BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
# ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
model_path = BERT 

# 微调后参数存放位置
SAVE_PATH = './saved_model/zerorc_sup.pt'

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
    """根据名字加载不同的数据集
    """
      
    with open(path, 'r') as f:
        return [(json.loads(line)['origin'], json.loads(line)['entailment'], json.loads(line)['contradiction']) for line in f]

def load_test(path, label_path):
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
            if os.path.exists(label_path):
                pid2name = json.load(open(label_path))
                label_sents.append(','.join(pid2name[key]))

    return sents, labels, label_sents   

class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text):
        return tokenizer([text[0], text[1], text[2]], max_length=MAXLEN, 
                         truncation=True, padding='max_length', return_tensors='pt')
    
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
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        # config = BertConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        
        # out = self.bert(input_ids, attention_mask, token_type_ids)
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
                  
            
def simcse_sup_loss(y_pred):
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]
    
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    use_row = torch.where((y_true + 1) % 3 != 0)[0]
    y_true = (use_row - use_row % 3 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss
    

def eval(model, dataloader, label_sents):
    """模型评估函数 
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_array = torch.tensor([], device=DEVICE)
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
            sim_array = torch.cat((sim_array, sim), dim=0)            
            label_array = np.append(label_array, np.array(label))

    pred_array = sim_array.argmax(-1).cpu().numpy()
    f1 = f1_score(label_array, pred_array, average='macro', average='macro', zero_division=0)
    p = precision_score(label_array, pred_array, average='macro', average='macro', zero_division=0)
    r = recall_score(label_array, pred_array, average='macro', average='macro', zero_division=0)
    return sim_array.cpu().numpy(), f1, p, r

def tsne(model, dataloader, label_sents):
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
        target_input_ids = target.get('input_ids').squeeze(1).to(DEVICE)
        target_attention_mask = target.get('attention_mask').squeeze(1).to(DEVICE)
        target_token_type_ids = target.get('token_type_ids').squeeze(1).to(DEVICE)
        target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
        embedding_array.append(target_pred.cpu().numpy())
        label_array.append(np.arange(100, 100+len(label_sents)))

        embedding_array = np.concatenate(embedding_array, 0)
        label_array = np.concatenate(label_array, 0)

    # model.train()
    tsne = TSNE()
    X_embedded = tsne.fit_transform(embedding_array)
    palette = sns.color_palette("bright", len(np.unique(label_array)))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=label_array, legend='full', palette=palette)
    plt.savefig(f'tsne_{way}.png')
    plt.close()


def kmeans(model, dataloader, label_sents):
    way = len(label_sents)
    embedding_array = []
    label_array = []
    model.eval()
    # target = tokenizer(label_sents, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
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
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y_kmeans, legend='full', palette=palette)
    plt.savefig(f'tsne_kmeans_{way}.png')
    plt.close()

    pred_array = np.zeros_like(y_kmeans)
    for i in range(way):
        #得到聚类结果第i类的 True Flase 类型的index矩阵
        mask = (y_kmeans == i)
        #根据index矩阵，找出这些target中的众数，作为真实的label
        pred_array[mask] = mode(label_array[mask])[0]

    f1 = f1_score(label_array, pred_array, average='macro', zero_division=0)
    p = precision_score(label_array, pred_array, average='macro', zero_division=0)
    r = recall_score(label_array, pred_array, average='macro', zero_division=0)
    return None, f1, p, r

def get_kmeans_sim(model, dataloader, label_sents):
    way = len(label_sents)

    sim_array = []
    embedding_array = []
    label_array = []
    model.eval()

    with torch.no_grad():
        for source, label in tqdm(dataloader):
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            
            embedding_array.append(source_pred.cpu().numpy())
            label_array.append(np.array(label))

        embedding_array = np.concatenate(embedding_array, 0)
        label_array = np.concatenate(label_array, 0)


    kmeans = KMeans(n_clusters=way)
    results = kmeans.fit(embedding_array)
    for embedding in embedding_array:
        sim_array.append(cosine_similarity(embedding, results.cluster_centers_))
    sim_array = np.concatenate(sim_array, 0)

    pred_array = sim_array.argmax(-1)
    f1 = f1_score(label_array, pred_array, average='macro', average='macro', zero_division=0)
    p = precision_score(label_array, pred_array, average='macro', average='macro', zero_division=0)
    r = recall_score(label_array, pred_array, average='macro', average='macro', zero_division=0)
    return sim_array, f1, p, r

def train(model, train_dl, test_dl, optimizer, label_sents):
    """模型训练函数 
    """
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(DEVICE)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_sup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 200 == 0 or batch_idx == len(train_dl):
            print(f'loss: {loss.item():.4f}')
            _, f1, p, r = eval(model, test_dl, label_sents)
            model.train()
            if best < f1:
                early_stop_batch = 0
                best = f1
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"higher macroF1: {best:.4f} in batch: {batch_idx}, save model")
                continue
            else:
                print(f"macroF1: {f1:.4f} in batch: {batch_idx}")
            early_stop_batch += 1
            if early_stop_batch == 10:
                print(f"macroF1 doesn't improve for {early_stop_batch} batch, early stop!")
                print(f"train use sample number: {(batch_idx - 10) * BATCH_SIZE}")
                return
 
def get_similar_sents(sim_array, test_sents, test_labels, topk):
    way = sim_array.shape[1]

    pred_array = sim_array
    
    selected_sents = {}
    selected_labels = []
    selected_preds = []
    for i in range(way):
        tmp_index = np.where(pred_array == i)[0]
        temp_tensor = sim_array[:,i][pred_array == i].cpu().numpy()
        sorted_index = np.argsort(-temp_tensor)
        sorted_index = tmp_index[sorted_index]
        selected_sents[i] = []
        # print(sim_tensor[:,i][sorted_index])
        for j in range(topk):
            selected_sents[i].append(test_sents[sorted_index[j]])
            selected_labels.append(test_labels[sorted_index[j]])
            selected_preds.append(pred_array[sorted_index[j]])
    f1score = f1_score(selected_labels[i], selected_preds[i], average='macro', zero_division=0)
    print(f"top {topk}", f"f1 {f1score}")
    return selected_sents, selected_labels

def get_intersection(a_selected_sents, a_selected_labels, b_selected_sents, b_selected_labels):
    selected_sents = {}
    selected_labels = []
    selected_preds = []

    for k, a_v in a_selected_sents:
        selected_sents[i] = []
        a_l = a_selected_labels[k]
        b_v = b_selected_sents[k]
        for i, v in enumerate(a_v):
            if v in b_v:
                selected_sents[i].append(v)
                selected_labels.append(a_l)
                selected_preds.append(k)
    f1score = f1_score(selected_labels[i], selected_preds[i], average='macro', zero_division=0)
    print(f"num {len(selected_labels)}", f"f1 {f1score}")
    return selected_sents, selected_labels

def get_pesudo_data(selected_sents, label_sents):
    pesudo_data = []
    for i, label_sent in enumerate(label_sents):
        for _ in range(50):
            sent1, sent_e1 = np.random.choice(selected_sents[i], 2, replace=False)
            sent_e2 = label_sent
            cont_keys = np.random.choice(list(set(range(len(label_sents))) - set([i])), 1, replace=False)
            for cont_key in cont_keys:
                snet_c1 = np.random.choice(selected_sents[cont_key], 1, replace=False)[0]
                snet_c2 = label_sents[cont_key]
                pesudo_data.append([sent1, sent_e1, snet_c1])
                pesudo_data.append([sent1, sent_e2, snet_c2])
    return pesudo_data

def fusion_similarity(a_sim, b_sim, label_array):
    sim_array = (a_sim + b_sim) / 2
    pred_array = sim_array.argmax(-1)
    f1 = f1_score(label_array, pred_array, average='macro', average='macro', zero_division=0)
    p = precision_score(label_array, pred_array, average='macro', average='macro', zero_division=0)
    r = recall_score(label_array, pred_array, average='macro', average='macro', zero_division=0)

    return sim_array, f1, p, r

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    
    print(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--seed', type=int, default=5546)
    parser.add_argument('--dataset', type=str, default='fewrel', choices=['wikizsl', 'fewrel'])
    
    args = parser.parse_args()

    if args.dataset == 'fewrel':
        DATA_TRAIN = './datasets/fewrel/train.txt'
        DATA_VAL = './datasets/fewrel/val.txt'
        DATA_LABEL = './datasets/fewrel/pid2name.json'
    else:
        DATA_TRAIN = './datasets/wikizsl/train.txt'
        DATA_VAL = './datasets/wikizsl/val.txt'
        DATA_LABEL = './datasets/wikizsl/pid2name.json'
    set_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # load data
    train_data = load_data(DATA_TRAIN)
    random.shuffle(train_data)
    test_sents, test_labels, label_sents = load_test(DATA_VAL, DATA_LABEL)
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(TestDataset(test_sents, test_labels), batch_size=BATCH_SIZE)
    # load model    
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train
    if args.do_train:
        best = 0
        for epoch in range(EPOCHS):
            print(f'epoch: {epoch}')
            train(model, train_dataloader, test_dataloader, optimizer, label_sents)
        print(f'train is finished, best model is saved at {SAVE_PATH}')
    # eval
    if args.do_predict:
        model.load_state_dict(torch.load(SAVE_PATH))
        label_sim_array, f1, p, r = eval(model, test_dataloader, label_sents)
        print(f'test_macroF1: {f1:.4f}, p: {p}, r: {r}')

        _, f1, p, r  = kmeans(model, test_dataloader, label_sents)
        print(f'kmeans test_macroF1: {f1:.4f}, p: {p}, r: {r}')
        kmeans_sim_array, f1, p, r = get_kmeans_sim(model, test_dataloader, label_sents)
        print(f'kmeans similarity test_macroF1: {f1:.4f}, p: {p}, r: {r}')
        # tsne(model, test_dataloader, label_sents)

        fusion_sim_array, f1, p, r = fusion_similarity(label_sim_array, kmeans_sim_array, test_labels)
        print(f'fusion test_macroF1: {f1:.4f}, p: {p}, r: {r}')

        label_selected_sents, label_selected_labels = get_similar_sents(label_sim_array, test_sents, test_labels, 100)
        kmeans_selected_sents, kmeans_selected_labels = get_similar_sents(kmeans_sim_array, test_sents, test_labels, 100)
        selected_sents, selected_labels = get_intersection(label_selected_sents, label_selected_labels, kmeans_selected_sents, kmeans_selected_labels)
        pesudo_data = get_pesudo_data(selected_sents, label_sents)
        pesudo_dataloader = DataLoader(TrainDataset(pesudo_data), batch_size=BATCH_SIZE)
        best = 0
        for epoch in range(EPOCHS):
            print(f'epoch: {epoch}')
            train(model, pesudo_dataloader, test_dataloader, optimizer, label_sents)
        print(f'train is finished, best model is saved at {SAVE_PATH}')
        pesudo_sim_array, f1, p, r = eval(model, test_dataloader, label_sents)
        print(f'test_macroF1: {f1:.4f}, p: {p}, r: {r}')

        _, f1, p, r  = kmeans(model, test_dataloader, label_sents)
        print(f'kmeans test_macroF1: {f1:.4f}, p: {p}, r: {r}')