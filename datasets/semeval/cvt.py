import json


filenames = ['TEST_FILE_FULL.TXT', 'TRAIN_FILE.TXT']

data_json = {}
with open('semeval.txt', 'w', encoding='utf-8') as fout:
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as fin:
            for idx, line, in enumerate(fin):
                if idx % 4 == 0:
                    text = line.strip().split('\t')[1][1:-1]
                elif idx % 4 == 1:
                    relation = line.strip().split('(')[0]
                    if relation == 'Other':
                        continue
                    order = line.strip().split('(')[1][:2]
                elif idx % 4 == 2:
                    pass
                elif idx % 4 == 3:
                    if relation == 'Other':
                        continue
                    if relation not in data_json:
                        data_json[relation] = []
                    if order == 'e2':
                        text = text.replace('<e1>', '[unused0] ')
                        text = text.replace('</e1>', ' [unused2]')
                        text = text.replace('<e2>', '[unused1] ')
                        text = text.replace('</e2>', ' [unused3]')
                    elif order == 'e1':
                        text = text.replace('<e2>', '[unused0] ')
                        text = text.replace('</e2>', ' [unused2]')
                        text = text.replace('<e1>', '[unused1] ')
                        text = text.replace('</e1>', ' [unused3]')
                    data_json[relation].append(text)
            json.dump(data_json, fout)
    
