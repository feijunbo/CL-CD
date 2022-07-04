import json
pid2name = {}
with open('property_list.html', 'r', encoding='utf-8') as fin, open('pid2name.json', 'w', encoding='utf-8') as fout:
    cnt = 0
    for line in fin:
        if cnt != 0:
            pid2name[pid].append(line.strip()[4:-5])
            cnt -= 1
        if 'data-sort-value' in line:
            cnt = 2
            split_line = line.split('</a></td>')[0]
            pid = split_line.split('>')[-1]
            pid2name[pid] = []
    fout.write(json.dumps(pid2name))