with open('tmp', 'r') as f:
    lines = f.readlines()

import json

for line in lines:
    line_dic = json.loads(line.strip().replace('\'', '\"'))
    loss = line_dic['loss']
    print("%e"%(sum(loss)/len(loss)))
    print(len(loss))
