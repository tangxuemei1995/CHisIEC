
import random
f = open('./train/train_ner_re_alpaca.json', 'w', encoding='utf-8')
all = []
for line in open('./train_ner_alpaca.json'):
    line = line.strip()
    all.append(line)
for line in open('./train_re_alpaca.json'):
    line = line.strip()
    all.append(line)
random.shuffle(all)
for x in all:
    f.write(x + '\n')
    
