def re_for_llm(path):
    import json
    type = 'test'
    glm = []
    alpaca = []
    label2id = {"任职":0,"隶属于":1,"子":2,"同名于":3,"号":4,"作战":5,"位于":6,"依附":7,"名":8,"讨伐":9,"去往":10,"升迁":11,"杀":12,"管理":13,"弟":14,"出生地":15,"葬于":16,"属于":17,"归属":18,"父":19,"朋友":20,"兄":21,"字":22,"作":23,"姓":24}
    f3 = open(path + type + '_re_glm.json', 'w', encoding='utf-8')
    f4 = open(path + type + '_re_alpaca.json', 'w', encoding='utf-8')
    for line in open(path + type + '_with_cws.tsv'):
        line = line.strip().split('\t')
        new_sen = [x for x in  line[0].replace(' ','')]
        head_entity = ''.join([x for x in  line[10]])
        tail_entity = ''.join([x for x in  line[11]])
        label_type, direction = line[5].split('(')
        instruction = '你是一个语义抽取工具，现在已定义关系包括一下这些' +'，'.join([label.split('(')[0] for label in label2id.keys()]) + '。' + '以下句子中，实体由“【”，“】”标注出来，请你找出两个实体之间的关系，关系的方向由首实体指向尾实体，输出形式为：(首实体，关系，尾实体)。输入的句子为：'
        instruction1 = '你是一个语义抽取工具，现在已定义关系包括一下这些' +'，'.join([label.split('(')[0] for label in label2id.keys()]) + '。' + '以下句子中，实体由“【”，“】”标注出来，请你找出两个实体之间的关系，关系的方向由首实体指向尾实体，输出形式为：(首实体，关系，尾实体)。'
        input = ''.join(new_sen).replace('b','【').replace('e','】').replace('m','【').replace('n','】').replace('u','').replace('v','')
        if direction == 'e1,e2)':
            output = '(' + head_entity + '，' + label_type +'，' + tail_entity +')' 
        if direction == 'e2,e1)':
            output = '(' + tail_entity + '，' + label_type +'，' + head_entity +')' 

        data = {'content':instruction + input,'answer':output}
        glm.append(data)
        # data = json.dumps(data, ensure_ascii=False)

        # f3.write(data + '\n')
        data = {'instruction':instruction1,'input':input,'output':output}
        alpaca.append(data)
        # data = json.dumps(data, ensure_ascii=False)
    import random
    random.shuffle(glm)
    random.shuffle(alpaca)
    for data in glm:
        data = json.dumps(data, ensure_ascii=False)
        f3.write(data + '\n')
    for data in alpaca:
        data = json.dumps(data, ensure_ascii=False)
        f4.write(data + '\n')
re_for_llm('./')