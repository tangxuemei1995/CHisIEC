
import json

from langconv import *
# en_type = {'OFI':'职 官','LOC':'地 点','TIME':'时 间','PER':'人 物','GOP':'团 体','GPE':'团 体','BOOK':'书 籍'}
def fan_jian(char):
    '''
    繁简转化
    '''

    jian_char = Converter('zh-hans').convert(char)

    return jian_char


f = open('acr_types.json','r',encoding='utf-8')
type = json.load(f)

relations = type['relations'].keys()
rel_types = '，'.join([re.replace(' ', '') for re in relations])
instruction = '你是一个实体语义关系抽取工具，我们预定义的关系类型有以下这些：' + rel_types + '。你需要对以下输入的句子做实体关系抽取，并输出所有的三元组，输出形式为：(实体，关系类型，实体)；(实体，关系类型，实体)。'

def to_alpaca_traing(path):
    f2 = open(path.replace('acr_',''),'w',encoding='utf-8')

    f1 = open(path,'r',encoding='utf-8')
    texts = json.load(f1)
    result = []
    # output = ''
    for text in texts:
        output = ''
        input = text['tokens']
        relations = text['relations']
        for relation in relations:
            type = relation['type'].replace(' ','')

            head_span = ''.join([fan_jian(x) for x in relation['head_span']])
            tail_span = ''.join([fan_jian(x) for x in relation['tail_span']])
            output += "（" + head_span + "，" + type + "，" + tail_span + "）" +'；'
        output = output[0:-1] + '。'
        data = {"instruction":instruction,"input":input,'output':output}

        data = json.dumps(data, ensure_ascii=False)
        f2.write(data + '\n')






to_alpaca_traing('./acr_test.json')
to_alpaca_traing('./acr_train.json')
to_alpaca_traing('./acr_train_dev.json')