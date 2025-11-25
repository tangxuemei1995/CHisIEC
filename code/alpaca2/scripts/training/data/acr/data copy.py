
import json



f = open('acr_types.json','r',encoding='utf-8')
type = json.load(f)

relations = type['relations'].keys()
rel_types = '，'.join([re.replace(' ', '') for re in relations])
instructions = '你是一个实体语义关系抽取工具，我们预定义的关系类型有以下这些：' + rel_types + '。你需要对以下输入的句子做实体关系抽取，并输出所有的三元组，输出形式为：(实体，关系类型，实体)。'
def to_alpaca_traing(path):
    f1 = open(path,'r',encoding='utf-8')
    text = json.load(f1) 
    input = text['tokens']
    relations = text['relations']



to_alpaca_traing('./acr_test.json')