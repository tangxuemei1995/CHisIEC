import json
import random
# import requests
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# 读取 JSON 文件

from langconv import *

def fan_jian(char):
    '''
    繁简转化
    '''

    jian_char = Converter('zh-hans').convert(char)

    return jian_char

def wenxinyiyan(content):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=24.ec670bfdfb6639efa7f0f751fed5537e.2592000.1705472986.282335-45227156"
    
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    return response.text

def emb(docs):


    sentence_model = SentenceTransformer('SIKU-BERT/sikubert')
#将所有的句子都转化为embedding
    embeddings = sentence_model.encode(docs, show_progress_bar=True)
    return embeddings


# file_path = './chisre_train_dev.json'

# with open(file_path, 'r') as file:
#     # 使用 json.load() 加载 JSON 数据
#     data_train = json.load(file)

# train_list = [] #用来存储所有的训练集句子，方便后面计算相似度
# train_token2sample = {}
# for d in data_train:
#     train_list.append(d['tokens'])
# other_embeddings = emb(train_list)
    # train_token2sample[d['tokens']] = d

# 现在，'data_train' 变量包含了从 JSON 文件中加载的数据

etag2char = {'PER':'人物','LOC':'地点','OFI':'职官','TIME':'时间','BOOK':'书籍','GPE':'团体'}

# demonstration = ''

file_path = './chisre_train.json'

with open(file_path, 'r') as file:
    # 使用 json.load() 加载 JSON 数据
    data_test = json.load(file)


#随机挑选示例，或者使用语义相似度来挑选最相似的示例
use_random = False

for j in range(len(data_test)):#使用前10条测试

    test_sample = data_test[j]
    token_test = test_sample['tokens']
    entities = test_sample['entities']
    relations = test_sample['relations']
    gold_e = ''
    e_test = {'人物':[],'地点':[],'职官':[],'时间':[],'书籍':[],'团体':[]}
    for x in entities:
            e_test[etag2char[x['type']]].append(''.join([fan_jian(xx) for xx in x['span']]))
    re_test = []
    gold_e += '以上句子中实体有:'
    for key in e_test.keys():
        gold_e += key + '：' + '['  + '；'.join(e_test[key]) + ']'  + '，'

    output = '' 
    for x in relations:
        re_test.append('[' + ''.join([fan_jian(xx) for xx in x['head_span']]) + '，' + x['type'].replace(' ','')  + '，' +''.join([fan_jian(xx) for xx in x['tail_span']]) + ']')
    
    output += '根据以上实体，找出以上句子中的关系三元组有：'
    output += '；'.join(re_test) + '。'
    

    # role = '你是一个实体关系抽取工具，现在已定义实体类型：[人物、时间、地点、职官、书籍、团体]，预定义关系包括一下这些：[其他，任职，到达，属地，攻伐，管理，位于，派遣，生于（地点），政治奥援，封，父，交谈，杀，作者，打败，别名，同僚，地点别名，兄弟，旧属，见面，设置，投靠，祖孙，隶属于，合作，离开，请求，游历，归附，救援，治所，修建，死于（时间），母，害怕，夫妻，旧臣，死于（地点）]。现在给你一个句子，你需要首先从句子中找出实体，输出格式为：人物:[人物;人物;人物],地点:[地点;地点;地点],职官:[职官;职官:职官],时间:[时间;时间;时间],书籍:[书籍;书籍;书籍],团体:[团体;团体;团体]，然后根据实体找出关系三元组，输出格式为：[实体,关系,实体];[实体,关系,实体];[实体,关系,实体]，关系请一定从预定义的关系中选择。\n输入是:' + token_test + '\n以上句子中实体有:'
    instruction = '你是一个实体关系抽取工具，现在已定义实体类型：[人物、时间、地点、职官、书籍、团体]，预定义关系包括一下这些：[其他，任职，到达，属地，攻伐，管理，位于，派遣，生于（地点），政治奥援，封，父，交谈，杀，作者，打败，别名，同僚，地点别名，兄弟，旧属，见面，设置，投靠，祖孙，隶属于，合作，离开，请求，游历，归附，救援，治所，修建，死于（时间），母，害怕，夫妻，旧臣，死于（地点）]。现在给你一个句子，以及句子中已有的实体，实体格式为：人物：[人物；人物；人物]，地点：[地点；地点；地点]，职官：[职官；职官；职官]，时间：[时间；时间；时间]，书籍：[书籍；书籍；书籍]，团体：[团体；团体；团体]。然后你需要根据实体找出关系三元组，输出格式为：[实体，关系，实体]；[实体，关系，实体]；[实体，关系，实体]，关系请一定从预定义的关系中选择。输入是：' 
    input = token_test + gold_e

    

   
    output = {'instruction':instruction, 'input': input, 'output':output}
    with open('train.json', 'a+', encoding='utf-8') as json_file:
        json.dump(output, json_file, ensure_ascii=False)
        json_file.write('\n')


