import pandas as pd
import numpy as np
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import font_manager

fname = './Songti.ttc' #读取本地的字体
font = font_manager.FontProperties(fname=fname, size=15)
plt.rcParams['figure.figsize'] = (12,12)
import json

# with open('./saved_data/zztj/rel2id.json', 'r', encoding='utf-8') as f:
#     tag2id = json.load(f)
#
# id2tag = {}
# for key in tag2id.keys():
#     id2tag[tag2id[key]] = key

label2english = {'共事':'Co-worker','位于':'Located in','管理':'Manage','对立':'Opposition',
                  '隶属于':'Be subordinate to','家族':'Family','学术':'Scholarship','任职&爵位&谥号':'Take a title/official',
                  '归并':'Merge','友好':'Friendship','别名':'Alias','见面':'Meet','其他':'Other','生于（地点）':'Birthplace',
                  '失去':'Lose','交谈':'Talk','游历&途经':'Pass by','请求':'Request','分割':'Divide'}
    
y_pred, y_true = [], []
with open('./tmp/zztj/our_bert/paper_nothing/zztj_test_true_pred.txt', 'r', encoding='utf-8') as f:
    text = f.read().split('\n')
    for line in text:
        if line.strip() == '':
            continue
        line = line.split('\t')
        
        y_true.append(label2english[line[1].replace('(e1,e2)','').replace('(e2,e1)','')])
        y_pred.append(label2english[line[2].replace('(e1,e2)','').replace('(e2,e1)','')])


# print(len(y_pred))
# y_true = []

# with open('./saved_data/zztj/test.json', "r", encoding="utf-8") as file:
#     test = json.load(file)
#     for i in range(len(test)):
#         y_true.append(test[i]['relation'])
# print(len(y_true))



# y_pred, y_true = [], []
# for x in text:
#     x = x.split('\t')
#     y_true.append(x[1])
#     y_pred.append(x[2])

# print(set(y_true))
# exit()


labels = list(set(y_true+y_pred))
# y_pred =  ['2','2','3','1','4'] # 类似的格式
# y_true =  ['0','1','2','3','4'] # 类似的格式
# # 对上面进行赋值

my_confusion_matrix = confusion_matrix(y_true, y_pred, labels=labels)  # 可将'1'等替换成自己的类别，如'cat'。


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='sans-serif', size='10')  # 设置字体样式、大小
    plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'SimHei', 'Lucida Grande', 'Verdana']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(200, 200))
    plt.rcParams['figure.dpi'] = 200  # 分辨率

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    import math
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if math.isnan(cm[i, j]):
                cm[i, j] = 0.0
            elif int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.matshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)  # 侧边的颜色条带

    plt.title('Confusion matrix')
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=list(range(len(classes))), yticklabels=list(range(len(classes))),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.05)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(cm[i, j] * 100 , fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.yticks(range(len(labels)), labels, fontproperties=font)
    plt.xticks(range(len(labels)), labels, fontproperties=font, rotation=-45)
    # plt.figure(figsize=(32, 32), dpi=300)
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')


plot_Matrix(my_confusion_matrix, labels, title='')
