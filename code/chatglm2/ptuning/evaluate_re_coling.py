

'''
coling ner评测
'''
import json



def each_class_f1(labels,y_pre,y_true,true_labels): 
        # labels = list(set([line.strip().split('(')[0] for line in open('/ceph/home/jun01/tangxuemei/re/data/coling/relation2id.tsv')] ))   
        count_predict, count_total,count_right = {'other':0}, {'other':0}, {'other':0}
        for x in labels:
            count_predict[x] = 0
            count_total[x] = 0
            count_right[x] = 0
        for y1,y2 in zip(y_pre,y_true):
               print(y1,y2)
               if y1 not in labels:
                    count_predict['other']+=1
               else:
                    count_predict[y1]+=1
               count_total[y2]+=1
               if y1==y2:
                   count_right[y1]+=1
        test_precision, test_recall= {}, {}

        for x in labels:
            test_precision[x] = 0
            test_recall[x] = 0
        correct, pre, true =0,0,0

        for i in labels:
               correct += count_right[i]
               pre += count_predict[i]
               true += count_total[i]
               
               if count_predict[i]!=0 :
                   test_precision[i] = float(count_right[i])/count_predict[i]
            
               if count_total[i]!=0:
                   test_recall[i] = float(count_right[i])/count_total[i]
            
        test_f1s, test_p, test_r = {}, {}, {}
        for x in labels:
            test_f1s[x] = 0
            test_p[x] = 0
            test_r[x] = 0

        for i in labels:
            if test_precision[i] == 0 and test_recall[i] == 0:
                test_f1s[i] = 0
                test_p[i] = 0
                test_r[i] = 0
            else:
                test_f1s[i] = (2*test_precision[i]*test_recall[i])/(test_precision[i]+test_recall[i])
        microp,micror,microf1 = 0,0,0
        f2 = open('/workspace/tangxuemei/chatglm2/ptuning/output/re_coling_10-128-2e-2/f1.txt','w',encoding='utf-8')
        labels = true_labels
        for x in test_f1s:
            if x in true_labels:
               microf1 += test_f1s[x]
               microp += test_precision[x]
               micror += test_recall[x]

               f2.write(x + '\t' + str(test_precision[x]*100) + '\t' + str(test_recall[x]*100)+ '\t' + str(test_f1s[x]*100) + '\n')
               print(x,test_precision[x], test_recall[x], str(test_f1s[x]))
        mean_microf1 =   (2*(microp/len(labels))*(micror/len(labels)))/ ((microp/len(labels))+(micror/len(labels)))  
        print('平均值microf1', microp/len(labels),micror/len(labels),mean_microf1)

        print('平均值macrof1', correct/pre*100,correct/true*100,(2*(correct/pre*100)*(correct/true*100))/((correct/pre*100)+(correct/true*100)))
        f2.write('p\t' + str((microp/len(labels))*100) + '\n')
        f2.write('r\t' + str((micror/len(labels))*100) + '\n')
        f2.write('f\t' + str(mean_microf1) + '\n')
        f2.write('acc\t' + str(correct/pre*100) + '\n\n')
        return microf1/len(labels)
#仅分类
import json
true, pre = [], []
no_pre_relation = 0
corr, act = 0, 0

# for line in open('/ceph/home/jun01/tangxuemei/glm2/ptuning/output/re-128-2e-2/generated_predictions.txt'):

#     line = line.strip().split(' [/INST] ')[1]
#     line = line.split('</s>		true label:	')
#     if len(line) != 2:
        
#         continue
#     pre_sen, true_sen = line[0].split('；'), line[1].split('；')
#     pre_sen = [x.replace('。','') for x in pre_sen]
#     print(pre_sen)
#     exit()
#     for x in true_sen:
#         if x == '':
#               continue
#         x = x.replace('。','')
#         act += 1
#         if x in pre_sen:
#             corr += 1
# print(corr/act)
import json
def eval_ner_alpaca(path):

    trues, pres, labels = [], [], []
    f = open(path,'r',encoding='utf-8')
    text = f.read().split('\n')
    for line in text:
        # print(line)
        line = json.loads(line)
        # print(line)
        # exit()

        true,pre = line['labels'],line['predict']
        pre = pre.split('，')[1]
        
        true = true.split('，')[1]
        trues.append(true)
        pres.append(pre)
    # print(trues)
    # print(pres)
    labels = pres + trues
    labels = list(set(labels))

    true_labels = list(set(trues))
    true_labels = ['上下级', '同僚', '政治奥援', '兄弟', '到达', '敌对攻伐', '出生于某地', '任职', '别名', '父母', '驻守', '管理']
    # print(true_labels)
    # exit()
    each_class_f1(true_labels, pres, trues, true_labels)

        

if __name__=="__main__":
    eval_ner_alpaca('/workspace/tangxuemei/chatglm2/ptuning/output/re_coling_10-128-2e-2/generated_predictions.txt')



# 别名 0.6666666666666666 0.631578947368421 0.6486486486486486
# 父母 0.8387096774193549 0.9122807017543859 0.8739495798319329
# 管理 0.56 0.5 0.5283018867924528
# 出生于某地 0.85 0.6538461538461539 0.7391304347826088
# 任职 0.976 0.9682539682539683 0.9721115537848605
# 兄弟 0.46153846153846156 0.6 0.5217391304347826
# 驻守 0.6785714285714286 0.7037037037037037 0.6909090909090909
# 同僚 0.4659090909090909 0.6119402985074627 0.5290322580645161
# 政治奥援 0.46296296296296297 0.25252525252525254 0.32679738562091504
# 到达 0.7058823529411765 0.8 0.7500000000000001
# 上下级 0.6929824561403509 0.7383177570093458 0.7149321266968325
# 敌对攻伐 0.7761194029850746 0.8297872340425532 0.8020565552699229
# 平均值microf1 0.6779452083445473 0.6835195014176039 0.674800720903047
# 平均值macrof1 75.72192513368984 75.56029882604055 75.64102564102564