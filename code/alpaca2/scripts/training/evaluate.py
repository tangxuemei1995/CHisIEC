
def each_class_f1(labels,y_pre,y_true,true_labels):     
        count_predict, count_total,count_right = {}, {}, {}
        for x in labels:
            count_predict[x] = 0
            count_total[x] = 0
            count_right[x] = 0
        for y1,y2 in zip(y_pre,y_true):
               count_predict[y1]+=1
               count_total[y2]+=1
               if y1==y2:
                   count_right[y1]+=1
        test_precision, test_recall= {}, {}

        for x in labels:
            test_precision[x] = 0
            test_recall[x] = 0
        for i in labels:
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
        f1 = 0
        for x in test_f1s:
            if x in labels:
               f1 += test_f1s[x]
               print(x + '\t' + str(test_f1s[x]))
        print('平均值f1', f1/len(labels))
#仅分类
import json
true, pre = [], []
no_pre_relation = 0
for line in open('/ceph/home/jun01/tangxuemei/Alpaca2/alpaca_cclue_re_epoch=30/test_result.txt'):

    line = line.strip().split(' [/INST] ')[1]
    line = line.split('</s>		true label:	')
    if len(line) != 2:
        
        continue
    pre_sen, true_sen = line[0].split('；'), line[1].split('；')
    # pre_sen = [x.replace('。','') for x in pre_sen]
    pre_label = pre_sen[0].split('，')[1]
    true_label = true_sen[0].split('，')[1]
    true.append(true_label)
    pre.append(pre_label)

true = true[0:len(pre)]

labels = pre + true

labels = list(set(labels))

true_labels = list(set(true))

# each_class_f1(labels, pre, true, output_dir, 'test')

each_class_f1(labels, pre, true, true_labels)
print(no_pre_relation)
