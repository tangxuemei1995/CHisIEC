import os
def each_class_f1(labels, y_pre, y_true, output_dir, epoch): 
        labels = labels
        count_predict, count_total,count_right = {}, {}, {}
        # print(labels)
        for x in labels:
            count_predict[x] = 0
            count_total[x] = 0
            count_right[x] = 0
        for y1,y2 in zip(y_pre, y_true):
            #    print(y1,y2)
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
        f2 = open( os.path.join(output_dir, 'f1.txt'),'a+',encoding='utf-8')
        f2.write('epoch\t' + epoch +'\n')
        labels = list(set(y_true))
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
output_dir= '/ceph/home/jun01/tangxuemei/Alpaca2/alpaca_cclue_re_epoch=30/'
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

each_class_f1(labels, pre, true, output_dir, 'test')
# print(no_pre_relation)
