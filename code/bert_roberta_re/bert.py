# logging.getLogger().setLevel(logging.INFO)
import argparse
import glob
import logging
logging.getLogger().setLevel(logging.INFO)

import os
import sys
import random
import torch.nn as nn
import numpy as np
import torch
import socket
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# wss
# import ptvsd
# Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('192.168.11.2', 3000), redirect_output=True)
# Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()
from collections import Counter


from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME, BertConfig, BertPreTrainedModel, BertModel)

from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import (RELATION_LABELS, compute_metrics, convert_examples_to_features,
                   output_modes, data_processors)
import torch.nn.functional as F
from sklearn.metrics import recall_score,f1_score,accuracy_score

from argparse import ArgumentParser
from config import Config
from model import BertForSequenceClassification
logger = logging.getLogger(__name__)
#additional_special_tokens = ["[E11]", "[E12]", "[E21]", "[E22]"]
additional_special_tokens = []
#additional_special_tokens = ["e11", "e12", "e21", "e22"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class Classification(nn.Module):
    def __init__(self,config):
        super(Classification, self).__init__()
        self.tanh = nn.Tanh()
        hidden_size = 768
        self.num_labels = config.num_labels
        self.full_con_Layer_0 = nn.Linear(hidden_size,self.num_labels,bias=True)
        
        self.full_con_Layer = nn.Linear(hidden_size,hidden_size,bias=True)
        self.full_con_Layer_1 = nn.Linear(hidden_size,hidden_size,bias=True)
        self.full_con_Layer_2 = nn.Linear(hidden_size*3,self.num_labels,bias=True)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(-1)
        self.loss_function = nn.NLLLoss()
        
        self.num_labels = config.num_labels
        self.l2_reg_lambda = config.l2_reg_lambda
        self.bert = BertModel.from_pretrained(config.pretrained_model_name)
        self.latent_entity_typing = config.latent_entity_typing
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        # 6754
        if config.use_position_embedding:
            self.pos1_embs = nn.Embedding(config.max_seq_len*2 + 2, 300)
            self.pos2_embs = nn.Embedding(config.max_seq_len*2 + 2, 300)
            
        if config.use_cnn:
            #使用PCNN，初始化CNN
            if config.use_position_embedding:
                feature_dim = 768 + 300 * 2
            else:
                feature_dim = 768
            # if config.use_position_embedding:
            self.convs = nn.ModuleList([nn.Conv2d(1, 230, (k, feature_dim), padding=(int(k / 2), 0)) for k in [3]])
            self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=50, kernel_size = 2,)
            self.maxpool = nn.MaxPool1d(3, stride=2)
            self.full_con_Layer_3 = nn.Linear(hidden_size*3 + 230, self.num_labels, bias=True)
#          classifier_size = config.hidden_size*3
        # if self.latent_entity_typing:
#             classifier_size += config.hidden_size*2
        # self.classifier = nn.Linear(
 #            classifier_size, self.config.num_labels)
        # self.latent_size = config.hidden_size
        # self.latent_type = nn.Parameter(torch.FloatTensor(
        #     3, config.hidden_size), requires_grad=True)
        # self.tokenizer =  BertTokenizer.from_pretrained(config.pretrained_model_name,do_lower_case=False)
        
    def forward(self, batch, config):
     
        batch = tuple(t.to(config.device) for t in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                      # XLM and RoBERTa don't use segment_ids
                  'token_type_ids': batch[2],
                      # 'labels':      batch[3],
#                       'e1_mask': batch[4],
#                       'e2_mask': batch[5],
# e1_e2_mask: batch[6]
# first_pos_ids:: batch[7]
# second_pos_ids:: batch[8]
           }

        # print(batch[0].size())
 #        print(batch[1].size())
 #        print(batch[2].size())
        # exit()
        sequence_output, pooled_output = self.bert(input_ids=batch[0], token_type_ids=batch[2], attention_mask=batch[1],output_all_encoded_layers=False)
        # print(sequence_output.size(), pooled_output.size()) #orch.Size([16, 180, 768]) torch.Size([16, 768])
#         exit()
        sequence_output, pooled_output = sequence_output.to(config.device), pooled_output.to(config.device)
        pooled_output = self.dropout(pooled_output)
        
        output = torch.mm(pooled_output.to(config.device),self.full_con_Layer_0.weight.t().to(config.device)) + self.full_con_Layer_0.bias.to(config.device)
        if config.latent_entity_typing:
                # print('使用实体信息')
                # print(batch[4].size())
                e1_length = batch[4].sum(1).float()
                e2_length = batch[5].sum(1).float()
                # print(e1_length.size())
 #                print(e1_length)
                e1_mask = batch[4].unsqueeze(dim=-1).expand_as(sequence_output).to(config.device)
                e1_mask = e1_mask.float().to(config.device)
                e1 = e1_mask * sequence_output
                # print(e1.size())
                e1 = e1.sum(1).to(config.device)
                e1_l = e1_length.unsqueeze(dim=-1).expand_as(e1).to(config.device)
                # average_e1 = torch.div(e1,e1_l).to(config.device) #首实体的表征
                
                e2_mask = batch[5].unsqueeze(dim=-1).expand_as(sequence_output).to(config.device)
                e2_mask = e2_mask.float().to(config.device)
                e2 = e2_mask * sequence_output
                # print(e2.size())
                e2 = e2.sum(1).to(config.device)
                e2_l = e2_length.unsqueeze(dim=-1).expand_as(e2).to(config.device)
                # average_e2 = torch.div(e2,e2_l).to(config.device) #首实体的表征
                # print(average_e1.size())
                e1_embedding = self.tanh(e1).to(config.device)
                e2_embedding = self.tanh(e2).to(config.device)

                e1_embedding_ = torch.mm(e1_embedding, self.full_con_Layer.weight.t().to(config.device)) + self.full_con_Layer.bias.to(config.device)
                e2_embedding_ = torch.mm(e2_embedding, self.full_con_Layer.weight.t().to(config.device)) + self.full_con_Layer.bias.to(config.device)

                sen_embedding = torch.mm(self.tanh(pooled_output).to(config.device),self.full_con_Layer_1.weight.t().to(config.device)) + self.full_con_Layer_1.bias.to(config.device)
              
                h = torch.cat([e1_embedding_, e2_embedding_],-1).to(config.device)
                h = torch.cat([h, sen_embedding],-1).to(config.device)
                output = torch.mm(h, self.full_con_Layer_2.weight.t().to(config.device)) + self.full_con_Layer_2.bias.to(config.device)
        if config.use_cnn:
                   # print('使用CNN')
                   # sequence_output = sequence_output.permute
                   # e1_e2_mask = batch[6]
                   # extended_e_mask = e1_e2_mask.unsqueeze(dim=-1).expand_as(sequence_output).to(config.device)
#
#                    extended_e_mask = extended_e_mask.float() * sequence_output 
                   # print(extended_e_mask.size())
                   if config.use_position_embedding:
                       pos1_emb = self.pos1_embs(batch[7]) 
                       pos2_emb = self.pos2_embs(batch[8]) 
                       sequence_output = torch.cat([sequence_output, pos1_emb, pos2_emb], 2) 
                       
                   # sequence_output = sequence_output.permute(0, 2, 1)
                   # output_conv = self.conv1(sequence_output)
#                    output_conv = self.relu(output_conv )
#                    conv = self.maxpool(output_conv)
#                    conv = torch.reshape(conv, [batch[1].size(0),-1])
                   sequence_output = self.dropout(sequence_output)
                   sequence_output = sequence_output.unsqueeze(1)
                   x = [self.relu(conv(sequence_output)).squeeze(3) for conv in self.convs]
                   x = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
                   x = torch.cat(x, 1)
                   # print(x.size())
                   # exit()
                   h = torch.cat([h, x], -1).to(config.device)
                # exit()
                   # print(h.size())
#                    print()
                   output = torch.mm(h, self.full_con_Layer_3.weight.t().to(config.device)) + self.full_con_Layer_3.bias.to(config.device)
        # output = h
        # print(output.size())
#         exit()
        logits = self.softmax(output)
        # print(logits.size())
#         exit()
        l_logits = self.logsoftmax(output)
                
        loss = self.loss_function(l_logits, batch[3])
                
        _, pre = torch.max(logits,-1)
        # print(pre.size())
 #        exit()
        # else:
#             logits = self.softmax(pooled_output)
#             l_logits = self.logsoftmax(pooled_output)
#
#             loss = self.loss_function(l_logits,batch[3] )
#
#             _, pre = torch.max(logits,-1)
#
    
        return loss, pre



def load_and_cache_examples(config, task, tokenizer, evaluate=False, test=False):
    if config.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = data_processors[config.task_name]()
    # Load data features from cache or dataset file
    # cached_features_file = os.path.join(config.data_dir, 'cached_{}_{}_{}_{}'.format(
    #     'dev' if evaluate else 'train',
    #     list(filter(None, 'bert-large-uncased'.split('/'))).pop(),
    #     str(config.max_seq_len),
    #     str(task)))
    # if os.path.exists(cached_features_file):
    #     print("Loading features from cached file %s",
    #                 cached_features_file)
    #     features = torch.load(cached_features_file)
    # else:
    print("Creating features from dataset file at %s",
                config.data_dir)
    label_list = processor.get_labels()
    examples = None
    if evaluate:
        examples = processor.get_dev_examples(config.data_dir)
    if test:
        examples = processor.get_test_examples(config.data_dir) 
        
    if examples == None:
        # print('读取训练数据')
        examples = processor.get_train_examples(config.data_dir)
        # print(examples)
        random.shuffle(examples)
    
    features = convert_examples_to_features(
        examples, label_list, config.max_seq_len, tokenizer,
         "classification", use_entity_indicator=config.use_entity_indicator,
         use_entity_type=config.use_entity_type, use_cnn=config.use_cnn,use_entity_position=config.use_entity_position)
    # if config.local_rank in [-1, 0]:
    #     print("Saving features into cached file %s",
    #                 cached_features_file)
    #     torch.save(features, cached_features_file)

    if config.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    output_mode = "classification"
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor(
        [f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor(
        [f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
    all_e1_e2_mask = torch.tensor(
        [f.e1_e2_mask for f in features], dtype=torch.long)  # add e1_e2 mask
        
    all_first_pos_ids = torch.tensor(
        [f.first_pos_ids for f in features], dtype=torch.long)
    all_second_pos_ids = torch.tensor(
        [f.second_pos_ids for f in features], dtype=torch.long)
        
    if output_mode == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)
            
            
    dataset = TensorDataset(all_input_ids,
                            all_input_mask,
                            all_segment_ids, 
                            all_label_ids, 
                            all_e1_mask, 
                            all_e2_mask, 
                            all_e1_e2_mask, 
                            all_first_pos_ids,
                            all_second_pos_ids)
    return dataset




def zztj_f1_score(pred_list, label_list, label_num):
    '''

    :param pred_list:
    :param label_list:
    :param label_num:
    :return:
    '''
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(pred_list)):
        guess = pred_list[i]
        gold = label_list[i]

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, label_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]

        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]

        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    recall = 0
    precision = 0
    micro_f1 = 0
    if sum(gold_by_relation.values()) != 0 and sum(guess_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        precision = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * precision / (recall + precision)
    return recall, precision, micro_f1


def main():
    torch.backends.cudnn.enabled = False
    parser = ArgumentParser(
        description="BERT for relation extraction (classification)")
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()
    config = Config(args.config)

    if os.path.exists(config.output_dir) and os.listdir(config.output_dir) and config.train and not config.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(config.output_dir))

    # Setup CUDA, GPU & distributed training
    if config.local_rank == -1 or config.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        config.n_gpu = 1
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        config.n_gpu = 1
    config.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if config.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   config.local_rank, device, config.n_gpu, bool(config.local_rank != -1))

    # Set seed
    set_seed(config.seed)

    # Prepare GLUE task
    processor = data_processors[config.task_name]()
    output_mode = output_modes[config.task_name]
    label_list = processor.get_labels()


    num_labels = len(label_list)
    
    # Load pretrained model and tokenizer
    if config.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        
    config.num_labels = num_labels
    model = Classification(config)
    
    if config.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(config.device)
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name,do_lower_case=False)

    # print("Training/evaluation parameters %s", config)

    # Training
    
    if config.train:
        train_dataset = load_and_cache_examples(
            config, config.task_name, tokenizer, evaluate=False)
        
        config.train_batch_size = config.per_gpu_train_batch_size * \
            max(1, config.n_gpu)
        if config.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            DistributedSampler(train_dataset)

        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=config.train_batch_size)

        if config.max_steps > 0:
            t_total = config.max_steps
            config.num_train_epochs = config.max_steps // (
                len(train_dataloader) // config.gradient_accumulation_steps) + 1
        else:
            t_total = len(
                train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=config.learning_rate, eps=config.adam_epsilon)
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=config.warmup_steps, t_total=t_total)
        # if config.n_gpu > 1:
 #            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if config.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
                                                              output_device=config.local_rank,
                                                              find_unused_parameters=True)

        # Train!
        print("***** Running training *****")
        print("  Num examples = %d", len(train_dataset))
        print("  Num Epochs = %d", config.num_train_epochs)
        print("  Instantaneous batch size per GPU = %d",
                    config.per_gpu_train_batch_size)
        print("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    config.train_batch_size * config.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if config.local_rank != -1 else 1))
        print("  Gradient Accumulation steps = %d",
                    config.gradient_accumulation_steps)
        print("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(config.num_train_epochs),
                                desc="Epoch", disable=config.local_rank not in [-1, 0])
        # Added here for reproductibility (even between python 2 and 3)
        set_seed(config.seed)
        best_f = -1
        print('------begin training------')
        if config.evaluate_during_training:
            eval_task = config.task_name
            eval_output_dir = config.output_dir
            results = {}
            eval_dataset = load_and_cache_examples(
                config, eval_task, tokenizer, evaluate=True)
            if not os.path.exists(eval_output_dir) and config.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)
            config.eval_batch_size = config.per_gpu_eval_batch_size * \
                max(1, config.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_sampler = SequentialSampler(
                eval_dataset) if config.local_rank == -1 else DistributedSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=config.eval_batch_size)
                
        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                                  disable=config.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                model.to(config.device)
                model.train()
                loss, pre =model(batch, config)
                # print(loss)
                # exit()
                if config.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps
                loss.to(config.device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % config.gradient_accumulation_steps == 0:

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # if config.local_rank in [-1, 0] and config.logging_steps > 0 and global_step % config.logging_steps == 0:
                        # Log metrics
                        # Only evaluate when single GPU otherwise metrics may not average well
                logging_loss = tr_loss
                if config.max_steps > 0 and global_step > config.max_steps:
                    epoch_iterator.close()
                    break
            if config.max_steps > 0 and global_step > config.max_steps:
                train_iterator.close()
                break
            tr_loss = tr_loss / global_step
            model.to(config.device)
            if config.local_rank == -1 and config.evaluate_during_training:
                                print('------begin dev------')
                                model.eval()
                                # Eval!
                                print("***** Running evaluation {} *****".format(''))
                                print("  Num examples = %d", len(eval_dataset))
                                print("  Batch size = %d", config.eval_batch_size)
                                eval_loss = 0.0
                                nb_eval_steps = 0
                                y_true, y_pre = [], []
                                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                                    batch = tuple(t.to(config.device) for t in batch)
                                    with torch.no_grad():
                                        tmp_eval_loss, pres =model(batch, config)

                                        eval_loss += tmp_eval_loss.mean().item()
                                    nb_eval_steps += 1
                                    # if preds is None:
        #                   
                                    pres = pres.detach().cpu().numpy().tolist()
                                    y_true += batch[3].detach().cpu().numpy().tolist()
                                    y_pre += pres
                                # print(y_true)
                                # print(y_pre)
#                                 exit()
                                # exit()
                                eval_loss = eval_loss / nb_eval_steps
                                # preds = np.argmax(preds, axis=1)
                                f1 = f1_score(y_true,y_pre,label_list,average='macro')
                                # result = compute_metrics(eval_task, y_pre, y_true)
                                acc = accuracy_score(y_true,y_pre)
                                #微平均计算
                                count_predict = [0]*config.num_labels
                                count_total = [0]*config.num_labels
                                count_right = [0]*config.num_labels
                                for y1,y2 in zip(y_pre,y_true):
                                    count_predict[y1]+=1
                                    count_total[y2]+=1
                                    if y1==y2:
                                        count_right[y1]+=1
                                precision = [0]*config.num_labels
                                recall = [0]*config.num_labels
                                for i in range(1,len(count_predict)):
                                    if count_predict[i]!=0 :
                                        precision[i] = float(count_right[i])/count_predict[i]
            
                                    if count_total[i]!=0:
                                        recall[i] = float(count_right[i])/count_total[i]
                                precision = sum(precision)/(config.num_labels-1) #实际只有48个标签
                                recall = sum(recall)/(config.num_labels-1) #实际只有48个标签
                                f1_micor = (2*precision*recall)/(precision+recall)
                                result = {
                                        "acc": acc,
                                        "宏平均f1": f1,
                                        'p': precision,
                                        'r': recall,
                                        "微平均f1": f1_micor ,
                                    }
                                results.update(result)
                                
                                if not os.path.exists(config.output_dir):
                                    os.makedirs(config.output_dir)
                                print("***** Eval results {} *****".format(''))
                                f = open(config.output_dir + config.task_name + "_m.txt",'a',encoding='utf-8')
                                f.write('epoch:' + str(epoch)+'\n' )
                                
                                for key in sorted(result.keys()):
                                    print("  %s = %s", key, str(result[key]))
                                    f.write(key+'\t' +str(result[key])+'\n')
                                f.write('\n')
                                # f.write('epoch:' + str(epoch)+'\n' + str(f1) +'\n' +str(acc)+'\n\n')
                                
                                output_eval_file = config.output_dir + config.task_name +"_true_pred.txt"
                                with open(output_eval_file, "w") as writer:
                                    for key in range(len(y_true)):
                                        writer.write("%d\t%s\t%s\n" %
                                                     (key, str(RELATION_LABELS[y_true[key]]),str(RELATION_LABELS[y_pre[key]])))
                                # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
                                if config.train and (config.local_rank == -1 or torch.distributed.get_rank() == 0):
                                    # Create output directory if needed
                                    # if not os.path.exists(config.output_dir) and config.local_rank in [-1, 0]:
     #                                    os.makedirs(config.output_dir)
                                    if best_f < f1_micor:
                                            best_f = f1_micor
                                            print("Saving model checkpoint to %s", config.output_dir)
                                            eval_model_path = os.path.join(config.output_dir, 'model.pt')
                                            torch.save(model, eval_model_path)
        # global_step, tr_loss = train(config, train_dataset, model, tokenizer)
            print(" global_step = %s, average loss = %s",global_step, tr_loss)
        f.write('best_f' + str(best_f))

    if config.test:
        test_task = config.task_name
        test_output_dir = config.output_dir
        results = {}
        test_dataset = load_and_cache_examples(
            config, test_task, tokenizer, evaluate=False,test=True)
        if not os.path.exists(test_output_dir) and config.local_rank in [-1, 0]:
            os.makedirs(test_output_dir)
        config.test_batch_size = config.per_gpu_eval_batch_size * \
            max(1, config.n_gpu)
        # Note that DistributedSampler samples randomly
        test_sampler = SequentialSampler(
            test_dataset) if config.local_rank == -1 else DistributedSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=config.test_batch_size)
        model_path = os.path.join(config.output_dir, 'model.pt')
        model = torch.load(model_path)
        # model.from_pretrained(checkpoint )
        model.to(device)
        model.eval()
        
        
        y_true, y_pre = [], []
        for batch in tqdm(test_dataloader, desc="Evaluating"):
               batch = tuple(t.to(config.device) for t in batch)
               with torch.no_grad():
                   _, pres =model(batch, config)

                                    
               pres = pres.detach().cpu().numpy().tolist()
               y_true += batch[3].detach().cpu().numpy().tolist()
               y_pre += pres
           # print(y_true)
           # print(y_pre)
        #                                 exit()
          
        test_f1 = f1_score(y_true, y_pre, label_list, average='macro')
           # result = compute_metrics(eval_task, y_pre, y_true)

                                    
        test_acc = accuracy_score(y_true,y_pre)
           #微平均计算
        count_predict = [0]*config.num_labels
        count_total = [0]*config.num_labels
        count_right = [0]*config.num_labels
        for y1,y2 in zip(y_pre,y_true):
               count_predict[y1]+=1
               count_total[y2]+=1
               if y1==y2:
                   count_right[y1]+=1
        test_precision = [0]*config.num_labels
        test_recall = [0]*config.num_labels
        for i in range(1,len(count_predict)):
               if count_predict[i]!=0 :
                   test_precision[i] = float(count_right[i])/count_predict[i]
            
               if count_total[i]!=0:
                   test_recall[i] = float(count_right[i])/count_total[i]
        test_f1s,test_p,test_r = [], [], []
        for i in range(1, len(test_precision)):
            if test_precision[i] == 0 and test_recall[i] == 0:
                test_f1s.append(0)
                test_p.append(0)
                test_r.append(0)
            else:

                test_f1s.append((2*test_precision[i]*test_recall[i])/(test_precision[i]+test_recall[i]))
        



        # test_precision = sum(test_precision)/(config.num_labels-1)  #实际只有48个标签
        # test_recall = sum(test_recall)/(config.num_labels-1)        #实际只有48个标签
        # test_f1 = (2*test_precision*test_recall)/(test_precision+test_recall)
        # r_w = sum(test_r)
        # p_w = sum(test_p)
        # f1_w = sum(test_f1s)
        # f = (2*p_w*r_w)/(p_w+r_w)

        f1_micro = zztj_f1_score(y_pre, y_true, config.num_labels)

        test_result = {
                   "test_acc": test_acc,
                   "test_宏平均f1": test_f1,
                   # 'tets_p': p_w,
                   # 'tets_r': r_w,
                   # "test加权f1": f1_w, #加权F1的正确算法，每个类别的F 乘以占比再相加
                   # "test加权f2": f,
                   '最终f1_micro': f1_micro,
               }
           # test_results.update(test_result)
                                
        if not os.path.exists(config.output_dir):
               os.makedirs(config.output_dir)
        print("***** test results {} *****".format(''))
        f = open(config.output_dir + config.task_name + "_test_m.txt",'a',encoding='utf-8')
        # f.write('epoch:' + str(epoch)+'\n' )
        for key in sorted(test_result.keys()):
               print("  %s = %s", key, str(test_result[key]))
               f.write(key+'\t' +str(test_result[key])+'\n')
        f.write('\n')
        f2 = open(config.data_dir +'/test.tsv','r',encoding='utf-8')
        test_text = f2.read().strip().split('\n')
        output_eval_file = config.output_dir + config.task_name + "_test_true_pred.txt"
        # print(len(y_true),len(y_pre),len(test_text))
        # exit()
        with open(output_eval_file, "w") as writer:
               for key in range(len(y_true)):
                   writer.write("%d\t%s\t%s\t%s\n" %
                                (key, str(RELATION_LABELS[y_true[key]]),str(RELATION_LABELS[y_pre[key]]), test_text[key]))
        output_file = config.output_dir + "relation_type_f1.txt"
        f3 = open('./data/' + config.task_name + '/relation2id.tsv','r',encoding='utf-8')
        relation = f3.read().split('\n')
        with open(output_file, "w") as writer:
            for i in range(1,len(test_f1s)):
                 writer.write(relation[i] + '\t' + str(test_f1s[i]) + '\n')
        #绘制hunxiaojuzhen
        # C = confusion_matrix(y_true, y_pred)
if __name__ == "__main__":
    
    main()
#