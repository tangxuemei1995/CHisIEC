# coding=utf-8
# Copyright 2019 Hao WANG, Shanghai University, KB-NLP team.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.a
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import re
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

from langconv import *
def fan_jian(char):
    '''
    繁简转化
    '''

    jian_char = Converter('zh-hans').convert(char)

    return jian_char

def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

#
# RELATION_LABELS = ['Other', 'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)',
#                    'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
#                    'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
#                    'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
#                    'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
#                    'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
#                    'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
#                    'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
#                    'Content-Container(e1,e2)', 'Content-Container(e2,e1)']

RELATION_LABELS =['O', '别名(e1,e2)', '别名(e2,e1)','任职&爵位&谥号(e2,e1)','任职&爵位&谥号(e1,e2)','管理(e2,e1)','管理(e1,e2)','共事(e1,e2)','共事(e2,e1)','对立(e1,e2)','对立(e2,e1)','见面(e1,e2)','见面(e2,e1)','位于(e1,e2)','位于(e2,e1)','隶属于(e1,e2)','隶属于(e2,e1)','交谈(e1,e2)','交谈(e2,e1)','学术(e1,e2)','学术(e2,e1)','家族(e1,e2)','家族(e2,e1)','失去(e2,e1)','失去(e1,e2)','生于（地点）(e2,e1)','生于（地点）(e1,e2)','归并(e1,e2)','归并(e2,e1)','友好(e1,e2)','友好(e2,e1)','请求(e1,e2)','其他(e1,e2)','其他(e2,e1)','游历&途经(e1,e2)','分割(e1,e2)','分割(e2,e1)']

GLUE_TASKS_NUM_LABELS = {
    "mrpc": 2,
    "semeval": 19,
    "cclue": 48,
    "zztj": len(RELATION_LABELS),

}
# RELATION_LABELS =['O', '别名(e1,e2)','别名(e2,e1)','任职&爵位&谥号(e2,e1)','任职&爵位&谥号(e1,e2)','管理(e2,e1)','管理(e1,e2)','共事(e1,e2)','共事(e2,e1)','友好(e1,e2)','友好(e2,e1)','对立(e1,e2)','对立(e2,e1)','想做(e1,e2)','想做(e2,e1)','见面(e1,e2)','见面(e2,e1)','位于(e1,e2)','位于(e2,e1)','隶属于(e1,e2)','隶属于(e2,e1)','交谈(e1,e2)','交谈(e2,e1)','学术(e1,e2)','学术(e2,e1)','家族(e1,e2)','家族(e2,e1)','失去(e2,e1)','失去(e1,e2)','生于（地点）(e2,e1)','生于（地点）(e1,e2)','归并(e1,e2)','归并(e2,e1)','攻占(e1,e2)','攻占(e2,e1)','同位语(e1,e2)','同位语(e2,e1)','请求(e1,e2)','其他(e1,e2)','其他(e2,e1)','游历&途经(e1,e2)','分割(e1,e2)']
# RELATION_LABELS =['别名(e1,e2)','别名(e2,e1)','任职&爵位&谥号(e2,e1)','任职&爵位&谥号(e1,e2)','管理(e2,e1)','管理(e1,e2)','合作(e1,e2)','合作(e2,e1)','朋友&友善(e1,e2)','朋友&友善(e2,e1)','拥立&拥护(e1,e2)','拥立&拥护(e2,e1)','想做(e1,e2)','想做(e2,e1)','同事(e1,e2)','派遣(e1,e2)','派遣(e2,e1)','愤怒(e1,e2)','愤怒(e2,e1)','见面(e1,e2)','见面(e2,e1)','驻&待在(e1,e2)','驻&待在(e2,e1)','上下级(e2,e1)','上下级(e1,e2)','打败(e2,e1)','打败(e1,e2)','属地(e1,e2)','属地(e2,e1)','封&赐官(e1,e2)','封&赐官(e2,e1)','讨厌(e2,e1)','讨厌(e1,e2)','杀(e2,e1)','杀(e1,e2)','游说&劝说(e1,e2)','游说&劝说(e2,e1)','攻击(e2,e1)','攻击(e1,e2)','作者(e1,e2)','起兵&聚兵(e1,e2)','起兵&聚兵(e2,e1)','到达(e1,e2)','到达(e2,e1)','害怕(e1,e2)','害怕(e2,e1)','回到(e1,e2)','离任&被罢免(e2,e1)','生于（地点）(e2,e1)','生于（地点）(e1,e2)','归附(e1,e2)','背叛(e1,e2)','背叛(e2,e1)','父子(e1,e2)','父子(e2,e1)','后人(e2,e1)','后人(e1,e2)','投靠(e1,e2)','投靠(e2,e1)','攻占(e1,e2)','攻占(e2,e1)','推荐(e1,e2)','推荐(e2,e1)','斗争(e1,e2)','斗争(e2,e1)','帮助(e1,e2)','帮助(e2,e1)','族弟(e2,e1)','族弟(e1,e2)','离心离德(e1,e2)','离心离德(e2,e1)','兄弟(e1,e2)','兄弟(e2,e1)','同位语(e1,e2)','同位语(e2,e1)','逃亡至(e1,e2)','请求(e1,e2)','守卫(e1,e2)','守卫(e2,e1)','祖孙(e1,e2)','祖孙(e2,e1)','爱戴(e1,e2)','爱戴(e2,e1)','礼遇(e1,e2)','礼遇(e2,e1)','其他(e1,e2)','其他(e2,e1)','叔侄(e2,e1)','叔侄(e1,e2)','迁徙至(e1,e2)','求救(e1,e2)','旧属(e1,e2)','游历&途经(e1,e2)','救(e1,e2)','救(e2,e1)','捉拿(e1,e2)','捉拿(e2,e1)']
# RELATION_LABELS = ['O', '别名(e1,e2)','别名(e2,e1)','任职&爵位&谥号(e2,e1)','任职&爵位&谥号(e1,e2)','管理(e2,e1)','管理(e1,e2)','合作(e1,e2)','合作(e2,e1)','朋友&友善(e1,e2)','朋友&友善(e2,e1)','拥立&拥护(e1,e2)','拥立&拥护(e2,e1)','想做(e1,e2)','想做(e2,e1)','同事(e1,e2)','派遣(e1,e2)','派遣(e2,e1)','愤怒(e1,e2)','愤怒(e2,e1)','见面(e1,e2)','见面(e2,e1)','驻&待在(e1,e2)','驻&待在(e2,e1)','上下级(e2,e1)','上下级(e1,e2)','打败(e2,e1)','打败(e1,e2)','属地(e1,e2)','属地(e2,e1)','封&赐官(e1,e2)','封&赐官(e2,e1)','讨厌(e2,e1)','讨厌(e1,e2)','杀(e2,e1)','杀(e1,e2)','游说&劝说(e1,e2)','游说&劝说(e2,e1)','攻击(e2,e1)','攻击(e1,e2)','作者(e1,e2)','起兵&聚兵(e1,e2)','起兵&聚兵(e2,e1)','到达(e1,e2)','到达(e2,e1)','害怕(e1,e2)','害怕(e2,e1)','回到(e1,e2)','离任&被罢免(e2,e1)','离任&被罢免(e1,e2)','生于（地点）(e2,e1)','生于（地点）(e1,e2)','归附(e1,e2)','背叛(e1,e2)','背叛(e2,e1)','父子(e1,e2)','父子(e2,e1)','领导(e1,e2)','领导(e2,e1)','后人(e2,e1)','后人(e1,e2)','投靠(e1,e2)','投靠(e2,e1)','攻占(e1,e2)','攻占(e2,e1)','推荐(e1,e2)','推荐(e2,e1)','斗争(e1,e2)','斗争(e2,e1)','位于(e2,e1)','位于(e1,e2)','帮助(e1,e2)','帮助(e2,e1)','族弟(e2,e1)','族弟(e1,e2)','离心离德(e1,e2)','离心离德(e2,e1)','兄弟(e1,e2)','兄弟(e2,e1)','同位语(e1,e2)','同位语(e2,e1)','逃亡至(e1,e2)','请求(e1,e2)','守卫(e1,e2)','守卫(e2,e1)','祖孙(e1,e2)','祖孙(e2,e1)','爱戴(e1,e2)','爱戴(e2,e1)','礼遇(e1,e2)','礼遇(e2,e1)','其他(e1,e2)','其他(e2,e1)','叔侄(e2,e1)','叔侄(e1,e2)','迁徙至(e1,e2)','出兵(e1,e2)','求救(e1,e2)','旧属(e1,e2)','旧属(e2,e1)','游历&途经(e1,e2)','救(e1,e2)','救(e2,e1)','投降(e2,e1)','投降(e1,e2)','捉拿(e1,e2)','捉拿(e2,e1)']

# RELATION_LABELS = ['O','任职(e2,e1)','隶属于(e1,e2)','任职(e1,e2)','子(e2,e1)',
#                     '同名于(e1,e2)','隶属于(e2,e1)','号(e1,e2)','作战(e2,e1)','位于(e1,e2)',
#                     '依附(e1,e2)','名(e1,e2)','讨伐(e1,e2)','子(e1,e2)','去往(e1,e2)','升迁(e1,e2)',
#                     '杀(e1,e2)','管理(e2,e1)','同名于(e2,e1)','弟(e1,e2)','出生地(e1,e2)',
#                     '出生地(e2,e1)','葬于(e1,e2)','管理(e1,e2)','属于(e1,e2)','归属(e2,e1)',
#                     '父(e1,e2)','依附(e2,e1)','属于(e2,e1)','朋友(e1,e2)','杀(e2,e1)',
#                     '朋友(e2,e1)','位于(e2,e1)','作战(e1,e2)','兄(e1,e2)','字(e1,e2)',
#                     '作(e1,e2)','弟(e2,e1)','姓(e2,e1)','姓(e1,e2)','兄(e2,e1)',
#                     '去往(e2,e1)','讨伐(e2,e1)','名(e2,e1)','号(e2,e1)','作(e2,e1)',
#                     '葬于(e2,e1)','父(e2,e1)','归属(e1,e2)']

entity2str = {'BOOK': '书籍', 'JOB': '职官', 'PER': '人物', 'LOC': '地点', 'ORG': '机构', 'OFI': '职官', 'GPE': '团体', 'TIME': '时间', 'KWD': '关键词'}


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, e1_type=None, e2_type=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.e1_type = e1_type
        self.e2_type = e2_type
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# class InputFeatures(object):
#     """A single set of features of data."""

#     def __init__(self,
#                  input_ids,
#                  input_mask,
#                  segment_ids,
#                  label_id):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.label_id = label_id

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 e11_p, e12_p, e21_p, e22_p,
                 e1_mask, e2_mask,
                 e1_e2_mask,
                 segment_ids,
                 label_id,
                 first_pos_ids,
                 second_pos_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.e11_p = e11_p
        self.e12_p = e12_p
        self.e21_p = e21_p
        self.e22_p = e22_p
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.e1_e2_mask = e1_e2_mask
        self.first_pos_ids = first_pos_ids
        self.second_pos_ids = second_pos_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        print("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            print(line)
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[4]
            text_b = line[5]
            label = RELATION_LABELS.index(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SemEvalProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        print("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(19)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets.
        e.g.,: 
        2	the [E11] author [E12] of a keygen uses a [E21] disassembler [E22] to look at the raw assembly code .	6
        """
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            print(line)
            text_a = line[1]
            text_b = None
            #label = RELATION_LABELS.index(int(line[2]))
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
        
class CLUEProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        print("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
            
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
    def get_labels(self):
        """See base class."""
        return [i for i in range(49)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets.
        e.g.,: 
        2	the [E11] author [E12] of a keygen uses a [E21] disassembler [E22] to look at the raw assembly code .	6
        """
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            # print(line)
            text_a = line[0]
            text_b = None
            #label = RELATION_LABELS.index(int(line[2]))
            label = str(int(line[1])+1)
            e1_type = line[2]
            e2_type = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, e1_type=e1_type, e2_type=e2_type, text_b=text_b, label=label))
        return examples


class ZZTJProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        print("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [i for i in range(GLUE_TASKS_NUM_LABELS['zztj'])]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets.
        e.g.,:
        2	the [E11] author [E12] of a keygen uses a [E21] disassembler [E22] to look at the raw assembly code .	6
        """
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            # print(line)
            # text_a = line[0]
            or_text = line[0].split()

            text = ''
            for char in or_text:
                text += fan_jian(char) + ' '
            text_a = text.strip()
            text_b = None
            # label = RELATION_LABELS.index(int(line[2]))
            label = str(int(line[1]) + 1)
            e1_type = line[2]
            e2_type = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, e1_type=e1_type, e2_type=e2_type, text_b=text_b, label=label))

        return examples

def token_pos(tokens, e11_p, e12_p, max_len):
    '''
    查找token相对于首实体的相对位置，前后最大的位置为180，
    将位置映射到（0，2*max_Len+1）2*max_Len+1对应pad，180对应实体本身
    '''
    pos = [0]*len(tokens)
    pos2id = [0]*len(tokens)
    for i in range(len(tokens)):
        # if i in range(e11_p, e12_p+1):
#             pos[i] = 1
        if i < e11_p:
            pos[i] = i-e11_p
            if pos[i] < -max_len:
                pos2id[i] = 2 * max_len + 1
            else:
                pos2id[i] = pos[i] + max_len
        elif i > e12_p:
            pos[i] = i-e12_p
            if pos[i] > max_len:
                pos2id[i] = 2 * max_len + 1
            else:
                pos2id[i] = pos[i] + max_len
        else:
            pos[i] = 0
            pos2id[i] = pos[i] + max_len
    return pos2id
    
        


def convert_examples_to_features(examples, label_list, max_seq_len,
                                 tokenizer, output_mode,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 use_entity_indicator=True,
                                 use_entity_type=True,
                                 use_cnn=True,
                                 use_entity_position=True):
    """ Loads a data file into a list of `InputBatch`s
        Default, BERT/XLM pattern: [CLS] + A + [SEP] + B + [SEP]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        
        # print(len(tokens_a))
#         print(len(example.text_a))
#         exit()
        if use_entity_indicator:
            # e11_p = tokens_a.index("e11")+1   # the start position of entity1
            # e12_p = tokens_a.index("e12")+2  # the end position of entity1
            # e21_p = tokens_a.index("e21")+1   # the start position of entity2
            # e22_p = tokens_a.index("e22")+2  # the end position of entity2
            # e11_p = tokens_a.index("[E11]")+2   # the start position of entity1
            # e12_p = tokens_a.index("[E12]")+1  # the end position of entity1
            # e21_p = tokens_a.index("[E21]")+2   # the start position of entity2
            # e22_p = tokens_a.index("[E22]")+1  # the end position of entity2
            l = len(tokens_a)
            # print(tokens_a)
            e11_p = tokens_a.index("b")+1-1   # the start position of entity1
            e12_p = l-tokens_a[::-1].index("e")-1+1  # the end position of entity1
            if 'm' not in tokens_a:
                print(example.text_a)
            e21_p = tokens_a.index("m")+1 - 1  # the start position of entity2
            # the end position of entity2
            e22_p = l-tokens_a[::-1].index("n")-1+1
            # print(tokens_a[e11_p:e12_p])
#             print(tokens_a[e21_p:e22_p])
            # exit()
        e1 = ''.join(tokens_a[e11_p+1:e12_p-1])
        e2 = ''.join(tokens_a[e21_p+1:e22_p-1])
        # print(tokens_a)
#         print( e1, e2)
#         exit()

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 3
            _truncate_seq_pair(tokens_a, tokens_b,
                               max_seq_len - special_tokens_count)
        else:
            if use_entity_type:
                # if use_position:
                if example.e1_type:
                   text_b = e1 + '为' + entity2str[example.e1_type] + '，' + e2 + '为' + entity2str[example.e2_type] + '。'
                   if use_entity_position:
                       if e21_p > e12_p: #尾实体在后：
                           text_b += entity2str[example.e1_type] + '在前，' + entity2str[example.e2_type] + '在后。'
                       # else:
                       #     text_b += entity2str[example.e2_type] + '在前，' + entity2str[example.e1_type] +'在后。'
                   tokens_b = tokenizer.tokenize(text_b)
                   special_tokens_count = 3
                   _truncate_seq_pair(tokens_a, tokens_b,
                                      max_seq_len - special_tokens_count)
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            else:
                special_tokens_count = 2
                if len(tokens_a) > max_seq_len - special_tokens_count:
                    tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        tokens = [cls_token] + tokens
        e11_p +=1
        e12_p +=1
        e21_p +=1
        e22_p +=1
        if e21_p > e12_p:  #尾实体的第一个字在首实体的最后一个字之后，尾实体在后
            e_begin = e11_p
            e_end = e22_p
        else:     #尾实体在前
            e_begin = e21_p
            e_end = e12_p
            
        first_pos_ids = token_pos(tokens,e11_p,e12_p,max_seq_len)
        second_pos_ids = token_pos(tokens,e21_p,e22_p,max_seq_len)
        # e1 = ''.join(tokens[e11_p+1:e12_p-1])
   #      e2 = ''.join(tokens[e21_p+1:e22_p-1])
   #      print(tokens)
   #      print( e1, e2)
   #      exit()

        
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + \
            ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + \
            ([pad_token_segment_id] * padding_length)
        first_pos_ids = first_pos_ids + ([2*max_seq_len+1] * padding_length)
        second_pos_ids = second_pos_ids + ([2*max_seq_len+1] * padding_length)

        if use_entity_indicator:
            e1_mask = [0 for i in range(len(input_mask))]

            e2_mask = [0 for i in range(len(input_mask))]
            for i in range(e11_p, e12_p):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p):
                print(i,len(e2_mask))
                e2_mask[i] = 1
        
        e1_e2_mask = [0 for i in range(len(input_mask))]
        if use_cnn:
            for i in range(e_begin, e_end):
                e1_e2_mask[i] = 1
       
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        if output_mode == "classification":
            # label_id = label_map[example.label]
            label_id = int(example.label)
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
        
        if ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            print("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            if use_entity_indicator:
                print("e11_p: %s" % e11_p)
                print("e12_p: %s" % e12_p)
                print("e21_p: %s" % e21_p)
                print("e22_p: %s" % e22_p)
                print("e1_mask: %s" %
                            " ".join([str(x) for x in e1_mask]))
                print("e2_mask: %s" %
                            " ".join([str(x) for x in e2_mask]))
            print("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          e11_p=e11_p,
                          e12_p=e12_p,
                          e21_p=e21_p,
                          e22_p=e22_p,
                          e1_mask=e1_mask,
                          e2_mask=e2_mask,
                          e1_e2_mask=e1_e2_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          first_pos_ids=first_pos_ids,
                          second_pos_ids=second_pos_ids))
    # print()
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='micro'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


data_processors = {
    "semeval": SemEvalProcessor,
    "mrpc": MrpcProcessor,
    "cclue": CLUEProcessor,
    "zztj": ZZTJProcessor,
   }

output_modes = {
    "mrpc": "classification",
    "semeval": "classification",
    "cclue": "classification",
    "zztj": "classification",
}


