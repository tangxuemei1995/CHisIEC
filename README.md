# 🏛️ CHisIEC

**CHisIEC: An Information Extraction Corpus for Ancient Chinese History**

This repository provides the datasets and code used in the paper *“CHisIEC: An Information Extraction Corpus for Ancient Chinese History”*.
It includes **Named Entity Recognition (NER)** and **Relation Extraction (RE)** resources for pre-modern Chinese historical texts.

---

## 📦 Data

### 🔤 NER

Path: `./data/ner/`

* Data is processed in **CoNLL format**.

### 🔗 RE

Path: `./data/re/`

* Data is stored in **JSON format**.

---

# 🧪 Code

## 1. 🏷️ NER Code

You can refer to our separate repository:
👉 [https://github.com/tangxuemei1995/AnChineseNERE](https://github.com/tangxuemei1995/AnChineseNERE)

---

## 2. 🔍 Relation Extraction Code

### 2.1 📘 BERT / RoBERTa Models (SikuBERT & SikuRoBERTa)

Directory: `code/bert_roberta_re/`

* Corresponds to Table 5 models in the paper
* Based on the method from
  **“Enriching Pre-trained Language Model with Entity Information for Relation Classification”**
  [https://arxiv.org/abs/1905.08284](https://arxiv.org/abs/1905.08284)
* Training settings located in:
  `code/bert_roberta_re/config.ini`

You may obtain **SikuBERT** and **SikuRoBERTa** from HuggingFace:

* Either download to a local directory
* Or load directly during training (if your environment allows Internet access)

---

### 2.2 🤖 ChatGLM2 (6B, P-tuning)

Steps:

1. Download the **alpaca2 instruction-tuned model** into the `glm2/` directory
2. Ensure all data is placed in `data/`
3. Modify training settings inside `train_coling.sh`

Run:

```bash
bash train_coling.sh          # fine-tune the model
bash evaluate_coling.sh       # generate generated_predictions.txt
python evaluate_re_coling.py  # evaluate
```

---

### 2.3 🐪 Alpaca2 (7B, LoRA)

Run:

```bash
bash run_sft_chapter.sh    # train model
bash merge_new_model.sh    # merge LoRA
bash run_test.sh           # inference
python evaluate.py         # evaluation
```

---

### ⚠️ Important

Before running any models, please ensure:

* Model paths are correct
* Data paths are set correctly
* Necessary dependencies and GPU environments are available

---

## 🖥️ Annotation Platform

The two datasets were annotated using:
👉 [https://wyd.pkudh.net/](https://wyd.pkudh.net/)
This platform was developed by the **Digital Humanities Research Centre, Peking University**.
It supports deep-learning–based annotation for arbitrary corpora.

---

## 📚 Citation

If you use CHisIEC, please cite:

```
@article{Tang_Deng_Su_Yang_Wang_2024,
  title={CHisIEC: An Information Extraction Corpus for Ancient Chinese History},
  url={http://arxiv.org/abs/2403.15088},
  note={arXiv:2403.15088 [cs]},
  number={arXiv:2403.15088},
  publisher={arXiv},
  author={Tang, Xuemei and Deng, Zekun and Su, Qi and Yang, Hao and Wang, Jun},
  year={2024},
  month=mar
}
```
