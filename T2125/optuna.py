import pickle as pickle
import os
import pandas as pd
import torch

import sklearn
from sklearn.model_selection import train_test_split  # train_valid 나누기 위해 추가
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, \
    TrainingArguments
from load_data import *


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
                  'org:product', 'per:title', 'org:alternate_names',
                  'per:employee_of', 'org:place_of_headquarters', 'per:product',
                  'org:number_of_employees/members', 'per:children',
                  'per:place_of_residence', 'per:alternate_names',
                  'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                  'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                  'org:member_of', 'per:parents', 'org:dissolved',
                  'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                  'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                  'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


###################
def my_hp_space(trial):  ##옵션을 줄 수 있어요
    return {
        "learning_rate": trial.suggest_float("learning_rate",5e-e, 3e-5, 2e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
    }


#################


model_name = 'klue/bert-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)
############
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 30
###########
train_dataset = load_data("../dataset/train/train.csv")
train_label = label_to_num(train_dataset['label'].values)
# trian_dev_split
train_dataset, dev_dataset, train_label, dev_label = train_test_split(train_dataset, train_label, test_size=0.1,
                                                                      shuffle=True, random_state=34)  # train_valid 나눔
#
# tokenizing dataset
tokenized_train = tokenized_dataset(train_dataset, tokenizer)
tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

# make dataset for pytorch.
RE_train_dataset = RE_Dataset(tokenized_train, train_label)
RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)


####################
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=config)  # search 하기위해서 model_init()사용


####################


# Evaluate during training and a bit more often than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = TrainingArguments(output_dir='./results',  # output directory
                                  save_total_limit=5,  # number of total save model.
                                  save_steps=500,  # model saving step.
                                  warmup_steps=500,  # number of warmup steps for learning rate scheduler
                                  weight_decay=0.01,  # strength of weight decay
                                  logging_dir='./logs',  # directory for storing logs
                                  logging_steps=100,  # log saving step.
                                  evaluation_strategy='steps',  # evaluation strategy to adopt during training
                                  eval_steps=500,  # evaluation step.
                                  load_best_model_at_end=True
                                  )
trainer = Trainer(
    args=training_args,
    train_dataset=RE_train_dataset,
    eval_dataset=RE_dev_dataset,
    model_init=model_init,  ######모델 대신 init
    compute_metrics=compute_metrics,
)

# Defaut objective is the sum of all metrics when metrics are provided, so we have to maximize it.
trainer.hyperparameter_search(direction="maximize", hp_space=my_hp_space,
                              n_trials=10)  ####direction minimize 가능 하고 n_trials번 시도