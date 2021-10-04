import pickle as pickle
import os
from re import X
import pandas as pd
import torch

import sklearn
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import SchedulerType
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
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c)
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


def train_one_fold(fold, train_dataset, eval_dataset, tokenizer, added_token_num, MODEL_NAME, device):
    torch.cuda.empty_cache()

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config)
    model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
    model.to(device)

    # train arguments & trainer
    training_args = TrainingArguments(
        output_dir=f'./results/{fold}',  # output directory
        save_total_limit=2,              # number of total save model.
        save_steps=500,                  # model saving step.
        num_train_epochs=10,              # total number of training epochs
        learning_rate=5e-5,              # learning_rate
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        weight_decay=1e-6,               # strength of weight decay
        evaluation_strategy='steps',     # evaluation strategy to adopt during training
        eval_steps=500,                  # evaluation step.
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=eval_dataset,           # evaluation dataset
        compute_metrics=compute_metrics      # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained(f'./best_model/{fold}')
    print(f'------{fold} finished------')


def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load model and tokenizer
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # insert speical tokens (unk, unused token)
    user_defined_symbols = ['[unused]']
    for i in range(1,10):
        user_defined_symbols.append(f'[UNK{i}]')
    for i in range(1,100):
        user_defined_symbols.append(f'[unused{i}]')
    
    special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
    tokenizer.add_special_tokens(special_tokens_dict)
    print(tokenizer.all_special_tokens)

    # load dataset
    train_dataset, train_label = load_data(
        "~/dataset/train/train.csv", train=True)

    # trian_dev_split
    n_splits = 5
    stratifiedKFold = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=42)

    for fold_index, (train_index, eval_index) in enumerate(stratifiedKFold.split(X=train_dataset, y=train_label)):
        print('Train start {}/{} Fold'.format(fold_index+1, n_splits))
        # fold_dataset, eval_dataset
        fold_dataset = train_dataset.iloc[train_index]
        fold_label = [train_label[i] for i in train_index]

        eval_dataset = train_dataset.iloc[eval_index]
        eval_label = [train_label[i] for i in eval_index]

        # tokenizing dataset
        tokenized_train = tokenized_dataset(fold_dataset, tokenizer)
        tokenized_dev = tokenized_dataset(eval_dataset, tokenizer)

        # make dataset for pytorch.
        re_train_dataset = RE_Dataset(tokenized_train, fold_label)
        re_eval_dataset = RE_Dataset(tokenized_dev, eval_label)

        train_one_fold(fold_index+1, re_train_dataset,
                       re_eval_dataset, tokenizer, len(user_defined_symbols), MODEL_NAME, device)

    print('------Finsh train------')


def main():
    train()


if __name__ == '__main__':
    main()
