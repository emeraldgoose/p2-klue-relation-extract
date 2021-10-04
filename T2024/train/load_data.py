import pickle as pickle
import os
import pandas as pd
import torch
from tqdm import tqdm


class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def label_to_num(label):
    """ label을 number로 변환 """
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    sub_token = '[SUB]'
    obj_token = '[OBJ]'
    sep = '[SEP]'

    for index, (s, e01, e02) in tqdm(enumerate(zip(dataset['sentence'], dataset['subject_entity'], dataset['object_entity']))):
        e01 = eval(e01); e02 = eval(e02)
        sub_word = e01['word']; e01_start = e01['start_idx']; e01_end = e01['end_idx']+1; e01_type = e01['type']
        obj_word = e02['word']; e02_start = e02['start_idx']; e02_end = e02['end_idx']+1; e02_type = e02['type']
        
        dataset['sentence'][index] = sub_word + sep + e01_type + sep + obj_word + sep + e02_type + sep + s

    return dataset


def load_data(dataset_dir, train=True):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    dataset = pd.read_csv(dataset_dir)
    if train:
        label = label_to_num(dataset['label'].values)
    else:
        label = dataset['label']
    dataset = dataset.drop(['label', 'source'], axis=1)

    # insert [obj], [/obj] tokens
    print('-'*6, ' insert special tokens ', '-'*6)
    dataset = preprocessing_dataset(dataset)
    print('-'*6, ' finish insert tokens ', '-'*6)
    return dataset, label


def tokenized_dataset(dataset, tokenizer):
    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=200,
        add_special_tokens=True
    )
    return tokenized_sentences
