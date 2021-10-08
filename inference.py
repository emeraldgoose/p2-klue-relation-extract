import os
import numpy as np
import pandas as pd
import argparse
import pickle as pickle

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_data import *


def inference(model, tokenized_sent, device):

    """ test dataset을 DataLoader로 만들어 준 후, batch_size로 나눠 model이 예측 합니다. """
    
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                # token_type_ids=data['token_type_ids'].to(device)
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


def num_to_label(label):

    """ 숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다. """
    
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def load_test_dataset(dataset_dir, tokenizer):
    """ test dataset을 불러온 후, tokenizing 합니다. """
    test_dataset, test_label = load_data(dataset_dir, train=False)
    test_label = list(map(int, test_label.values))
    
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset['id'], tokenized_test, test_label


def main(args):
    # device
    device = torch.device(args.device)
    print(f'device : {device}')

    MODEL_NAME = "klue/roberta-large"

    # load tokenizer and insert [UNK], [unused] tokens
    user_defined_symbols = ['[unused]']

    for i in range(1, 10):
        user_defined_symbols.append(f'[UNK{i}]')

    for i in range(1, 200):
        user_defined_symbols.append(f'[unused{i}]')

    special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens(special_tokens_dict)

    # Variables
    num_of_fold = args.fold
    pred = [0] * 7765
    prob = [[0]*30 for _ in range(7765)]

    for fold in range(1, num_of_fold+1):
        MODEL_DIR = os.path.join(args.model_dir, f'{fold}')

        # load model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.resize_token_embeddings(
            tokenizer.vocab_size + len(user_defined_symbols))
        model.to(device)

        # load test datset
        test_dataset_dir = "~/dataset/test/test_data.csv"
        test_id, test_dataset, test_label = load_test_dataset(
            test_dataset_dir, tokenizer)

        Re_test_dataset = RE_Dataset(test_dataset, test_label)

        # predict answer
        pred_answer, output_prob = inference(
            model, Re_test_dataset, device)  # model에서 class 추론

        pred = [x + y for x, y in zip(pred, pred_answer)]
        prob = [[x + y for x, y in zip(prob[i], output_prob[i])]
                for i in range(7765)]

    pred = [x/5 for x in pred] # 5-fold 평균

    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.
    output = pd.DataFrame(
        {'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    output.to_csv('./prediction/submission.csv', index=False)

    print('---- Finish! ----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default="./best_model")
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print(args)
    main(args)
