  """
    [summary]
    --fold 뒤에 번호를 입력하면 해당 폴드만 inference를 진행합니다.
    python3 inference_fold.py --model_dir model_dir --fold fold_number --device device  
    
    default
      model_dir = '/best_model' (fold가 저장된 상위 디렉토리 위치)
      fold = 5
      device = 'cuda'

  """

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import os
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

def inference(model, tokenized_sent, device):
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
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후, tokenizing 합니다.
  """
  test_dataset, test_label = load_data(dataset_dir, train=False)
  test_label = list(map(int, test_label.values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(args):
  device = torch.device(args.device)
  print(device)

  # load tokenizer and insert special tokens
  MODEL_NAME = "klue/roberta-large"
  user_defined_symbols = ['[unused]']
  for i in range(1,10): user_defined_symbols.append(f'[UNK{i}]')
  for i in range(1,200): user_defined_symbols.append(f'[unused{i}]')
    
  special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenizer.add_special_tokens(special_tokens_dict)

  ## load my model 
  num_fold = args.fold

  MODEL_DIR = os.path.join(args.model_dir, f'{num_fold}') # model dir.

  model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
  model.resize_token_embeddings(tokenizer.vocab_size + len(user_defined_symbols))
  model.to(device)

  ## load test datset
  test_dataset_dir = "~/dataset/test/test_data.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)

  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론

  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv(f'./prediction/submission_{num_fold}.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.

  print('---- Finish! ----')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./best_model")
  parser.add_argument('--fold', type=int, default=1)
  parser.add_argument('--device', type=str, default='cuda:0')
  args = parser.parse_args()
  print(args)
  main(args)
  
