import torch
import pandas as pd
import numpy as np
import re
import os
import tqdm

# Pre processing

class PreProcessor():

    def __init__(self, csv_path):
        self.values = pd.read_csv(csv_path, index_col = 0).values[:, :-1] # 출처를 제외한 모든 정보를 가져옴
        self.result = []

    def _run_part(self, str_):

        str_ = re.sub("[“‘’”\"]","\'", str_) # unify different quotation marks
        str_ = re.sub("[\[{(（［｛〔]","(", str_) # unify different brackets
        str_ = re.sub("[\]})）］｝〕]",")", str_)
        str_ = re.sub("\({2,}","(", str_)
        str_ = re.sub("\){2,}",")", str_)

        str_ = re.sub("[〈《「『【]","《", str_) # unify different brackets
        str_ = re.sub("[〉》」』】]","》", str_)
        str_ = re.sub("《{2,}","《", str_)
        str_ = re.sub("》{2,}","》", str_)

        str_ = re.sub("·",", ",str_)

        str_ = re.sub("[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥\'\-~.,?!()《》 ]", "", str_) # remove special chars
        str_ = re.sub("\.{2,}",".", str_) # multiple punkts to one. 
        str_ = re.sub(" {2,}"," ", str_) # multiple blanks to one. 

        str_ = re.sub("\'{2,}","\'", str_) # remove duplicated quotation marks
        
        return str_

    def _tokenize(self, entity):

        type_ = entity["type"]
        word = entity["word"]

        return "[" + type_ + "]" + word + "[/" + type_ + "]"


    def _run_by_step(self, idx, original = False):

        sentence = self.values[idx, 0]
        sbj = eval(self.values[idx, 1]) # dict {word, start_idx, end_idx, type}
        obj = eval(self.values[idx, 2]) # dict {word, start_idx, end_idx, type}
        label = self.values[idx, 3]

        sub_first_flag = 1 if sbj["start_idx"] < obj['start_idx'] else 0 # if subject comes first 1 else 0

        # split the sentence
        if sub_first_flag:
            part_a = sentence[:sbj["start_idx"]]
            part_b = sentence[sbj['end_idx']+1 : obj['start_idx']]
            part_c = sentence[obj["end_idx"]+1 :]

        else:
            part_a = sentence[:obj["start_idx"]]
            part_b = sentence[obj['end_idx']+1 : sbj['start_idx']]
            part_c = sentence[sbj["end_idx"]+1 :]

        part_a = self._run_part(part_a).lstrip()
        part_b = self._run_part(part_b)
        part_c = self._run_part(part_c).rstrip()

        sub_token = self._tokenize(sbj)
        obj_token = self._tokenize(obj)

        # concatenate the parts
        if sub_first_flag:
            sentence = part_a + sub_token + part_b + obj_token + part_c
        else:
            sentence = part_a + obj_token + part_b + sub_token + part_c

        if original:
            return self.values[idx], [sentence, sbj, obj, label]
        else:
            return [sentence, sbj, obj, label]

    def run(self, save_csv):

        for record_idx in range(len(self.values)):
            record = self._run_by_step(record_idx)
            self.result.append(record)

        print("preprocessing done!")

        pd.DataFrame(self.result, columns = ["sentence", "sub_entity", "obj_entity", "label"]).to_csv(save_csv)
