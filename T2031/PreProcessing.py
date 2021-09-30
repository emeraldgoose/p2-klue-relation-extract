import torch
import pandas as pd
import numpy as np
import re
import tqdm

class Row():

    def __init__(self, list_):
        self.list_ = list_ # [sentence, sub, obj, label]

        self.sentence = self.list_[0]
        self.subject = eval(self.list_[1]) # dict {word, start_idx, end_idx, type}
        self.object = eval(self.list_[2]) # dict {word, start_idx, end_idx, type}
        self.label = self.list_[3]
        self.flag_who_comes_faster = 0 # 0 : subject first, 1 : object_first

    def _get_splited_point(self):

        if self.subject["start_idx"] > self.object["start_idx"]:
            self.flag_who_comes_faster = 1

        if self.flag_who_comes_faster == 0:
            split_a = self.subject["start_idx"]
            split_b = self.subject["end_idx"]+1
            split_c = self.object["start_idx"]
            split_d = self.object["end_idx"]+1

        else:
            split_c = self.subject["start_idx"]
            split_d = self.subject["end_idx"]+1
            split_a = self.object["start_idx"]
            split_b = self.object["end_idx"]+1

        return split_a, split_b, split_c, split_d

    def _sub_run(self, str_):

        str_ = re.sub("[“‘’”\']","\"", str_) # unify different quotation marks
        str_ = re.sub("\"{2,}","\"", str_) # remove duplicated quotation marks

        str_ = re.sub("[\[{(（［｛〔]","(", str_) # unify different brackets
        str_ = re.sub("[\]})）］｝〕]",")", str_)
        str_ = re.sub("\({2,}","(", str_)
        str_ = re.sub("\){2,}",")", str_)

        str_ = re.sub("[〈《「『【]","《", str_) # unify different brackets
        str_ = re.sub("[〉》」』】]","》", str_)
        str_ = re.sub("《{2,}","《", str_)
        str_ = re.sub("》{2,}","》", str_)

        str_ = re.sub("·",", ",str_)

        str_ = re.sub("[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥\"\-.,?!()《》 ]", "", str_) # remove special chars
        str_ = re.sub("\.{2,}",".", str_) # multiple punkts to one. 
        str_ = re.sub(" {2,}"," ", str_) # multiple blanks to one. 

        return str_

    def run(self):
        
        if self.sentence.startswith("\""):
            self.sentence = re.sub("\"(.+)\"",r"\1", self.sentence).strip() # remove first and last quotation mark

        split_a, split_b, split_c, split_d = self._get_splited_point()

        part_a = self._sub_run(self.sentence[:split_a]).lstrip()
        part_b = self._sub_run(self.sentence[split_b:split_c])
        part_c = self._sub_run(self.sentence[split_d:]).rstrip()

        if self.flag_who_comes_faster == 0:

            self.sentence = (part_a + self.subject["word"] + part_b + self.object["word"] + part_c)
            self.subject["start_idx"] = len(part_a)
            self.subject["end_idx"] = self.subject["start_idx"] + len(self.subject["word"])-1
            self.object["start_idx"] = self.subject["end_idx"] + len(part_b) +1
            self.object["end_idx"] = self.object["start_idx"] + len(self.object["word"]) -1

        else:
            self.sentence = (part_a + self.object["word"] + part_b + self.subject["word"] + part_c)
            self.object["start_idx"] = len(part_a)
            self.object["end_idx"] = self.object["start_idx"] + len(self.object["word"]) -1
            self.subject["start_idx"] = self.object["end_idx"] + len(part_b) +1 
            self.subject["end_idx"] = self.subject["start_idx"] + len(self.subject["word"]) -1

    def get_result(self):
        return [self.sentence, self.subject, self.object, self.label]

class PreProcessing:
    
    def __init__(self, csv_path):
        self.values = pd.read_csv(csv_path, index_col = 0).values[:, :-1] # 출처를 제외한 모든 정보를 가져옴
        self.result = []

    def _run_by_step(self, index, original = False):

        row = self.values[index]
        row_instance = Row(row)

        row_instance.run()

        if original:
            return row, row_instance.get_result()
        else:
            return row_instance.get_result()

    def run(self):
        for i in tqdm.tqdm(range(len(self.values))):
            temp = self._run_by_step(i)
            self.result.append(temp)

    def to_csv(self, file_name):
        pd.DataFrame(self.result, columns = ["sentence", "subject_entity", "object_entity", "label"]).to_csv(file_name)
        
