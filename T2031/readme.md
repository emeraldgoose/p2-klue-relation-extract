# 1. Preprocessing.py

> [Code Link](https://github.com/boostcampaitech2/klue-level2-nlp-10/blob/master/T2031/PreProcessing.py)

# 1.1 클래스 소개

전처리를 위한 클래스입니다.
**Preprocessing.py**는 두 가지 class로 구분됩니다. 

첫 번째 class는 `Row()` 입니다. 이는 다른 클래스인 `PreProcessing()`을 사용하기 위한 것으로, 깊은 이해를 하지 않으셔도 됩니다.

```python
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

        str_ = re.sub("·",",",str_)

        str_ = re.sub("[^a-zA-Z0-9가-힇ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥\"\-.,?!()《》 ]", "", str_) # remove special chars
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
            self.subject["end_idx"] = self.subject["start_idx"] + len(self.subject["word"])
            self.object["start_idx"] = self.subject["end_idx"] + len(part_b)
            self.object["end_idx"] = self.object["start_idx"] + len(self.object["word"])

        else:
            self.sentence = (part_a + self.object["word"] + part_b + self.subject["word"] + part_c)
            self.object["start_idx"] = len(part_a)
            self.object["end_idx"] = self.object["start_idx"] + len(self.object["word"])
            self.subject["start_idx"] = self.object["end_idx"] + len(part_b)
            self.subject["end_idx"] = self.subject["start_idx"] + len(self.subject["word"])

    def get_result(self):
        return [self.sentence, self.subject, self.object, self.label]
```

두 번째 클래스는 `PreProcessing()`입니다. csv파일의 경로를 입력으로 받습니다.

`_run_by_step`은 특정 index의 문장만 전처리하는 함수입니다.
`index`와 `original`를 입력으로 받으며, `original = True`로 하면 전처리 전, 후 레코드를 모두 반환합니다. 기본값은 `False`입니다.

`run`은 인스턴스 생성 시 입력받은 csv파일의 모든 레코드를 전처리하는 함수입니다.

`to_csv`는 전처리가 완료된 파일을 csv파일로 전환하는 함수입니다. 저장 경로를 반드시 입력하셔야 합니다.

```python
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
```

## 1.2 사용법

```python
from PreProcessing import PreProcessing

p = PreProcessing("csv파일경로") # PreProcessing instance를 생성합니다.
p.run() # 정규식을 실행합니다.
p.to_csv("저장경로")
```
