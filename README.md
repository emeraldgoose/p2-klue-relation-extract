# NLP10 Relation Extraction
Code of solution for label classification of **KLUE** **Relation Extraction** Data

## What is Relation Extraction?

**Relation Extraction** is a problem of predicting properties and relationships for words in a sentence.

```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며,
GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)

- input: sentence, subject_entity, object_entity의 정보를 입력으로 사용 합니다.
- output: relation 30개 중 하나를 예측한 pred_label, 그리고 30개 클래스 각각에 대해 예측한 확률 probs을 제출해야 합니다! class별 확률의 순서는 주어진 dictionary의 순서에 맡게 일치시켜 주시기 바랍니다.
```


# Getting Started    
## Requirements
- pandas==1.1.5
- scikit-learn~=0.24.1
- transformers==4.10.0
- omegaconf==2.1.1

## Installation
```
git clone https://github.com/pudae/kaggle-understanding-clouds.git
pip install -r requirements.txt
```

# Dataset

Two csv files are supplied for the competition. `train.csv` and `test.csv`

`train.csv` consists of 32,640 records and `test.csv` has 7,765 records, each of which are represented in the form of `[sentence, subject_entity, object_entity, label, source]`

The goal of the competition is to predict proper relations between subject entities and object entities in the sentences in `test.csv` and to calculate the probability that predicted relation belongs to each class.

The type of `subject_entity` and `object_entity` is a dictionary-form string, and explanation of the keys are like below:

- `word`: subject/object word
- `start_idx`: index where the word begins
- `end_idx`: index where the word ends
- `type`: type of the word
  - `ORG`: Organization
  - `PER`: Person
  - `DAT`: Date and time
  - `POH`: Other proper nouns
  - `NOH`: Other numerals 
  - `LOC`: Location

explanation of the `label` column is like below:

|labels|||||
|-|-|-|-|-|
|no_relation|org:dissolved|org:founded|org:place_of_headquarters|org:alternate_names|
|org_member_of|org:members|org:political/religious_affiliation|org:product|org;founded_by|
|org:top_members/employees|org:number_of_employees/members|per:date_of_birth|per:date_of_death|per:place_of_birth|
|per:place_of_death|per:place_of_residence|per:origin|per:employee_of|per:schools_attended|
|per:alternate_names|per:parents|per:children|per:siblings|per:spouse|
|per:other_family|per:colleagues|per:product|per:religion|per:title|

source : https://github.com/KLUE-benchmark/KLUE


## input data example

|id|sentence|subject_entity|object_entity|label|source|
|--|------|---|---|---|---|
|0|〈Something〉는 조지 해리슨이 쓰고 비틀즈가 1969년 앨범 《Abbey Road》에 담은 노래다.|{'word': '비틀즈', 'start_idx': 24, 'end_idx': 26, 'type': 'ORG'|{'word': '조지 해리슨', 'start_idx': 13, 'end_idx': 18, 'type': 'PER'}|no_relation|wikipedia|

## output data example
|id|pred_label|probs|
|--|--|--|
|0|org:product|\[5.7E-05,	2.3E-05,	4.4E-02,	9.5E-01,	2.1E-05,	7.0E-05,	3.7E-05,	3.8E-05,	1.3E-04,	2.1E-04,	1.4E-05,	5.4E-05,	4.4E-05,	5.4E-05,	1.3E-04,	6.2E-05,	5.5E-05,	1.4E-04,	3.2E-05,	1.2E-04,	3.9E-05,	5.6E-05,	6.9E-05,	1.5E-04,	8.4E-05,	8.6E-05,	1.1E-04,	1.1E-04,	5.8E-05,	4.9E-05]|



## Dataset folder path
```
train/
└─train.csv

test
└─test_data.csv
```

# Code Components 
```
├── best_model/
├── results/ 
├── prediction/
├── logs/
├── dict_label_to_num
├── dict_num_to_label
├── README.md
├── requirements.txt
├── config.json
├── inference.py
├── load_data.py
└── train.py
```
# Evaluation Metric
Same as KLUE-RE evaluation metric

## micro F1 score except no_relation class (considered first)
Micro F1 score is the harmonic average between micro-precision and micro-recall, which gives same importance to each samples, that results more weight on class with more samples.

$precision = \displaystyle\frac{TP}{TP + FP}$

$Recall = \displaystyle\frac{TP}{TP+FN}$

$F1\,score = 2\times \displaystyle\frac{Precision \times Recall}{Precision + Recall}$

## area under the precision-recall curve (AUPRC) for every class
x-axis represents Recall, y-axis represents Precision, and it measures score by calculating average AUPRC of every class. Useful metric for imbalance data.

# Model
Models are included like below
* [klue/roberta-large](https://huggingface.co/roberta-large)

If you want HyperParameter Fine Tuning, you can modify `config.json`

# Train
We train models **klue/roberta-large**. You can train using our train.py file.
```
python3 train.py --fold [num_of_fold]
```

# Inference
If you finish training model, you can create submission.csv file using inference.py. We give sample inference.py usage.
```
python3 inference.py --model_dir [model_dir] --fold [num_of_fold] --device [device]
```

# Performance

**LB score (public)**:
- micro f1 : 73.248, auprc : 71.119

**LB score (private)**: 
- micro f1 : 71.556, auprc : 73.898



# reference paper
* https://arxiv.org/pdf/2105.09680.pdf #klue dataset
* https://arxiv.org/pdf/1907.11692.pdf #RoBERTa
* https://arxiv.org/pdf/1901.11196.pdf #Easy Data Augmentation
