import torch
from torch.utils.data import Dataset
import pandas as pd

class TR_Dataset(Dataset):
    """[summary]
    Description:
        Dataset을 구성하기 위한 class
    Args:
        pair_dataset ([type]): [description]
    """
    def __init__(self, pair_dataset, labels):
        super(self, TR_Dataset).__init__()
        self.label = labels
        self.pair_dataset = pair_dataset

    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, index):
        pass

    def load_data(dataset_dir):
        """ csv 파일을 읽어 전처리된 DataFrame으로 반환 """
        df = pd.read_csv(dataset_dir)
        # dataset = preprocessing(df)
        # return dataset

    def preprocessing(dataframe):
        pass