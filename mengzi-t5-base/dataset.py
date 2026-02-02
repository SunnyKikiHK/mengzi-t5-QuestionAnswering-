from torch.utils.data import Dataset 
from transformers import T5Tokenizer
from utils import exceed_length_threshold

MAX_DATASET_SIZE = 20000

class MengziT5Dataset(Dataset):
    def __init__(self, data: list, tokenizer: T5Tokenizer):
        self.data = self.load_data(data, tokenizer) 

    def load_data(self, data: list, tokenizer: T5Tokenizer):
        filter_num = 0
        filter_data = []
        for i, d in enumerate(data):
            if i >= MAX_DATASET_SIZE or exceed_length_threshold(d, tokenizer):
                filter_num += 1
                continue
            filter_data.append(d)
        print("Total data filtered away:", filter_num)
        return filter_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]