from torch.utils.data import Dataset 
from utils import read_json

class MengziT5Dataset(Dataset):
    def __init__(self, data: list):
        self.data = data 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]