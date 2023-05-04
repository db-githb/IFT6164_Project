from common_imports import Dataset

class AdvDataset(Dataset):
    def __init__(self, adv_x, adv_y):
        super().__init__()
        self.adv_x= adv_x
        self.adv_y= adv_y
    
    def __len__(self):
        return self.adv_x.shape[0]
    
    def __getitem__(self, idx):
        return self.adv_x[idx], self.adv_y[idx]