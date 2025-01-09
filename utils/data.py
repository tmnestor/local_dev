import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """Shared dataset class for loading model data"""
    def __init__(self, df, target_column):
        if df is None or df.empty:
            raise ValueError("Empty dataframe provided")
            
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        self.features = torch.FloatTensor(df.drop(target_column, axis=1).values)
        self.labels = torch.LongTensor(df[target_column].values)
        
        if self.features.nelement() == 0 or self.labels.nelement() == 0:
            raise ValueError("Failed to create data tensors")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
