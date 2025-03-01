import json
import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import gdown
import zipfile

class VietnameseSelfEvaluationDataset(Dataset):
    @classmethod()
    def download(cls, root = "./", sampling_rate = 8000):
        output_path = os.path.join(root,"vn_evaluate.zip")
        gdown.download(id="1yV5yJPRvxq9BLt38DqgsVLhzMeKiO5B9", output=output_path, quiet=False)
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(root)
        os.remove(output_path)
        return cls(root,sampling_rate)

    def __init__(self, root, sampling_rate = 8000):
        super().__init__()

        self.metadata = pd.read_csv(os.path.join(root,"metadata.csv"))
        self.emb_folder = os.path.join(root,"emb")
        self.mix_folder = os.path.join(root,"mix")
        self.ground_truth_folder = os.path.join(root,"ground_truth")
        self.sampling_rate = sampling_rate
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        mix_file, ground_truth_file, ref_emb_file, mix_emb_file = row['mix'], row['ground_truth'], row['ref_embedding'], row['mix_embedding']

        mix, r = torchaudio.load(os.path.join(self.mix_folder,mix_file))
        mix = torchaudio.functional.resample(mix.squeeze(),r,self.sampling_rate)

        ground_truth, r =  torchaudio.load(os.path.join(self.ground_truth_folder,ground_truth_file))
        ground_truth = torchaudio.functional.resample(ground_truth.squeeze(),r,self.sampling_rate)

        with open(os.path.join(self.emb_folder,ref_emb_file),mode='r') as f:
            emb = json.load(f)
        
        emb = torch.tensor(emb).float()

        return mix, ground_truth, emb