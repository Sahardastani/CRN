import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

class FlowsDataset(Dataset):
    def __init__(self, frames_dir, transform=None):
        self.frames_dir = frames_dir
        self.frames = os.listdir(frames_dir)
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_path = os.path.join(self.frames_dir, self.frames[idx])
        frame = Image.open(frame_path).convert("RGB")
        if self.transform:
            frame = self.transform(frame)
        return frame
