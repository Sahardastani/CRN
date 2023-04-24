import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class FrameDataset(Dataset):
    def __init__(self, root_dir, seq_length, transform=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.seq_length = seq_length
        self.transform = transform


    def __len__(self):
        # print('len:', len(self.videos))
        return len(self.videos)

    def __getitem__(self, index):
        video_dir = os.path.join(self.root_dir, self.videos[index])
        frames = os.listdir(video_dir)
        frames.sort() # sort the frames in ascending order
        video = []
        for frame_name in frames:
            frame_path = os.path.join(video_dir, frame_name)
            frame = Image.open(frame_path).convert('RGB')
            if self.transform is not None:
                frame = self.transform(frame)
            video.append(frame)
        return torch.stack(video[:self.seq_length]), self.videos[index].split('_')[1]