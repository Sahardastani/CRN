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
        return torch.stack(video[:self.seq_length]), torch.tensor(index)

# # transform
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# # Define the dataloader function
# def get_dataloader(root_dir, batch_size, sequence_length, transform):
#     dataset = FrameDataset(root_dir=root_dir, seq_length = sequence_length, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     return dataloader
# dataloader = get_dataloader('/home/sdastani/projects/rrg-ebrahimi/sdastani/CRN/data', 2, 20, transform)

# for batch_idx, (data, target) in enumerate(dataloader):
#     breakpoint()