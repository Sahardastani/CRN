import os
import cv2 
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define the dataset class
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, seq_length, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_length = seq_length

        # Get a list of all the video files in the directory
        self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.avi')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # Open the video file
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        video = cv2.VideoCapture(video_path)

        # Read each frame of the video and store them in a list
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:# or len(frames) >= 50:
                break
            frames.append(frame)

        # Convert the list of frames to a tensor
        video_tensor = torch.stack([transforms.ToTensor()(frame) for frame in frames])

        # Apply any specified transforms to the video tensor
        if self.transform:
            video_tensor = self.transform(video_tensor)

        # Close the video file and return the tensor
        video.release()
        return video_tensor[:self.seq_length], self.video_files[idx].split('_')[1]






# class FrameDataset(Dataset):
#     def __init__(self, root_dir, seq_length, transform=None):
#         self.root_dir = root_dir
#         self.videos = os.listdir(root_dir)
#         self.seq_length = seq_length
#         self.transform = transform


#     def __len__(self):
#         # print('len:', len(self.videos))
#         return len(self.videos)

#     def __getitem__(self, index):
#         video_dir = os.path.join(self.root_dir, self.videos[index])
#         frames = os.listdir(video_dir)
#         frames.sort() # sort the frames in ascending order
#         video = []
#         for frame_name in frames:
#             frame_path = os.path.join(video_dir, frame_name)
#             frame = Image.open(frame_path).convert('RGB')
#             if self.transform is not None:
#                 frame = self.transform(frame)
#             video.append(frame)
#         return torch.stack(video[:self.seq_length]), self.videos[index].split('_')[1]