import os
import sys 
import cv2 
import time 
import yaml
import argparse 
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans, kmeans_predict

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.rnn import VideoRNN
from models.resnet_simclr import ResNetSimCLR
from models.feature_extractor import FeatureExtractor

from datasets.ucf101 import VideoDataset, FrameDataset
from __init__ import top_dir, data_dir, configs_dir

MEAN = torch.tensor((0.485 * 255, 0.456 * 255, 0.406 * 255))
STD = torch.tensor((0.229 * 255, 0.224 * 255, 0.225 * 255))

# Set config and device
config = yaml.load(open("/home/sdastani/projects/rrg-ebrahimi/sdastani/CRN/src/configs/config.yaml", "r"), Loader=yaml.FullLoader)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the dataloader function
def get_dataloader(root_dir, batch_size, sequence_length, transform):

    # get the dataset
    dataset = VideoDataset(root_dir, sequence_length, transform)
    
    # split the dataset into train and test
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # load the dataset
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Define dataloader
train_loader, val_loader = get_dataloader('/home/sdastani/scratch/ucf101/UCF101', 
                                            config['parameter']['batch_size'], 
                                            config['parameter']['sequence_length'], 
                                            transform)


# load the model from checkpoint
model = ResNetSimCLR(base_model='resnet18', out_dim=128)
torch.save({'model_state_dict': model.state_dict()}, config['checkpoint']['simclr'])
checkpoint = torch.load(config['checkpoint']['simclr'], map_location = device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
resnetmodel = FeatureExtractor(model.to(device))

# create an instance of the VideoRNN module
rnn = VideoRNN(input_size = config['parameter']['input_size'], 
                hidden_size = config['parameter']['hidden_size'], 
                num_layers = config['parameter']['num_layers'], 
                num_classes = config['parameter']['num_classes']).to(device)

# Define optimizer and loss
optimizer = optim.Adam(rnn.parameters())
criterion = nn.CrossEntropyLoss()

# train function
def train(cnn, rnn, dataloader, optimizer, criterion):
    train_loss = 0.
    rnn.train()
    start = time.time()

    for batch_idx, (data, target) in enumerate(dataloader):
        start1 = time.time()

        # Create a dictionary that maps each unique element to a unique integer value
        unique_elements = list(set(target))
        element_to_int = {element: i for i, element in enumerate(unique_elements)}
        # Use a list comprehension to replace each element in the tuple with its corresponding integer value
        target = torch.tensor([element_to_int[element] for element in target])

        btch = []
        for i, batch in enumerate(data):
            seqlength = []
            for j in range(batch.shape[0]):
                cur_frame = data[0][j].permute(1,2,0)
                pre_frame = data[0][j-1].permute(1,2,0)
                input1 = torch.from_numpy(np.array((cur_frame - MEAN) / STD).transpose((2, 0, 1))).float().unsqueeze(0).cuda()
                input2 = torch.from_numpy(np.array((pre_frame - MEAN) / STD).transpose((2, 0, 1))).float().unsqueeze(0).cuda()
                flownet = FlowModel(args=None, pretrained='/home/sdastani/projects/rrg-ebrahimi/sdastani/flownet2-pytorch/FlowNet2_checkpoint.pth.tar', save_flow=True)
                flownet.cuda()
                flow, image, loss = flownet(input1, input2)
                image = transform(image.float())
                seqlength.append(image)
            length = torch.stack(seqlength, dim=0)
            btch.append(length)
        final = torch.stack(btch, dim=0)

        final, target = final.to(device), target.to(device)

        optimizer.zero_grad()

        features = []
        for frame in final:
            with torch.no_grad():
                feature = cnn(frame.float())
            features.append(feature)
        features = torch.stack(features, dim=0)
        output, hidden = rnn(features)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        end1 = time.time()
        print('train_loss:', train_loss, 'train_time', end1-start1, f'--------- {batch_idx}/{len(dataloader)}')
    end = time.time()
    train_loss /= len(dataloader)
    train_time = end - start
    return train_loss, train_time

# Validation function
def validate(cnn, rnn, dataloader, criterion):
    val_loss = 0.
    rnn.eval()
    start = time.time()
    outk = []
    targetk = []

    for batch_idx, (data, target) in enumerate(dataloader):

        # Create a dictionary that maps each unique element to a unique integer value
        unique_elements = list(set(target))
        element_to_int = {element: i for i, element in enumerate(unique_elements)}
        # Use a list comprehension to replace each element in the tuple with its corresponding integer value
        target = torch.tensor([element_to_int[element] for element in target])

        btch = []
        for i, batch in enumerate(data):
            seqlength = []
            # print(f'batch {i} is being processed ...')
            for j in range(batch.shape[0]):
                cur_frame = data[0][j].permute(1,2,0)
                pre_frame = data[0][j-1].permute(1,2,0)
                input1 = torch.from_numpy(np.array((cur_frame - MEAN) / STD).transpose((2, 0, 1))).float().unsqueeze(0).cuda()
                input2 = torch.from_numpy(np.array((pre_frame - MEAN) / STD).transpose((2, 0, 1))).float().unsqueeze(0).cuda()
                flownet = FlowModel(args=None, pretrained='/home/sdastani/projects/rrg-ebrahimi/sdastani/flownet2-pytorch/FlowNet2_checkpoint.pth.tar', save_flow=True)
                flownet.cuda()
                flow, image, loss = flownet(input1, input2)
                image = transform(image.float())
                seqlength.append(image)
            length = torch.stack(seqlength, dim=0)
            btch.append(length)
        final = torch.stack(btch, dim=0)

        final, target = final.to(device), target.to(device)

        features = []
        for frame in final:
            with torch.no_grad():
                feature = cnn(frame.float())
            features.append(feature)
        features = torch.stack(features, dim=0)
        output, hidden = rnn(features)

        outk.append(hidden)
        targetk.append(target)

        loss = criterion(output, target)
        val_loss += loss.item()

    end = time.time()
    val_loss /= len(dataloader)
    val_time = end - start
    out = torch.cat(outk, dim=0)
    target = torch.cat(targetk, dim=0)
    return val_loss, val_time, out, target

results = {
    "train_losses": [],
    "train_times":  [],
    "valid_losses": [],
    "valid_times":  []
}

# Get the current directory of predict.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the "flownet" directory to the module search path
flownet_dir = os.path.join(current_dir, "../flownet2")
sys.path.append(flownet_dir)


from ops.resample2d import Resample2d
from components.flownet2 import FlowNet2
from components.flowlib import flow_to_image
from tools.flownet_test import FlowModel 

# train
for epoch in range(config['parameter']['epochs']):
    train_loss, train_time = train(resnetmodel, rnn, train_loader, optimizer, criterion)
    valid_loss, valid_time, out, target = validate(resnetmodel, rnn, val_loader, criterion)

    print(f"Epoch: {epoch}")
    print(f'Train Loss: {train_loss:.4f} | Train Time: {train_time:.3f}')
    print(f'Validation Loss: {valid_loss:.4f} | Validation Time: {valid_time:.3f}')

    results["train_losses"].append(train_loss)
    results["train_times"].append(train_time)
    results["valid_losses"].append(valid_loss)
    results["valid_times"].append(valid_time)

# plot train losses
plt.plot(np.cumsum(results["train_times"]), results["train_losses"], color='blue', label='Training loss')
plt.plot(np.cumsum(results["valid_times"]), results["valid_losses"], color='orange', label='Validation loss')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Training and validation loss over time')
plt.legend()
plt.savefig('train.png')

# test
test_loss, test_time, out, target = validate(resnetmodel, rnn, train_loader, criterion)

# k-means
cluster_ids_x, cluster_centers = kmeans(X=out, num_clusters=config['parameter']['num_classes'], distance='euclidean', device=device)

# accuracy
c = cluster_ids_x == target.to("cpu")
accuracy = int((torch.sum(c.int().view(c.shape) == 1) / c.shape[0]).item() * 100)
print('accuracy is:', accuracy, '%')


# # more data
# y = np.random.randn(5, dims) / 6
# y = torch.from_numpy(y)

# # predict cluster ids for y
# cluster_ids_y = kmeans_predict(y, cluster_centers, 'euclidean', device=device)