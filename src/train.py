import os
import sys 
import cv2 
import time 
import yaml
import wandb
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

# Set config and device
config = yaml.load(open("./src/configs/config.yaml", "r"), Loader=yaml.FullLoader)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

wandb.init(
    # set the wandb project where this run will be logged
    project="agitation_detection",
    entity="crn_project",
    name="first_run",
    # track hyperparameters and update meta-data. This is really important to get right.
    config=config
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the dataloader function
def get_dataloader(root_dir, batch_size, sequence_length, transform):

    # get the dataset
    dataset = FrameDataset(root_dir, sequence_length, transform)
    
    # split the dataset into train and test
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # load the dataset
    train_loader = DataLoader(train_set, batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size, shuffle=False)

    return train_loader, val_loader

# Define dataloader
train_loader, val_loader = get_dataloader('/home/sdastani/scratch/new', 
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

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        features = torch.empty(data.shape[0], data.shape[1], 128)
        with torch.no_grad():
            for i, frame in enumerate(data):
                features[i] = cnn(frame.float())
        output, hidden = rnn(features.to(device))

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

        data, target = data.to(device), target.to(device)

        features = torch.empty(data.shape[0], data.shape[1], 128)
        with torch.no_grad():
            for i, frame in enumerate(data):
                features[i] = cnn(frame.float())
        output, hidden = rnn(features.to(device))

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



# train
for epoch in range(config['parameter']['epochs']):
    train_loss, train_time = train(resnetmodel, rnn, train_loader, optimizer, criterion)
    valid_loss, valid_time, out, target = validate(resnetmodel, rnn, val_loader, criterion)

    wandb.log({"train_loss": train_loss, "val_loss": valid_loss})

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