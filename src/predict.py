import time 
import yaml
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
from torchvision.transforms import transforms
from models.rnn import VideoRNN
from models.resnet_simclr import ResNetSimCLR
from models.feature_extractor import FeatureExtractor
from datasets.ucf101 import FrameDataset
from torch.utils.data import Dataset, DataLoader
from __init__ import top_dir, data_dir, configs_dir
import torch.optim as optim

# Set config and device
config = yaml.load(open("./src/configs/config.yaml", "r"), Loader=yaml.FullLoader)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the dataloader function
def get_dataloader(root_dir, batch_size, sequence_length, transform):
    dataset = FrameDataset(root_dir=data_dir(), seq_length = sequence_length, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# concatenate the CNN features and pass them through the RNN
def forward_rnn(frames):
    features = []
    for frame in frames:
        with torch.no_grad():
            feature = resnetmodel(frame.float())
        features.append(feature)
    features = torch.stack(features, dim=0)
    output = rnn(features)
    return output

# train function
def train(resnet, rnn, dataloader, optimizer, criterion):
    train_loss = 0.
    rnn.train()
    start = time.time()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        features = []
        for frame in data:
            with torch.no_grad():
                feature = resnet(frame.float())
            features.append(feature)
        features = torch.stack(features, dim=0)
        output = rnn(features)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    end = time.time()
    train_loss /= len(dataloader)
    train_time = end - start
    return train_loss, train_time

def validate(model, dataloader, criterion):
    val_loss = 0.
    model.eval()
    start = time.time()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        features = []
        for frame in data:
            with torch.no_grad():
                feature = resnet(frame.float())
            features.append(feature)
        features = torch.stack(features, dim=0)
        output = rnn(features)
        
        loss = criterion(output, target)
        val_loss += loss.item()

    end = time.time()
    val_loss /= len(dataloader)
    val_time = end - start
    return val_loss, val_time

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

# Define dataloader
dataloader = get_dataloader(data_dir(), config['parameter']['batch_size'], config['parameter']['sequence_length'], transform)

results = {
    "train_losses": [],
    "train_times":  [],
    "valid_losses": [],
    "valid_times":  []
}

# train
for epoch in range(config['parameter']['epochs']):
    train_loss, train_time = train(resnetmodel, rnn, dataloader, optimizer, criterion)

    print(f'Epoch {epoch+1}/{10}, Train Loss: {train_loss:.4f}, Train Time: {train_time:.3f}%')

    results["train_losses"].append(train_loss)
    results["train_times"].append(train_time)