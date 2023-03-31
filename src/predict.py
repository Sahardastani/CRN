import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from models.resnet_simclr import ResNetSimCLR
from models.feature_extractor import FeatureExtractor
from models.rnn import RNNModel, RNN
from datasets.ucf101 import FlowsDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--pre_trained_model', 
                    type=str, 
                    default='/home/sdastani/scratch/resnet18/checkpoint_0100.pth.tar', 
                    help='The directory of pre-trained model.')

parser.add_argument('--flows', 
                    type=str, 
                    default='/home/sdastani/projects/rrg-ebrahimi/sdastani/crn/FlowFrames',
                    help='The direcotry of videos.')
args = parser.parse_args()


# load the model from checkpoint
model = ResNetSimCLR(base_model='resnet18', out_dim=128)
torch.save({'model_state_dict': model.state_dict()}, args.pre_trained_model)
checkpoint = torch.load(args.pre_trained_model, map_location = device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
resnetmodel = FeatureExtractor(model.to(device))

# Define the pre-processing steps for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

frames_dataset = FlowsDataset(frames_dir=args.flows, transform=transform)
frames_dataloader = DataLoader(frames_dataset, batch_size=32, shuffle=True)

input_size = 128
hidden_size = 128
num_layers = 2
num_classes = 10
num_epochs = 10

rnnmodel = RNNModel(input_size, hidden_size, num_layers)#, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnnmodel.parameters(), lr=0.001)


extracted_features = {}
for i, batch in enumerate(frames_dataloader):
    frame = batch.to(device).float()
    with torch.no_grad():
        feature = resnetmodel(frame)
    breakpoint()
    # extracted_features[i] = feature.to("cpu")
    # del feature
    # torch.cuda.empty_cache()

    # Train the model
    for epoch in range(num_epochs):
        # Forward pass
        outputs = rnnmodel(feature)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

# Save the trained model
torch.save(rnnmodel.state_dict(), 'image_rnn_model.pth')

