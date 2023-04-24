import torch 
import torch.nn as nn
import yaml
import numpy as np

# define the RNN module
class VideoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VideoRNN, self).__init__()

        # Defining some parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Defining the layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, hidden = self.rnn(x)
        final_hidden = hidden[-1, :]
        out = self.fc(final_hidden)
        return out, final_hidden

        # x has shape (batch_size, sequence_length, input_size)
        # out has shape (batch_size, sequence_length, hidden_size)
        # hidden has shape (num_layers, batch_size, hidden_size)
        # we use the final hidden state for classification


# # Set config and device
# config = yaml.load(open("./src/configs/config.yaml", "r"), Loader=yaml.FullLoader)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# rnn = VideoRNN(input_size = config['parameter']['input_size'], 
#                 hidden_size = config['parameter']['hidden_size'], 
#                 num_layers = config['parameter']['num_layers'], 
#                 num_classes = config['parameter']['num_classes'])


# x = torch.rand([5,20,128])
# out = rnn(x)