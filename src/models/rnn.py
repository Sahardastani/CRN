import torch 
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        
    def forward(self, x, hx=None):
        batch_size = x.size(0)
        
        # If hx is not provided, initialize hidden state with zeros
        if hx is None:
            hx = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Reshape x to a 3D tensor
        x = x.unsqueeze(1)
        
        # Pass input and hidden state through RNN
        out, hx = self.rnn(x, hx)
        
        # Reshape output back to a 2D tensor
        out = out.squeeze(1)
        
        return out, hx