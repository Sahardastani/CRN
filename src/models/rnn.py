import torch 
import torch.nn as nn

# define the RNN module
class VideoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VideoRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x has shape (batch_size, sequence_length, input_size)
        out, hidden = self.rnn(x)
        # out has shape (batch_size, sequence_length, hidden_size)
        # hidden has shape (num_layers, batch_size, hidden_size)
        # we use the final hidden state for classification
        final_hidden = hidden[0][-1, :, :]
        logits = self.fc(final_hidden)
        return logits