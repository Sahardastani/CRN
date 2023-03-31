import torch
import torch.nn as nn
import torchvision.models as models

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

# load a pretrained CNN model
cnn_model = models.resnet18(pretrained=True)

# remove the last layer of the CNN model
cnn_model = nn.Sequential(*list(cnn_model.children())[:-1])

# freeze the parameters of the CNN model
for param in cnn_model.parameters():
    param.requires_grad = False



# create an instance of the VideoRNN module
input_size = 512 # assuming the output of the CNN model is 512
hidden_size = 256
num_layers = 2
num_classes = 10 # number of output classes
rnn = VideoRNN(input_size, hidden_size, num_layers, num_classes)

# concatenate the CNN features and pass them through the RNN
def forward_rnn(frames):
    features = []
    for frame in frames:
        # frame = frame.unsqueeze(0)
        # breakpoint()
        with torch.no_grad():
            feature = cnn_model(frame)
        feature = feature.squeeze(2).squeeze(2)
        features.append(feature)
    features = torch.stack(features, dim=0)
    logits = rnn(features)
    breakpoint()
    return logits

# generate some sample input data
batch_size = 20
sequence_length = 10
frame_size = (224, 224)
input_data = torch.randn(batch_size, sequence_length, 3, *frame_size)

# generate some sample target labels
target_labels = torch.randint(num_classes, size=(batch_size,))

# compute the logits and loss
logits = forward_rnn(input_data)
loss = nn.CrossEntropyLoss()(logits, target_labels)

# compute the accuracy
predicted_labels = torch.argmax(logits, dim=1)
accuracy = torch.mean((predicted_labels == target_labels).float())

breakpoint()