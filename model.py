import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, drop_prob):
        
        super(DecoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        self.init_weights()

    
    def forward(self, features, captions):
        
        embeddings = torch.cat((features.unsqueeze(1), self.embedding(captions[:,:-1])), 1)
        x, hidden_state = self.lstm(embeddings)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)          
            outputs = self.fc(hiddens.squeeze(1))         
            _, predicted = outputs.max(1)                       
            sampled_ids.append(predicted.item())
            inputs = self.embedding(predicted)                      
            inputs = inputs.unsqueeze(1)   
            # Return if <end> is predicted
            if predicted.item() == 1:
                return sampled_ids
        return sampled_ids
    
    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        initrange = 0.1
        
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)
