import numpy as np
import random
import json

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open("C:\\Users\munta\\.vscode\\ExtraModel\\intensts.json", 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# LOOPS THROUGH EACH SENTENCE IN OUR INTENSTS PATTERN
for intent in intents['intents']:
    tag = intent['tag']
    # ADD TAG TO TAGLIST
    tags.append(tag)
    for pattern in intent['patterns']:
        # TOKENIZE EACH SENTENCE IN WORDS
        w = tokenize(pattern)
        # ADD TO OUR WORD LIST
        all_words.extend(w)
        # ADD TO xy PAIR
        xy.append((w, tag))

# STEM AND LOWER EACH
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# REMOVE DIPLICATES AND SORT
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# CREATE TRAINING DATA
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: BAG OF WORDS FOR EACH pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss NEEDS ONLY CLASS LABEL, NOT one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# HYPER-PARAMETERS
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # SUPPORT INDEXING SUCH 
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # WE CAN CALL len(dataset) TO RETURN THE SIZE
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# LOSS AND OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TRAIN THE MODEL
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # FORWARD PASS
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # BACKWARD AND OPTIMIZE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')