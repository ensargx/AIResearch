# -*- coding: utf-8 -*-
"""
MNIST example with PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# examine the data
print(train_data.describe())

class MNIST(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Input: MNIST dataset
        Output: 10 classes (0-9)
        """
        super(MNIST, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.softmax(out)
        return out
    
# Hyperparameters
input_size = 28*28
hidden_size = 16
output_size = 10

# Initialize the model
model = MNIST(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 16
batch_size = 100
learning_rate = 0.001

# Train the model
print('Training the model...')
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        # Get mini batch
        batch_x = train_data.iloc[i:i+batch_size, 1:].values
        batch_y = train_data.iloc[i:i+batch_size, 0].values
        
        # Convert to torch tensors
        batch_x = torch.from_numpy(batch_x).float()
        batch_y = torch.from_numpy(batch_y).long()
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #if (i//batch_size) % 100 == 0:
            #print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i//batch_size, len(train_data)//batch_size, loss.item()))
        tamami = len(train_data)//batch_size
        biten = i//batch_size
        oran = int(biten/tamami * 50)
        kalan = 50 - oran - 1
        print('Epoch [{}/{}]\t[{}>{}]\tLoss: {:.4f}'.format(epoch+1, num_epochs, '='*oran, ' '*kalan, loss.item()), end='\r')
    #print('Epoch [{}/{}],\t Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    print()

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for i in range(0, len(test_data), batch_size):
        batch_x = test_data.iloc[i:i+batch_size, 1:].values
        batch_y = test_data.iloc[i:i+batch_size, 0].values
        
        batch_x = torch.from_numpy(batch_x).float()
        batch_y = torch.from_numpy(batch_y).long()
        
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

def check_image_by_hand(id):
    """
    Print the image and the model's prediction with the given id
    """
    image = test_data.iloc[id, 1:].values
    image = torch.from_numpy(image).float()
    image = image.view(1, 784)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    print('Prediction: {}'.format(predicted.item()))
    image = image.view(28, 28)
    # Plot the image by using matplotlib with label 'Prediction: {}'
    plt.imshow(image)
    plt.title('Prediction: {}'.format(predicted.item()))
    plt.show()


# Save the model checkpoint
print('Saving the model as \'mnist-model.ckpt\'..')
torch.save(model.state_dict(), 'mnist-model.ckpt')

# Check the image by hand
print('Check the image by hand, press q to quit')
num_examples = len(test_data)
prompt = 'Enter the id of the image (0-{}): '.format(num_examples-1)
while True:
    id = input(prompt)
    if id == 'q':
        print('Exitting the program...')
        break
    if not id.isdigit():
        print('Please enter a number or press q to quit')
        continue
    if int(id) < 0 or int(id) >= num_examples:
        print('Please enter a number between 0 and {}'.format(num_examples-1))
        continue
    id = int(id)
    check_image_by_hand(id)