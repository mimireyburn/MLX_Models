import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train = datasets.MNIST("", train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True) 

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # define kernel size = 2*padding + 1
        self.conv1 = nn.Conv2d(1, 32, 5, padding = 2) # input image, output, kernel size, padding
        self.conv2 = nn.Conv2d(32, 64, 5, padding = 2) # zero-padding on bth sides of input


        self.fc1 = nn.Linear(64*7*7, 128) # arbitrary use of 128 neurons 
        self.fc2 = nn.Linear(128, 10) # reduced to 10 outputs 

    def convs(self, x): 
        # in 32 28x28s pooling cuts it in half to reduce dimensions
        # 32 14x14s
        # 64 7x7s 

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # run conv net then put through rect linear activation and pool 
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        return x 

    def forward(self, x): # This bit runs stuff
        x = self.convs(x) 
        x = x.view(-1, 64*7*7) # 64*7*7 is the length of the 1d vector
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

net = Net()

# Set up an optimiser 

optimizer = optim.Adam(net.parameters(), lr = 0.001)

Epochs = 3

for epoch in range(Epochs): 
    for data in trainset: 
        X, y = data
        net.zero_grad()                         # sets the gradient stored in each variable of our network to zero. If we didn’t use this command, then at each iteration autograd will keep adding the new value of the gradient to the previous one
        output = net.forward(X)                 # We compute the output using the forward method.  The method .view transforms our 2 dimensional tensor input (a 28 by 28 matrix) to a 1 dimensional one (a 784 vector). 
        loss = F.nll_loss(output,y)             # This sets up the loss function. Since we are working with a classifier, we use a cross entropy loss function
        loss.backward()                         # computes the gradient with respect to the loss function over each parameter of the network (again, without the command at line 46, this gradient would accumulate over each iteration leading to an incorrect training)
        optimizer.step()                        # updates the parameters of our network according to the optimization algorithm and the gradient stored within each variable
    
    print(loss)

# test if training was successful

correct = 0
total = 0

with torch.no_grad():                           # network won't update the gradient stored in each variable during the test session, as there is no loss to be minimised
    for data in testset: 
        X,y = data
        output = net.forward(X)
        for idx, i in enumerate(output): 
            if torch.argmax(i) == y[idx]:
                correct += 1 
            total += 1

print("accuracy: ", round(correct/total, 3))