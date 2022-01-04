from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(784, 512)
        
        # Second fully connected layer, 512-256 units 
        self.fc2 = nn.Linear(512, 256)
        
        # Third fully connected layer
        self.fc3 = nn.Linear(256, 128)
        
        # Fourth fully connected layer
        self.fc4 = nn.Linear(128, 64)
        
        # Output - one for each digit
        self.out = nn.Linear(64, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.dropout(self.relu(self.fc1(x))) # First, second, third and fourth layer
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.out(x)
        x = self.log_softmax(x)
        
        return x
