import torch.nn as nn
import torch.nn.functional as F


#Creating model
# CNN1 - 3x3 filters, 7 convolution layers

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 16, 3)  # 68x318
#         self.conv2 = nn.Conv2d(16, 32, 3) # 66x316
#         self.pool1 = nn.MaxPool2d(2, 2) # 33x158

#         self.conv3 = nn.Conv2d(32, 64, 3) # 31x156
#         self.conv4 = nn.Conv2d(64, 64, 3) # 29x154
#         self.pool2 = nn.MaxPool2d(2, 2) # 14x77

#         self.conv5 = nn.Conv2d(64, 64, 3) # 12x75
#         self.conv6 = nn.Conv2d(64, 64, 3) # 10x73
#         self.conv7 = nn.Conv2d(64, 64, 3) # 8x71
#         self.pool3 = nn.MaxPool2d(2, 2) # 4x35

        
#         self.fc1 = nn.Linear(4*35*64, 128)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(128, 64)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc3 = nn.Linear(64, 1)
        

#     def forward(self, x):
  
#         x = F.relu(self.conv1(x))
#         x = self.pool1(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = self.pool2(F.relu(self.conv4(x)))
#         x = F.elu(self.conv5(x))
#         x = F.elu(self.conv6(x))
#         x = self.pool3(F.relu(self.conv7(x)))

#         x = x.view(-1, 4*35*64)
#         x = F.relu(self.fc1(x))
#         #print(x.shape)
#         x = self.dropout1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return x




# # Creating model
#CNN2 - 6 convolution layers, 3x3 filters, padding-same
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 16, 3, padding = 'same')  # 70x320
#         self.conv2 = nn.Conv2d(16, 32, 3, padding = 'same') # 70x320
#         self.pool1 = nn.MaxPool2d(2, 2) # 35x160

#         self.conv3 = nn.Conv2d(32, 64, 3, padding = 'same') # 35x160
#         self.conv4 = nn.Conv2d(64, 64, 3, padding = 'same') # 35x160
#         self.pool2 = nn.MaxPool2d(2, 2) # 17x80

#         self.conv5 = nn.Conv2d(64, 64, 3, padding = 'same') # 17x80
#         self.conv6 = nn.Conv2d(64, 64, 3, padding = 'same') # 17x80
#         self.pool3 = nn.MaxPool2d(2, 2) # 8x40

        
#         self.fc1 = nn.Linear(8*40*64, 128)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(128, 64)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc3 = nn.Linear(64, 1)
        

#     def forward(self, x):
  
#         x = F.relu(self.conv1(x))
#         x = self.pool1(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = self.pool2(F.relu(self.conv4(x)))
#         x = F.relu(self.conv5(x))
#         x = self.pool3(F.relu(self.conv6(x)))

#         x = x.view(-1, 8*40*64)
#         x = F.relu(self.fc1(x))
#         #print(x.shape)
#         x = self.dropout1(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return x



# CNN3 - 3 convolution layers, 3x3 filters, padding-same
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 32, 3, padding = 'same')  # 70x320
#         self.pool1 = nn.MaxPool2d(2, 2) # 35x160

#         self.conv2 = nn.Conv2d(32, 64, 3, padding = 'same') # 35x160
#         self.pool2 = nn.MaxPool2d(2, 2) # 17x80

#         self.conv3 = nn.Conv2d(64, 128, 3, padding = 'same') # 17x80
#         self.pool3 = nn.MaxPool2d(2, 2) # 8x40
#         self.dropout1 = nn.Dropout2d(0.25)

        
#         self.fc1 = nn.Linear(8*40*128, 512)
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(512, 1)
        

#     def forward(self, x):
  
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.dropout1(x)

#         x = x.view(-1, 8*40*128)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x

# CNN4 - 5 convolution layers, 5x5 filters
#class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 32, 5, padding = 'same')  # 70x320
#         self.pool1 = nn.MaxPool2d(2, 2) # 35x160

#         self.conv2 = nn.Conv2d(32, 64, 5, padding = 'same') # 35x160
#         self.pool2 = nn.MaxPool2d(2, 2) # 17x80

#         self.conv3 = nn.Conv2d(64, 128, 5, padding = 'same') # 17x80
#         self.pool3 = nn.MaxPool2d(2, 2) # 8x40
        
#         self.conv4 = nn.Conv2d(128, 128, 5, padding = 'same') # 8x40
#         self.pool4 = nn.MaxPool2d(2, 2) # 4x20
        
#         self.conv5 = nn.Conv2d(128, 128, 5, padding = 'same') # 4x20
#         self.pool5 = nn.MaxPool2d(2, 2) # 2x10
        
#         self.dropout1 = nn.Dropout2d(0.25)

        
#         self.fc1 = nn.Linear(2*10*128, 512)
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(512, 1)
        

#     def forward(self, x):
  
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.pool4(F.relu(self.conv4(x)))
#         x = self.pool5(F.relu(self.conv5(x)))
#         x = self.dropout1(x)

#         x = x.view(-1, 2*10*128)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x

# CNN5 - 5 convolution layers, 7x7 and 5x5 filters
# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 32, 7, padding = 'same')  # 70x320
#         self.pool1 = nn.MaxPool2d(2, 2) # 35x160

#         self.conv2 = nn.Conv2d(32, 64, 7, padding = 'same') # 35x160
#         self.pool2 = nn.MaxPool2d(2, 2) # 17x80

#         self.conv3 = nn.Conv2d(64, 128, 7, padding = 'same') # 17x80
#         self.pool3 = nn.MaxPool2d(2, 2) # 8x40
        
#         self.conv4 = nn.Conv2d(128, 128, 5, padding = 'same') # 8x40
#         self.pool4 = nn.MaxPool2d(2, 2) # 4x20
        
#         self.conv5 = nn.Conv2d(128, 128, 5, padding = 'same') # 4x20
#         self.pool5 = nn.MaxPool2d(2, 2) # 2x10
        
#         self.dropout1 = nn.Dropout2d(0.25)

        
#         self.fc1 = nn.Linear(2*10*128, 512)
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(512, 1)
        

#     def forward(self, x):
  
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.pool4(F.relu(self.conv4(x)))
#         x = self.pool5(F.relu(self.conv5(x)))
#         x = self.dropout1(x)

#         x = x.view(-1, 2*10*128)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x

# Creating model
# CNN6 - 3 convolution layers, 7x7 filters

#class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 16, 7, 2)  # 32x157
#         self.conv2 = nn.Conv2d(16, 32, 7, 2) # 13x76
#         self.conv3 = nn.Conv2d(32, 64, 7, 2) # 4x35
#         self.pool1 = nn.MaxPool2d(2, 2) # 2x17
#         self.dropout1 = nn.Dropout2d(0.25)
        
#         self.fc1 = nn.Linear(2*17*64, 512)
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(512, 1)
        

#     def forward(self, x):
  
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool1(F.relu(self.conv3(x)))
#         x = self.dropout1(x)

#         x = x.view(-1, 2*17*64)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x

# Creating model
# CNN7 - 3 convolution layers, 3x3 filters, YUV image

#class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 128x128x3
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 16, 3, padding = 'same')  # 70x320
#         self.pool1 = nn.MaxPool2d(2, 2) # 35x160

#         self.conv2 = nn.Conv2d(16, 32, 3, padding = 'same') # 35x160
#         self.pool2 = nn.MaxPool2d(2, 2) # 17x80

#         self.conv3 = nn.Conv2d(32, 64, 3, padding = 'same') # 17x80
#         self.pool3 = nn.MaxPool2d(2, 2) # 8x40
#         self.dropout1 = nn.Dropout2d(0.25)

        
#         self.fc1 = nn.Linear(8*40*64, 512)
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(512, 1)
        

#     def forward(self, x):
  
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.dropout1(x)

#         x = x.view(-1, 8*40*64)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x


# Creating model
# CNN8 - 4 convolution layers, 3x3 filters, YUV image

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 16, 3, padding = 'same')  # 70x320
#         self.pool1 = nn.MaxPool2d(2, 2) # 35x160

#         self.conv2 = nn.Conv2d(16, 32, 3, padding = 'same') # 35x160
#         self.pool2 = nn.MaxPool2d(2, 2) # 17x80

#         self.conv3 = nn.Conv2d(32, 64, 3, padding = 'same') # 17x80
#         self.pool3 = nn.MaxPool2d(2, 2) # 8x40
        
#         self.conv4 = nn.Conv2d(64, 64, 3, padding = 'same') # 8x40
#         self.pool4 = nn.MaxPool2d(2, 2) # 4x20
#         self.dropout1 = nn.Dropout2d(0.25)

        
#         self.fc1 = nn.Linear(4*20*64, 512)
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(512, 1)
        

#     def forward(self, x):
  
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.pool4(F.relu(self.conv4(x)))
#         x = self.dropout1(x)

#         x = x.view(-1, 4*20*64)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x

# Creating model
# CNN9 - 4 convolution layers, 3x3 and 5x5 filters, YUV image

# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 16, 3, padding = 'same')  # 70x320
#         self.pool1 = nn.MaxPool2d(2, 2) # 35x160

#         self.conv2 = nn.Conv2d(16, 32, 3, padding = 'same') # 35x160
#         self.pool2 = nn.MaxPool2d(2, 2) # 17x80

#         self.conv3 = nn.Conv2d(32, 64, 3, padding = 'same') # 17x80
#         self.pool3 = nn.MaxPool2d(2, 2) # 8x40
        
#         self.conv4 = nn.Conv2d(64, 64, 5, 2) # 2x18, strides=2
#         self.pool4 = nn.MaxPool2d(2, 2) # 1x9
#         self.dropout1 = nn.Dropout2d(0.25)

        
#         self.fc1 = nn.Linear(1*9*64, 256)
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(256, 1)
        

#     def forward(self, x):
  
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.pool4(F.relu(self.conv4(x)))
#         x = self.dropout1(x)

#         x = x.view(-1, 1*9*64)
#         x = F.relu(self.fc1(x))
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         return x

# # Creating model
# # CNN10 - 4 convolution layers, 3x3 and 5x5 filters, 2FCs, YUV image

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Input image size = 70x320x3

        self.conv1 = nn.Conv2d(3, 16, 3, padding = 'same')  # 70x320
        self.pool1 = nn.MaxPool2d(2, 2) # 35x160

        self.conv2 = nn.Conv2d(16, 32, 3, padding = 'same') # 35x160
        self.pool2 = nn.MaxPool2d(2, 2) # 17x80

        self.conv3 = nn.Conv2d(32, 64, 3, padding = 'same') # 17x80
        self.pool3 = nn.MaxPool2d(2, 2) # 8x40
        
        self.conv4 = nn.Conv2d(64, 64, 5, 2) # 2x18, strides=2
        self.pool4 = nn.MaxPool2d(2, 2) # 1x9
        self.dropout1 = nn.Dropout2d(0.25)

        
        self.fc1 = nn.Linear(1*9*64, 256)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc3 = nn.Linear(64, 1)
        

    def forward(self, x):
  
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.dropout1(x)

        x = x.view(-1, 1*9*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x



# Creating model
# CNN11 - 4 convolution layers, 5x5 and 3x3 filters, 2FCs, YUV image

#class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 16, 5, 2)  # 33x158
#         self.pool1 = nn.MaxPool2d(2, 2) # 16x79

#         self.conv2 = nn.Conv2d(16, 32, 3, padding = 'same') # 16x79
#         self.pool2 = nn.MaxPool2d(2, 2) # 8x39

#         self.conv3 = nn.Conv2d(32, 64, 3, padding = 'same') # 8x39
#         self.pool3 = nn.MaxPool2d(2, 2) # 4x19
        
#         self.conv4 = nn.Conv2d(64, 64, 3, padding = 'same') # 4x19
#         self.pool4 = nn.MaxPool2d(2, 2) # 2x9
#         self.dropout1 = nn.Dropout2d(0.25)

        
#         self.fc1 = nn.Linear(2*9*64, 256)
#         self.fc2 = nn.Linear(256, 64)
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.fc3 = nn.Linear(64, 1)
        

#     def forward(self, x):
  
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.pool4(F.relu(self.conv4(x)))
#         x = self.dropout1(x)

#         x = x.view(-1, 2*9*64)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return x


# Creating model
# CNN12 - 4 convolution layers, 5x5 and 3x3 filters, 2FCs, YUV image

#class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # Input image size = 70x320x3

#         self.conv1 = nn.Conv2d(3, 16, 5, 2)  # 33x158
#         self.conv2 = nn.Conv2d(16, 32, 5, 2) # 15x77
#         self.pool1 = nn.MaxPool2d(2, 2) # 7x38

#         self.conv3 = nn.Conv2d(32, 64, 3) # 5x36       
#         self.conv4 = nn.Conv2d(64, 64, 3) # 3x34
#         self.pool2 = nn.MaxPool2d(2, 2) # 1x17
#         self.dropout1 = nn.Dropout2d(0.25)

        
#         self.fc1 = nn.Linear(1*17*64, 256)
#         self.fc2 = nn.Linear(256, 64)
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.fc3 = nn.Linear(64, 1)
        

#     def forward(self, x):
  
#         x = F.relu(self.conv1(x))
#         x = self.pool1(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = self.pool2(F.relu(self.conv4(x)))
#         x = self.dropout1(x)

#         x = x.view(-1, 1*17*64)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#         return x



# Define model
print("==> Initialize model ...")
model = ConvNet()
print("==> Initialize model done ...")