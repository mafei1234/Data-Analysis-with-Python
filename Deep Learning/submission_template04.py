import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()

        # ВАШ КОД ЗДЕСЬ
        # определите слои сети
        import torch
        from torch.autograd import Variable
        import torch.nn as nn
        import torch.nn.functional as F
        class ConvNet(nn.Module):
          self.pool1=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(5,5))
          
       
        self.pool1= nn.Conv2d(in_channels=3,out_channels=3,kernel_size=(5,5))
        self.pool1 = nn.Maxpool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=5,kernel_size=(3,3))
        self.pool2 = nn.Maxpool2d(kernel_size=(2,2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(180,50)
        self.fc2 = nn.Linear(50,10)



    def forward(self, input):
        # размерность х ~ [64, 3, 32, 32]

        # ВАШ КОД ЗДЕСЬ
        # реализуйте forward pass сети
        x=self.pool1(F.relu(self.conv1(input)))


        x=self.pool2(F.relu(self.conv2(input)))
        x=self.fc2(self.fc1(x))
        return x
       
        
    
        
   
    
        ...

    
    def forward(self, x):
        ...

def create_model():
    return ConvNet()
