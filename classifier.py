import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Classifier(nn.Module):
    # TODO: implement me
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 254, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(254),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(254, 127, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(127),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.25))
        self.layer3 = nn.Sequential(
            nn.Conv2d(127, 125, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(125),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(125, 62, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(62),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.25))
        self.layer5 = nn.Sequential(
            nn.Conv2d(62, 60, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(60, 30, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(0.25))
        self.fc = nn.Sequential(
            nn.Linear(28*28*30, 512),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(512, NUM_CLASSES))
        
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
