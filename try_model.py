import sys
import os
import warnings
from model import CSRNet
#from utils import save_checkpoint
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import timeit
#import argparse
#import json
#import cv2
#import dataset
#import time
#from image import *
import PIL.Image as Image
from matplotlib import pyplot as plt
model = CSRNet()

#defining the model
#model = model.cuda()
# colab

#loading the trained weights
checkpoint = torch.load('0model_best.pth.tar',map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

# basic image
img_path = 'nom.jpeg'


transform=transforms.Compose([
transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]),
])

img = transform(Image.open(img_path).convert('RGB')) # .cuda()

# time it...
start = timeit.default_timer()
output = model(img.unsqueeze(0))
stop = timeit.default_timer()
execution_time = stop - start

print("Program Executed in "+ str(execution_time)) #It returns time in sec


print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
plt.imshow(temp) # ,cmap = c.jet)
plt.show()