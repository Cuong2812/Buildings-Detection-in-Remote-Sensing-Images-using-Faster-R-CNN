import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose, Normalize, PILToTensor
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from PIL import ImageFilter, Image
import albumentations as A
import torchvision.transforms.functional as F
from skimage import io, transform
import cv2
import os
import random
import shutil
import json
import geopandas as gpd
from torchvision.ops import box_convert
from gbbox import LineString

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Load an image
image = Image.open('s2Dataset/buildings/3band/3band_AOI_1_RIO_img6790.tif')

# Preprocess the image
transform = T.Compose([T.ToTensor()])
input_image = transform(image).unsqueeze(0)

# Predict on the input image
predictions = model(input_image)

# Print the predicted bounding boxes and labels
predictions =(predictions[0]['boxes']).detach().numpy()


visualize_bbox('s2Dataset/buildings/3band/3band_AOI_1_RIO_img6790.tif', predictions)

#6790, 6348 6874