import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose, Normalize, PILToTensor
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

# set the paths for the source directory containing the TIF images and masks, and the destination directory for the split data
dest_dir = "buildings/3bandData/splitData"

# create the directories for the split data
train_dir = os.path.join(dest_dir, "train")
train_images_dir = os.path.join(train_dir, "images")
train_masks_dir = os.path.join(train_dir, "masks")
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)

val_dir = os.path.join(dest_dir, "val")
val_images_dir = os.path.join(val_dir, "images")
val_masks_dir = os.path.join(val_dir, "masks")
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)

test_dir = os.path.join(dest_dir, "test")
test_images_dir = os.path.join(test_dir, "images")
test_masks_dir = os.path.join(test_dir, "masks")
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)


src_dir = "buildings/3bandData"
# set the fraction of data to use for validation and test sets
val_frac = 0.1
test_frac = 0.1
buildings = []
# for i in os.listdir(src_dir+'/masksFolder'):
#     with Image.open(src_dir+'/masksFolder/'+i) as src:
#         new = np.asarray(src)
#         if (len(np.unique(new)) > 1):
#             buildings.append(i.replace('mask',''))

geojson = 'buildings/3bandData/geojson'
for i in os.listdir(geojson):
    with open(geojson +'/'+i) as data:
        data = json.load(data)
        if(data['features']):
            buildings.append('3band_AOI_1_RIO_img'+i[17:-8]+'.tif')
print(len(buildings))

# get a list of all the TIF files in the source directory
tif_files = buildings

# shuffle the list of TIF files
random.shuffle(tif_files)

# split the data into training, validation, and test sets
num_tif_files = len(tif_files)
num_val_files = int(val_frac * num_tif_files)
num_test_files = int(test_frac * num_tif_files)
num_train_files = num_tif_files - num_val_files - num_test_files

train_files = tif_files[:num_train_files]
val_files = tif_files[num_train_files:num_train_files+num_val_files]
test_files = tif_files[num_train_files+num_val_files:]

# move the TIF images and masks into the appropriate directories for the training set
i = 0
for file in train_files:
    i += 1
    src_path = os.path.join(src_dir+'/3band - Copy/', file)
    mask_file = file.replace(".tif", ".geojson").replace("3band","Geo")
    mask_path = os.path.join(src_dir+'/geojson', mask_file)
    
    train_images_dest_path = os.path.join(train_images_dir, file)
    train_masks_dest_path = os.path.join(train_masks_dir, mask_file)
    
    shutil.copy(src_path, train_images_dest_path)
    shutil.copy(mask_path, train_masks_dest_path)
    print(i)

# move the TIF images and masks into the appropriate directories for the validation set
for file in val_files:
    src_path = os.path.join(src_dir+'/3band - Copy/', file)
    mask_file = file.replace(".tif", ".geojson").replace("3band","Geo")
    mask_path = os.path.join(src_dir+'/geojson', mask_file)
    
    val_images_dest_path = os.path.join(val_images_dir, file)
    val_masks_dest_path = os.path.join(val_masks_dir, mask_file)
    
    shutil.copy(src_path, val_images_dest_path)
    shutil.copy(mask_path, val_masks_dest_path)

# move the TIF images and masks into the appropriate directories for the test set
for file in test_files:
    src_path = os.path.join(src_dir+'/3band - Copy/', file)
    mask_file = file.replace(".tif", ".geojson").replace("3band","Geo")
    mask_path = os.path.join(src_dir+'/geojson', mask_file)
    
    test_images_dest_path = os.path.join(test_images_dir, file)
    test_masks_dest_path = os.path.join(test_masks_dir, mask_file)
    
    shutil.copy(src_path, test_images_dest_path)
    shutil.copy(mask_path, test_masks_dest_path)

# Define the transform to apply to the images
