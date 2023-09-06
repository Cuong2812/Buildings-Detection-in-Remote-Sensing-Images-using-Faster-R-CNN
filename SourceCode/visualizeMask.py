import rasterio
from PIL import Image
import cv2
from rasterio.plot import reshape_as_image
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import os
from PIL import Image
import tifffile
import geopandas as gpd

newFile = gpd.read_file('C:/Users/Hai Nguyen/Documents/SN2_buildings_train_AOI_5_Khartoum/dataset/train/json/buildings_AOI_5_Khartoum_img2.geojson')
for item in newFile['Shape_Area']:
    print(item)

with rasterio.open('buildings/testFolder/tif/3band_AOI_1_RIO_img4250.tif') as f:
    data = f.read()
    data= (reshape_as_image(data))
    plt.imshow(data)

with rasterio.open('buildings/testFolder/mask/3band_AOI_1_RIO_img4250mask.tif') as f: 
    data = f.read()
    data = reshape_as_image(data)
    plt.imshow(data, cmap ="Greys", alpha = 0.5)
    plt.show()

# i = rasterio.open("testFolder\ROIs0000_test_s2_0_p4.tif")
import os
import json
imageArr = []
maskArr = []
image_dir = 'buildings/testFolder/tif/'
mask_dir = 'buildings/testFolder/mask/'
img_name = '3band_AOI_1_RIO_img4250.tif'
import geopandas as gpd
import os
id = []
counter = 0
for file in os.listdir('buildings/3bandData/geojson'):
  print(counter)
  gdf = gpd.read_file('buildings/3bandData/geojson/'+file)
  if( len(gdf) > 0 ):
    for item in gdf['Shape_Area']:
      if(item == 1):
          break
    counter+=1
print(id)


with open('C:/Users/Hai Nguyen/Documents/s2Dataset/buildings/3bandData/geojson/Geo_AOI_1_RIO_img1010.geojson') as f:
    data = json.load(f)
    if(data['features']):
        print(data['features'])

with Image.open('buildings/3bandData/masksFolder/3band_AOI_1_RIO_img1010mask.tif') as src:
        new = np.asarray(src)
        print(len(np.unique(new)) == 1)
        plt.imshow(src)
        plt.show()


for i in os.listdir('buildings/3bandData\splitData/train/masks'):
    print(i)
    with Image.open('buildings/3bandData/splitData/train/masks/'+i) as src:
        plt.imshow(src)
        plt.show()

import geopandas as gpd #To read geojson file

# geo = gpd.read_file('buildings_AOI_5_Khartoum_img105.geojson')
# print(geo['geometry'][0])
import json
import os
with open('C:/Users/Hai Nguyen/Documents/SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train/geojson/buildings/buildings_AOI_5_Khartoum_img1.geojson') as f:
    data = json.load(f)
    print(data)
    if(data['features']):
        print("asdf")

buildings = []
geojson = 'buildings/3bandData/geojson'
for i in os.listdir(geojson):
    with open(geojson +'/'+i) as data:
        data = json.load(data)
        if(data['features']):
            buildings.append(i)
print(len(buildings))

    



import tensorflow as tf

# Replace filename with the name of your event file
filename = "buildings/log_dir/base_line/0/train/events.out.tfevents.1680039652.NTQH.15652.0.v2"

# Create a summary iterator for the event file
for e in tf.compat.v1.train.summary_iterator(filename):
    for v in e.summary.value:
        if v.tag == 'loss':
            print(tf.make_ndarray(v.tensor))



img = tf.keras.preprocessing.image.load_img('buildings/trainDataNew/tifFolder/3band_AOI_1_RIO_img1027_aug0.tif', target_size = (403,438))

plt.imshow(img)
img = tf.keras.preprocessing.image.load_img('buildings/trainDataNew/masksFolder/3band_AOI_1_RIO_img1027_aug0.tif', target_size = (403,438))
plt.imshow(img, cmap = "Greys", alpha = 0.5)
plt.show()




with rasterio.open('buildings/trainNewData/tifFolder/3band_AOI_1_RIO_img1027_aug0.tif') as dataset:
    image = dataset.read()
    image = reshape_as_image(image)
    plt.imshow(image)

with rasterio.open('buildings/trainNewData/masksFolder/3band_AOI_1_RIO_img1027_aug0.tif') as dataset:
    image1 = dataset.read()
    image1 = reshape_as_image(image1)
    plt.imshow(image1, cmap="Greys",alpha=0.5)
    plt.show()





for filename in os.listdir("buildings/testFolder/mask"):
    with rasterio.open('buildings/testFolder/tif/'+filename.replace('mask','')) as dataset:
        image = dataset.read()
        image = tf.image.resize(image, (128, 128), method="nearest")
        if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
            image = tf.image.flip_left_right(image)
        image = tf.cast(image, tf.float32) / 255.0
    imageArr.append(image)

    with rasterio.open('buildings/testFolder/mask/'+filename) as dataset:
        mask = dataset.read()
        mask = tf.image.resize(mask, (128, 128), method="nearest")
        if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
            mask = tf.image.flip_left_right(mask)
        mask = tf.cast(mask, tf.float32) / 255.0
    maskArr.append(mask)

print(len(imageArr))
print(len(maskArr))

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
   
with rasterio.open('buildings/testFolder/tif/'+filename.replace('mask','')) as dataset:
    image = dataset.read()
    image = tf.image.resize(image, (128, 128), method="nearest")
    if tf.random.uniform(()) > 0.5:
    # Random flipping of the image and mask
        image = tf.image.flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
imageArr.append(image)

with rasterio.open('buildings/testFolder/mask/'+filename) as dataset:
    mask = dataset.read()
    mask = tf.image.resize(mask, (128, 128), method="nearest")
    if tf.random.uniform(()) > 0.5:
    # Random flipping of the image and mask
        mask = tf.image.flip_left_right(mask)
    mask = tf.cast(mask, tf.float32) / 255.0
maskArr.append(mask)



    



# image = np.stack([bands[2], bands[1], bands[0]], axis=2)
# image = cv2.normalize(image, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# # Resize the image to a fixed size that is compatible with the Fast R-CNN model
# image = tf.image.resize(image, (800, 800))


# greyscale = (i[2] + i[1] + i[0]) / 3
