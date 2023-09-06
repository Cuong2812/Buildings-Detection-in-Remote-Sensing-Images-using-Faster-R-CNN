import geopandas as gpd
import os 
import cv2
# for col in df.columns:
#     print(feature['geometry']['coordinates'][0])
####################################################
import numpy as np
import json
import os
from PIL import Image, ImageDraw
import matplotlib.image as img
import matplotlib.pyplot as plt
import rasterio
import rioxarray

#Def below named geoJson_to_tif has been created by 
#me to obtain masked images.

geojsondir = 'buildings/geojson/'



def geoJson_to_tif(filename):
  fileNum = str(filename.replace('.geojson','').replace('buildings_AOI_5_Khartoum_img',''))  #In this training below, 'Shanghai' dataset from SpaceNet v2 has been used.
  print(filename)
  #To call Shanghai raster images and polygons, a directory has been created
  #in Google Drive for Google Colab.

  rasterfilenamepre = "RGB-PanSharpen_AOI_5_Khartoum_img"  

  #Location of raster images in .tiff format
  dirtif = "SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train/RGB-PanSharpen/"
  
  #Location of polygon files in .geojson format
  dirgjson = "SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train/geojson/buildings/"
  
  #Location of the place which created masks will held place.
  dirmasktif = "SN2_buildings_train_AOI_5_Khartoum/dataset/masks"
  
  try:
    # load in the geojson file
    with open(dirgjson + filename) as igj:
        data = json.load(igj)
    # if GDAL 3+
    crs = data["crs"]["properties"]["name"]
    # crs = "EPSG:4326" # if GDAL 2
    geoms = [feat["geometry"] for feat in data["features"]]

    #Create empty mask raster based on the input raster
    rds = rioxarray.open_rasterio(dirtif + rasterfilenamepre + fileNum + ".tif").isel(band=0)
    rds.values[:] = 1
    rds.rio.write_nodata(0, inplace=True)
    rds.rio.to_raster("SN2_buildings_train_AOI_5_Khartoum/dataset/masks/" + rasterfilenamepre + fileNum + ".tif", dtype="uint8")
    
    # clip the raster to the mask
    clipped = rds.rio.clip(geoms, crs, drop=False)
    clipped.rio.to_raster("SN2_buildings_train_AOI_5_Khartoum/dataset/masks/" + rasterfilenamepre + fileNum + ".tif", dtype="uint8")
  except:
    rds = rioxarray.open_rasterio(dirtif + rasterfilenamepre + fileNum + ".tif").isel(band=0)
    rds.values[:] = 0
    rds.rio.write_nodata(0, inplace=True)
    rds.rio.to_raster("SN2_buildings_train_AOI_5_Khartoum/dataset/masks/" + rasterfilenamepre + fileNum + ".tif", dtype="uint8")

for filename in sorted(os.listdir("SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train/geojson/buildings/")):
    #SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train/geojson/buildings/buildings_AOI_5_Khartoum_img1.geojson
    if filename.endswith(".geojson"): 
        print(os.path.join("SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train/geojson/buildings/", filename))
        #Create masks using the def above to the directory above.
        geoJson_to_tif(filename)
        continue
    else:
        continue