# https://jovian.ai/droste-benedikt/02-article-pytorch-multilabel-classification-v2/v/1?utm_source=embed#C2

#Import suppporting libraries

import tarfile
import urllib.request as urllib2
import os
from os import listdir
from os.path import isfile, join
import re
#Import deep learning libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms, models
import torchvision.models as models
#Import data analytics libraries
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns
#Import image visualization libraries
from PIL import *
from PIL import ImageFile
from PIL import Image
class color:
   RED = '\033[91m'
   BOLD = '\033[1m'
   END = '\033[0m'
def getting_data(url, path):
   print("Data extracted")
   data = urllib2.urlopen(url)

   tar_package = tarfile.open(name="C:/Users\Admin\Downloads\car_ims.tgz", mode='r:gz')
   print("Data extracted")
   tar_package.extractall(path)
   tar_package.close()
   return print("Data extracted and saved.")
def getting_metadata(url,filename):
  '''
  Downloading a metadata file from a specific url and save it to the disc.
  '''
  labels = urllib2.urlopen(url)
  file = open(filename, 'wb')
  file.write(labels.read())
  file.close()
  return print("Metadata downloaded and saved.")
class MetaParsing():
  '''
  Class for parsing image and meta-data for the Stanford car dataset to create a custom dataset.
  path: The filepah to the metadata in .mat format.
  *args: Accepts dictionaries with self-created labels which will be extracted from the metadata (e.g. {0: 'Audi', 1: 'BMW', 3: 'Other').
  year: Can be defined to create two classes (<=year and later).
  '''
  def __init__(self,path,*args,year=None):
    self.mat = scipy.io.loadmat(path)
    self.year = year
    self.args = args
    self.annotations = np.transpose(self.mat['annotations'])
    #Extracting the file name for each sample
    self.file_names = [annotation[0][0][0].split("/")[-1] for annotation in self.annotations]
    #Extracting the index of the label for each sample
    self.label_indices = [annotation[0][5][0][0] for annotation in self.annotations]
    #Extracting the car names as strings
    self.car_names = [x[0] for x in self.mat['class_names'][0]]
    #Create a list with car names instead of label indices for each sample
    self.translated_car_names = [self.car_names[x-1] for x in self.label_indices]
  def brand_types(self,base_dict, x):
    y = list(base_dict.keys())[-1]
    for k,v in base_dict.items():
      if v in x: y=k
    return y
  def parsing(self):
    result = []
    for arg in self.args:
      temp_list = [self.brand_types(arg,x) for x in self.translated_car_names]
      result.append(temp_list)
    if self.year != None:
      years_list = [0 if int(x.split(" ")[-1]) <= self.year else 1 for x in self.translated_car_names]
      result.append(years_list)
    brands = [x.split(" ")[0] for x in self.translated_car_names]
    return result, self.file_names, self.translated_car_names
def count_classes(base_dict, base_list):
  for i in range(len(list(base_dict.keys()))):
    print("{}: {}".format(base_dict[i], str(base_list.count(i))))

#System settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['WANDB_CONSOLE'] = 'off'
#Coloring for print outputs
#getting_data("http://ai.stanford.edu/~jkrause/car196/car_ims.tgz","/content/carimages")

#getting_metadata("http://ai.stanford.edu/~jkrause/car196/cars_annos.mat","car_metadata.mat")
brand_dict = {0: 'Audi', 1: 'BMW', 2: 'Chevrolet', 3: 'Dodge', 4: 'Ford', 5: 'Other'}
vehicle_types_dict = {0: 'Convertible', 1: 'Coupe', 2: 'SUV', 3: 'Van', 4: 'Other'}

results, file_names, translated_car_names = MetaParsing("car_metadata.mat",brand_dict,vehicle_types_dict,year=2009).parsing()
len(results)
count_classes(brand_dict,results[0])
count_classes(vehicle_types_dict,results[1])