import os
import cv2 as cv
import glob as gb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Model, layers, Sequential, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import callbacks, layers, Model
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_dir= 'train'
test_dir='test'

training_images_files_names_bull = len(os.listdir("train/Bull"))-1
training_images_files_names_cattle = len(os.listdir("train/Cattle"))-1
training_images_files_names_pig = len(os.listdir("train/Pig"))-1
training_images_files_names_horses = len(os.listdir("train/Horse"))-1
training_images_files_names_sheep = len(os.listdir("train/Sheep"))-1
training_images_files_names_goat = len(os.listdir("train/Goat"))-1
training_images_files_names_monkey = len(os.listdir("train/Monkey"))-1
training_images_files_names_giraffe = len(os.listdir("train/Giraffe"))-1
raining_images_files_names_elephant = len(os.listdir("train/Elephant"))-1
training_images_files_names_deer = len(os.listdir("train/Deer"))-1
# Calculate training data size
training_data_size = (
    training_images_files_names_bull + training_images_files_names_cattle +
    training_images_files_names_pig + training_images_files_names_horses +
    training_images_files_names_sheep + training_images_files_names_goat +
    training_images_files_names_monkey + training_images_files_names_giraffe +
    training_images_files_names_elephant + training_images_files_names_deer
)

# Calculate occurrences based on counts
occurrences = [
    training_images_files_names_bull * 3.9 / training_data_size,
    training_images_files_names_horses / training_data_size,
    training_images_files_names_goat * 1.95 / training_data_size,
    training_images_files_names_pig * 2.1 / training_data_size,
    training_images_files_names_sheep * 4 / training_data_size,
    training_images_files_names_cattle * 5.7 / training_data_size,
    training_images_files_names_monkey * 2 / training_data_size,
    training_images_files_names_giraffe * 1 / training_data_size,
    training_images_files_names_elephant * 2.53 / training_data_size,
    training_images_files_names_deer * 1.2 / training_data_size
    # add other animals based on their counts here
]
fig = plt.figure(figsize=[20,10])
animals_to_detect = ["Bull","Cattle","Pig","Horse","Sheep","Goat","Monkey","Giraffe","Elephant","Deer"]
os.mkdir("yolo")
os.mkdir("yolo/test")
os.mkdir("yolo/test/images")
os.mkdir("yolo/test/labels")
os.mkdir("yolo/train")
os.mkdir("yolo/train/images")
os.mkdir("yolo/train/labels")
size = (640,640)
for animal_specie in animals_to_detect:
    image_file_name = os.listdir(train_dir+"/"+animal_specie)
    for i in range(0,len(image_file_name)):
            if image_file_name[i] != "Label":
                img = cv2.imread(train_dir+"/"+animal_specie+"/"+image_file_name[i], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                cv2.imwrite("yolo/train/images/"+image_file_name[i], img) 

    image_file_name = os.listdir(test_dir+"/"+animal_specie)
    for i in range(0,len(image_file_name)):
            if image_file_name[i] != "Label":
                img = cv2.imread(test_dir+"/"+animal_specie+"/"+image_file_name[i], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                cv2.imwrite("yolo/test/images/"+image_file_name[i], img) 
animals_to_detect = ["Cattle","Pig","Horse","Sheep","Goat","Elephant","Deer", "Monkey","Bull","Giraffe"]
animals_encoding = {"Cattle":0,"Pig":1,"Horse":2,"Sheep":3,"Goat":4,"Elephant":5,"Deer":6,"Monkey":7,"Bull":8, "Giraffe":9}
def process_files(input_files_path,output_files_path):
    for animal_specie in animals_to_detect:
        txt_file_name = os.listdir(input_files_path+"/"+animal_specie+"/Label")
        for i in range(0,len(txt_file_name)):
                with open(input_files_path+"/"+animal_specie + "/Label/" + txt_file_name[i], "r") as source:
                       with open(output_files_path+"/"+ txt_file_name[i], "w") as destination :
                            image_file_name_no_ext = txt_file_name[i][0:len(txt_file_name[i])-4]
                            img = cv2.imread(input_files_path+"/"+animal_specie+"/"+image_file_name_no_ext+".jpg", cv2.IMREAD_COLOR)
                            height = img.shape[0]
                            width = img.shape[1]
                            for line in source:
                                        labeling_data = line.split()
                                        labeling_data[0] = animals_encoding[labeling_data[0]]
                                        xmin = float(labeling_data[1])
                                        ymin = float(labeling_data[2])
                                        xmax = float(labeling_data[3])
                                        ymax = float(labeling_data[4])
                                        cx = (xmin + xmax)/2.0/width
                                        cy = (ymin + ymax)/2.0/height
                                        box_width = (xmax - xmin)/width
                                        box_height = (ymax - ymin)/height
                                        destination.write(str(labeling_data[0])+" ")
                                        destination.write(str(cx)+" ")
                                        destination.write(str(cy)+" ")
                                        destination.write(str(box_width)+" ")
                                        destination.write(str(box_height)+"\n")

process_files("train","yolo/train/labels")
process_files("test","yolo/test/labels")
with open("data/animals.yaml", "w") as yaml_file:
    yaml_file.write("path: ../yolo  # train images (relative to 'path') 128 images"+"\n")
    yaml_file.write("train: train/images  # train images (relative to 'path') 128 images"+"\n")
    yaml_file.write("val: test/images  # val images (relative to 'path') 128 images"+"\n")
    yaml_file.write("names:"+"\n")
    yaml_file.write(" 0: Cattle"+"\n")
    yaml_file.write(" 1: Pig"+"\n")
    yaml_file.write(" 2: Horse"+"\n")
    yaml_file.write(" 3: Sheep"+"\n")
    yaml_file.write(" 4: Goat"+"\n")
 
    
    yaml_file.write(" 18: Elephant"+"\n")
    yaml_file.write(" 20: Deer"+"\n")


results_yolov5s = os.listdir("runs/train/exp")
img = mpimg.imread('runs/train/exp/results.png')
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread('runs/train/exp/train_batch0.jpg')
figure_size = 10
plt.figure(figsize=(figure_size,figure_size))
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread('runs/train/exp/train_batch1.jpg')
figure_size = 8
plt.figure(figsize=(figure_size,figure_size))
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread('runs/train/exp/train_batch2.jpg')
figure_size = 8
plt.figure(figsize=(figure_size,figure_size))
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread('runs/train/exp/F1_curve.png')
figure_size = 10
plt.figure(figsize=(figure_size,figure_size))
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread('runs/train/exp/PR_curve.png')
figure_size = 10
plt.figure(figsize=(figure_size,figure_size))
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread('runs/train/exp/P_curve.png')
figure_size = 10
plt.figure(figsize=(figure_size,figure_size))
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread('runs/train/exp/R_curve.png')
figure_size = 10
plt.figure(figsize=(figure_size,figure_size))
imgplot = plt.imshow(img)
plt.show()
import os 
os.chdir(r'/kaggle/working')