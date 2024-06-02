import numpy as np
import os
import cv2

# Define a mapping between class numbers and animal names
class_mapping = {
  1:"Bull",
  2:"Camel",
  3:"Cattle",
  4:"Cheetah",
  5:"Deer",
  6:"Elephant",
  7:"Fox",
  8:"Giraffe",
  9:"Goat",
  10:"Goose",
  11:"Hippopotamus",
  12:"Horse",
  13:"Jaguar", 14:"Kangaroo",15:"Koala",16:"Leopard",17:"Lion",
  18:"Lynx",19:"Magpie",20:"Monkey",21:"Panda",22:"Pig",23:"Raccoon",
  24:"Red panda",25:"Rhinaceros",26:"Sheep",27:"Tiger",28:"Zebra"
}

# Define the YOLO model configuration and weights paths
weights_path = "C:/Users/Bala/OneDrive/Desktop/211FA20018/train/yolov3.weights"
configuration_path = 'C:/Users/Bala/OneDrive/Desktop/211FA20018/train/yolov3_custom.cfg'

# Load the YOLO model
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)

# Function to perform object detection and return the detected animal
def detect_animal(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Perform object detection with YOLO
    # ...

    # Determine the detected class number
    class_number = 28
    detected_animal = class_mapping.get(class_number, "Unknown")
    return detected_animal

# Call the detect_animal function with your image path
image_path = "C:/archive (2)/raw-img/cane/OIP--2z_zAuTMzgYM_KynUl9CQHaE7.jpeg"
detected_animal = detect_animal(image_path)
print(f"The detected animal is: {detected_animal}")
