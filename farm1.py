



import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
# Example usage
train_data_dir = './train'
test_image_path = "test/Horse/3a80fd5cc3e2eec4.jpg"

# Constants
img_width, img_height = 224, 224
batch_size = 32

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load ResNet50 model with pre-trained weights (excluding top layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Custom classifier for our dataset
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=0)  # Adjust epochs as needed

# Load and preprocess the test image
test_image = image.load_img(test_image_path, target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255.

# Get predictions for the test image
predictions = model.predict(test_image)
predicted_class = np.argmax(predictions)

# Mapping class index to animal category (you might need your own mapping)
class_mapping = {0: 'Cattle', 1: 'Pig', 2: 'Horse',3:'Goat',4:"Giraffe" ,5:"Horse",6:'Monkey',7:"Pig",8:"Sheep",9:'Elephant'}

predicted_animal = class_mapping[predicted_class]
print(f"Detected animal: {predicted_animal}")
