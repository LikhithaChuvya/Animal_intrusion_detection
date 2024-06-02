import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import pygame

# Function to play audio
def play_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# Define a dictionary mapping animal names to audio files
animal_audio = {
    "Bull": "scream.mp3",
    "Camel": "blank.mp3",
    "Cattle": "thunder.mp3",
    "Cheetah": "blank.mp3",
    "Deer": "fireworks.mp3",
    "Elephant": "tiger.mp3",
    "Fox": "scream.mp3",
    "Giraffe": "blank.mp3",
    "Goat": "scream.mp3",
    "Goose": "dog.mp3",
    "Hippopotamus": "blank.mp3",
    "Horse": "tiger.mp3",
    "Jaguar": "blank.mp3",
    "Kangaroo": "metalclang.mp3",
    "Lion": "metalclang.mp3",
    "Monkey": "fireworks.mp3",
    "Pig": "thunder.mp3",
    "Rhinoceros": "blank.mp3",
    "Sheep": "dog.mp3",
    "Tiger": "metalclang.mp3",
    "Zebra": "blank.mp3"
}
# ...
def train_animal_detector(train_data_dir, num_epochs=10):
    # Set up data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create a generator for training data
    batch_size = 32
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Use a pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom layers for animal detection
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(train_generator.class_indices), activation='softmax'))

    # Freeze the pre-trained layers for fine-tuning
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, epochs=0)

    return model, train_generator.class_indices

# ...


def detect_animal(model, test_image_path, class_indices):
    # Load and preprocess the test image
    test_img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
    test_img = tf.keras.preprocessing.image.img_to_array(test_img)
    test_img = test_img.reshape((1, 224, 224, 3))
    test_img = test_img / 255.0

    # Make a prediction on the test image
    predictions = model.predict(test_img)
    predicted_class = list(class_indices.keys())[tf.argmax(predictions, axis=1)[0]]

    return predicted_class

if __name__ == "__main__":
    train_data_dir = "C:/Users/Bala/OneDrive/Desktop/211FA20018/train"
    test_image_path = "C:/Users/Bala/OneDrive/Desktop/211FA20018/ANIMAL DETECTION/train/Pig/dcd6a34ee087ca63.jpg"

    # Train the model for more epochs for better accuracy
    model, class_indices = train_animal_detector(train_data_dir, num_epochs=20)

    predicted_animal = detect_animal(model, test_image_path, class_indices)
    
    print(f"The detected animal is: {predicted_animal}")
    
    # Play audio based on the detected animal
    audio_file = animal_audio.get(predicted_animal, "default_audio.mp3")
    play_audio(audio_file)
