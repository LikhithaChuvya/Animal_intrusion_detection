import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import pygame
import cv2

def play_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
animal_audio={ "Bull":"scream.mp3", "Camel":"Blank.mp3", "Cattle":"thunder.mp3","Cheetah":"Blank.mp3",
"Deer": "fireworks.mp3","Elephant":"tiger.mp3.mp3","Fox":"scream.mp3","Giraffe":"Blank.mp3","Goat":"scream.mp3","Goose":"dog.mp3.mp3", "Hippopotamus":"Blank.mp3",
"Horse":"tiger.mp3.mp3","Jaguar": "Blank.mp3","Kangaroo":"metalclang.mp3","Lion":"metalclang.mp3","Monkey":"fireworks.mp3","Pig":"thunder.mp3","Rhinocerous":"Blank.mp3",
"Sheep":"dog.mp3.mp3","Tiger":"metalclang.mp3","Zebra":"Blank.mp3"
}

def train_animal_detector(train_data_dir):
    # Your training code remains the same
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
    epochs = 0  # You can increase the number of epochs
    model.fit(train_generator, epochs=epochs)

    return model, train_generator.class_indices
def detect_animal(model, frame, class_indices):
    # Preprocess the frame from the webcam
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = frame.reshape((1, 224, 224, 3))

    # Make a prediction on the frame
    predictions = model.predict(frame)
    predicted_class = list(class_indices.keys())[tf.argmax(predictions, axis=1)[0]]

    return predicted_class

if __name__ == "__main__":
    # Initialize the model and class indices
    train_data_dir = "C:/Users/Bala/OneDrive/Desktop/211FA20018/train"  # Specify the directory containing your training data
    model, class_indices = train_animal_detector(train_data_dir)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Detect the animal in the frame
        predicted_animal = detect_animal(model, frame, class_indices)
        print(f"The detected animal is: {predicted_animal}")

        # Play the corresponding animal sound
        if predicted_animal in animal_audio:
            audio_to_play = animal_audio[predicted_animal]
            play_audio(audio_to_play)

        # Display the frame with detected animal (optional)
        cv2.putText(frame, f"Animal: {predicted_animal}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Animal Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
