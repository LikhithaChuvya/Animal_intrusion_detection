import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load a dataset (e.g., CIFAR-10) for demonstration
(train_images, train_labels), _ = datasets.cifar10.load_data()
train_images = train_images / 255.0  # Normalize pixel values to [0, 1]

# Define your CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # Output layer with 10 units (for CIFAR-10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model (you should have your own training dataset and labels)
model.fit(train_images, train_labels, epochs=10)

# Define a directory to save your model as a SavedModel
model_dir = "C:/Users/Bala/OneDrive/Desktop/211FA20018/train"  # Replace with your desired directory

# Save the model as a TensorFlow SavedModel
tf.saved_model.save(model, model_dir)

# Optionally, if you have additional custom objects (e.g., custom layers or metrics), you can register them
# For example, if you have a custom layer named 'CustomLayer':
# tf.keras.utils.get_custom_objects()['CustomLayer'] = CustomLayer

# Your model is now saved as a SavedModel in the specified directory
