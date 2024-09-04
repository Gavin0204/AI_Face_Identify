import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Paths to your dataset directories
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Load the pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
print("1: This is the base model:", base_model)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = Flatten()(base_model.output)
print("2: This is the Flatten", x)
x = Dense(128, activation='relu')(x)
print("3: This is the Dense1", x)
x = Dense(1, activation='sigmoid')(x)  # Binary classification
print("4: This is the Dense2", x)
# # Create the model
model = Model(base_model.input, x)
print("5: This is the first model", model)
# # Compile the model with updated learning rate argument
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
print("6: This is the compiled model:", model)

# Prepare data augmentation and load images from directories
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
print("7: This is the train data generator", train_datagen)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

print("8: This is the validation data genereator", validation_datagen)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
print("9: This is the train_generator", train_generator)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
print("10: This is the validation_generator", validation_generator)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
print("11: This is the history", history)

# Save the model
model.save('face_classifier_model.keras')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Load an image for prediction (change this path as needed)
img_path = 'images/path_to_image.jpg'
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Image file at path {img_path} could not be loaded. Check the file path and try again")
img = cv2.resize(img, (150, 150))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)
print("12: This is the image", img)
# Make a prediction
predictions = model.predict(img)
print("13: This is the predictions", predictions)
predicted_class = (predictions[0] > 0.5).astype(int)

print(f"Predicted class: {'face' if predicted_class[0] == 1 else 'non-face'}")
