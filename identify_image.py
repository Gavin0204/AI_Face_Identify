import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('face_classifier_model.keras')

def identify_image(img_path):
    # Load and preprocess the image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not load image.")
        return "Wrong image"
    
    # Resize the image to 150x150 pixels (the input size expected by the model)
    img = cv2.resize(img, (150, 150))
    
    # Normalize the image (convert pixel values to [0, 1])
    img = img.astype('float32') / 255.0
    
    # Expand dimensions to match the model's input shape
    img = np.expand_dims(img, axis=0)
    
    # Make a prediction
    predictions = model.predict(img)
    
    # Interpret the result
    predicted_class = (predictions[0] > 0.5).astype(int)
    
    # Display the result
    if predicted_class[0] == 1:
        return "welcome"
    else:
        return "Wrong image"

# Main loop to input image and display result
while True:
    # Get the image path from the user
    image_path = 'images/path_to_image.jpg'
    
    # Allow the user to exit the loop
    if image_path.lower() == 'exit':
        print("Exiting the program.")
        break
    
    # Identify the image and print the result
    result = identify_image(image_path)
    print(result)
