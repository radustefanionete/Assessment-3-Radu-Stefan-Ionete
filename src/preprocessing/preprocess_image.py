import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from PIL import Image

# Defining image dimensions for resizing
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Loading pre-trained ResNet50 model + higher level layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Adding global average pooling layer to reduce dimensions of the output
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)

# Function to preprocess and extract features from a single image
def preprocess_image(img_path):
    # Loading the image file
    img = Image.open(img_path)
    
    # Resizing image to match ResNet50 input size
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Converting the image to an array
    img_array = np.array(img)
    
    # If the image has an alpha channel (RGBA), convert to RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # Preprocessing the image for ResNet50 model
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocessing for ResNet50
    
    return img_array

# Function to extract features from all images in a directory
def extract_features_from_images(image_dir):
    features = []
    image_paths = []
    
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            # Preprocess image and extract features
            img_array = preprocess_image(img_path)
            feature = model.predict(img_array)
            
            features.append(feature)
            image_paths.append(img_name)
    
    # Converting the list of features to a numpy array
    features = np.array(features)
    
    # Flattening the features to make it 2D for training (samples, features)
    features = features.reshape(features.shape[0], -1)
    
    return features, image_paths

# Main function to preprocess and extract image features
if __name__ == '__main__':
    # Path to the directory containing images
    image_dir = 'data/twitter_sentiment_data/images'  # Adjust to the path in your dataset

    # Extracting features from images
    image_features, image_paths = extract_features_from_images(image_dir)
    
    # Output the shape of the extracted features
    print(f"Extracted image features shape: {image_features.shape}")
    
    # Saving the extracted features if necessary
    joblib.dump(image_features, 'image_features.pkl')
    joblib.dump(image_paths, 'image_paths.pkl')
