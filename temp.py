from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from tqdm import tqdm, tqdm_notebook

# Load pre-trained VGG16 model without the classification layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

def get_feature_vector(img_path):
    # Load image from file path and resize to 400x400
    img = Image.open("./LLD-logo_files/LLD-logo-files/"+img_path)
    img = img.resize((224, 224))
    
    # Convert image to numpy array and expand dimensions to match input shape of VGG16
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # Preprocess input image using the same method as used during training VGG16
    x = preprocess_input(x)
    
    # Pass the preprocessed image through the VGG16 model and obtain the feature vector
    features = base_model.predict(x)
    
    # Flatten the feature vector and return as 1D array
    feature_vector = features.flatten()
    return feature_vector


def process_photos(photo_subset):
    # Process each photo in the subset
    subset_feature_vectors = []
    for photo_path in tqdm_notebook(photo_subset, desc='Processing subset'):
        # Process the photo and extract the feature vector
        feature_vector = get_feature_vector(photo_path)
        subset_feature_vectors.append(feature_vector)
    
    return subset_feature_vectors

if __name__ == "__main__":
    
