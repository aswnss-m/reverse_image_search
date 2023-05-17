# Importing Necessary libraries
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from tqdm import tqdm_notebook
from PIL import Image
import pandas as pd
import numpy as np
import os
# Load pre-trained VGG16 model without the classification layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def get_feature_vector(img_path):
    # Load image from file path and resize to 400x400
    print(img_path)
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
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
        print("done one")
        subset_feature_vectors.append(feature_vector)
    
    return subset_feature_vectors
# Define the directory path where the logos are stored
logo_dir = './LLD-logo_files/LLD-logo-files/'



# Iterate through all the image files in the logos directory
for i in [10,11]:
    counter = 1
    feature_vectors = []
    labels = []

    with open("./LLD-logo_files/subsets/subset_files_{}.txt".format(str(i))) as f:
        filenames = f.read().split("\n")


    for filename in filenames:
        # Check if the file is a PNG image
        print("item : ",counter)
        counter += 1
        if filename.endswith('.png'):
            # Construct the full file path
            img_path = os.path.join(logo_dir, filename)
            # Obtain the feature vector for the image and append it to the feature_vectors list
            feature_vector = get_feature_vector(img_path)
            
            if len(feature_vector) > 0:
                feature_vectors.append(feature_vector)
                # Extract the label from the filename and append it to the labels list
                label = filename.split('.')[0]  # Assuming the label is the part of the filename before the extension
                labels.append(label)

    # Convert the feature_vectors and labels lists to numpy arrays
    feature_vectors = np.array(feature_vectors)
    labels = np.array(labels)

    # Round the feature vectors to a specified number of decimal places (e.g., 6)
    feature_vectors = np.round(feature_vectors, decimals=6)

    # Save the feature_vectors array as a numpy file
    np.save('feature_vectors_{}.npy'.format(str(i)), feature_vectors)

    dataframe_vectors = [vector.flatten() for vector in feature_vectors]

    # Check if the lengths of feature_vectors and labels are the same
    if len(dataframe_vectors) == len(labels):
        # Create a pandas DataFrame with the feature vectors and labels
        data = pd.DataFrame({'Label': labels, 'Feature Vector': dataframe_vectors})

        # Save the DataFrame as a CSV file
        data.to_csv('annotated_feature_vectors_{}.csv'.format(str(i)), index=False)
    else:
        print("Error: Length mismatch between feature vectors and labels.")
    