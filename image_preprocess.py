# image_preprocess.py - handles loading and visualizing images from LFW dataset

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# os handles directory paths and file listings.
# cv2 Used for reading, transforming, and manipulating images and video


def preprocess_image(path, target_size=(180, 180)):
    # Read the image in color (default for cv2.imread, or use cv2.IMREAD_COLOR)
    # OpenCV reads images as BGR by default, so we convert to RGB
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Warning: Could not load image from {path}. Skipping.")
        return None

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image
    # Note: This will still stretch/squish if aspect ratio is different.
    # For more advanced handling, see previous discussion on cropping/padding.
    img = cv2.resize(img, target_size)

    # Normalize pixel values to [0, 1]
    img = img / 255.0

    # For color images, we no longer need to expand_dims as they already have 3 channels (H, W, C)
    # img = np.expand_dims(img, axis=-1) # REMOVE THIS LINE for color images

    return img


"""
    takes the data from non face pictures and puts them into array
"""
def load_neg_data(dataset_path):
    print("[INFO] Loading sample non-faces from:", dataset_path) # Changed message
    neg_data = []
    images = os.listdir(dataset_path)

    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff') # Added common image extensions

    for image_name in images:
        img_path = os.path.join(dataset_path, image_name)
        # Check if it's a file and has a valid image extension
        if os.path.isfile(img_path) and image_name.lower().endswith(image_extensions):
            processed_img = preprocess_image(img_path)
            if processed_img is not None:
                neg_data.append(processed_img)
        else:
            print(f"Skipping non-image file or directory: {img_path}")
            

    return neg_data
        
"""
    takes the data from face pictures and puts them into array
"""
def load_face_data(dataset_path):
    print("[INFO] Loading sample faces from:", dataset_path)

    face_data = []
    people = os.listdir(dataset_path)
    people = [ # this filters for non directories hidden, that aren;t wanted
        person for person in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, person))
    ]

    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff') # Added common image extensions

    for person in people:
        person_dir = os.path.join(dataset_path, person) # makes a path for the person's folder content
        
        # Iterate through files in the person's directory to find an image
        found_image = False
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            if os.path.isfile(img_path) and image_name.lower().endswith(image_extensions):
                processed_img = preprocess_image(img_path)
                if processed_img is not None:
                    face_data.append(processed_img)
                    found_image = True
                    # If you only want one image per person, keep break. 
                    # If you want all images, remove break.
                    break 
        if not found_image:
            print(f"Warning: No valid image found in {person_dir}. Skipping.")


    return face_data

"""
    put pos and neg data and put it into x array, build y label array
"""     
def load_data(face_img_path, other_img_path):
    face_data = load_face_data(face_img_path)

    neg_data = load_neg_data(other_img_path)

    # create labels for the data we just stored in an array
    y_face = [1] * len(face_data)
    y_neg = [0] * len(neg_data)

    # Combine
    X_data = face_data + neg_data
    y_data = y_face + y_neg

    # Convert to NumPy arrays for training
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data

def load_test_image(face_img_path):
    print("[INFO] Loading test data from:", face_img_path)
    test_data = []
    images = os.listdir(face_img_path)

    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff') # Added common image extensions

    for image_name in images:
        img_path = os.path.join(face_img_path, image_name)

        if os.path.isfile(img_path) and image_name.lower().endswith(image_extensions): # Use the generic extensions
            # print(f"Processing image: {img_path}")
            processed_img = preprocess_image(img_path)
            if processed_img is not None:
                test_data.append(processed_img)
        else:
            print(f"Skipping non-image file or directory: {img_path}")

    return np.array(test_data)