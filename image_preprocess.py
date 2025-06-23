# image_preprocess.py - handles loading and visualizing images from LFW dataset

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# os handles directory paths and file listings.
# cv2 Used for reading, transforming, and manipulating images and video

"""
    Preprocesses a face image:
    - Reads with OpenCV
    - Resizes to target dimensions
    - Normalizes pixel values to [0, 1]
    - Converts to NumPy array
    - Returns shape: (100, 100, 3)
"""
def preprocess_face_image(path, target_size=(100, 100)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Corrects blue tint
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img

"""
    Preprocesses a non-face image:
    - Reads with OpenCV
    - Resizes to target dimensions
    - Normalizes pixel values to [0, 1]
    - Converts to NumPy array
    - Returns shape: (100, 100, 3)
"""
def preprocess_non_face_image(path, target_size=(100, 100)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at path: {path}")
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img.astype(np.float32)

"""
    takes the data from non face pictures and puts them into array
"""
def load_neg_data(dataset_path):
    print("[INFO] Loading sample faces from:", dataset_path)
    neg_data = []
    images = os.listdir(dataset_path)

    for image in images:
        img_path = os.path.join(dataset_path, image)
        neg_data.append(preprocess_non_face_image(img_path))

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


    for person in people:
        person_dir = os.path.join(dataset_path, person) # makes a path for the person's folder content
        image_name = os.listdir(person_dir)[0]  # Get first image only
        img_path = os.path.join(person_dir, image_name)
        face_data.append(preprocess_face_image(img_path))

    return face_data
        
def load_data(face_img_path, other_img_path):
    face_data = load_face_data(face_img_path)
     
#     for image in face_data: 
#         plt.imshow(image) 
#         plt.title("label")
#         plt.axis("off")
#         plt.show()



    neg_data = load_neg_data(other_img_path)

    for image in neg_data:
        plt.imshow(image) 
        plt.title("label")
        plt.axis("off")
        plt.show()

    # create labels for the data we just stored in an array
    y_face = [1] * len(face_data)
    y_neg = [0] * len(neg_data)

    # Combine
    X_data = face_data + neg_data
    y_data = y_face + y_neg

    # Convert to NumPy arrays for training
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    print(f"[INFO] Total samples: {len(X_data)}")
    print(f"[INFO] X_data shape: {X_data.shape}")
    print(f"[INFO] y_data shape: {y_data.shape}")
    print(f"[INFO] Positive samples: {sum(y_data)} | Negative samples: {len(y_data) - sum(y_data)}")

    return
