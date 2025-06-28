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
def preprocess_image(path, target_size=(180, 180)):
    # Read the image directly as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Resize the image
    img = cv2.resize(img, target_size)

    # Normalize pixel values to [0, 1]
    img = img / 255.0

#     # print images
#     plt.imshow(img, cmap='gray') 
#     plt.title("label")
#     plt.axis("off")
#     plt.show()

    # CNNs typically expect this 3D format even for grayscale
    img = np.expand_dims(img, axis=-1)

    return img


"""
    takes the data from non face pictures and puts them into array
"""
def load_neg_data(dataset_path):
    print("[INFO] Loading sample faces from:", dataset_path)
    neg_data = []
    images = os.listdir(dataset_path)


    for image in images:
        img_path = os.path.join(dataset_path, image)
        
        neg_data.append(preprocess_image(img_path))

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
        face_data.append(preprocess_image(img_path))

    return face_data
        
def load_data(face_img_path, other_img_path):
    face_data = load_face_data(face_img_path)

    neg_data = load_neg_data(other_img_path)

#     for i in range(5):
#         face_img = face_data[i]
#         plt.imshow(face_img) 
#         plt.title("label")
#         plt.axis("off")
#         plt.show()
#         non_face_img = neg_data[i]
#         plt.imshow(non_face_img) 
#         plt.title("label")
#         plt.axis("off")
#         plt.show()


    # create labels for the data we just stored in an array
    y_face = [1] * len(face_data)
    y_neg = [0] * len(neg_data)
    print(f"size of face ing data set is: {len(face_data)}")
    print(f"size of non face ing data set is: {len(neg_data)}")
    # Combine
    X_data = face_data + neg_data
    y_data = y_face + y_neg

    # Convert to NumPy arrays for training
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    

    return X_data, y_data
