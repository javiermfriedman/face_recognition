# image_preprocess.py - handles loading and visualizing images from LFW dataset

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# os handles directory paths and file listings.
# cv2 Used for reading, transforming, and manipulating images and video

def print_img(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title("Processed Square Image")
    plt.axis('off')
    plt.show()

def crop_img(img):
    h, w, _ = img.shape
    if h != w:
        # Determine the size of the square crop
        min_dim = min(h, w)
        
        # Calculate the starting coordinates for the crop to center it
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        
        # Perform the crop
        img = img[start_h:start_h + min_dim, start_w:start_w + min_dim]
        
        # print(f"Cropped image dimensions: {img.shape[1]}x{img.shape[0]}")

    return img


def sharpen_image_unsharp_mask(img_rgb_float_01, sigma=1.0, strength=1.5):
    # Convert to uint8 (0-255)
    img_uint8 = (img_rgb_float_01 * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(img_uint8, (0, 0), sigma)
    sharpened_uint8 = cv2.addWeighted(img_uint8, 1.0 + strength, blurred, -strength, 0)
    
    # Convert back to float 0-1
    sharpened_float_01 = sharpened_uint8 / 255.0
    return sharpened_float_01


def preprocess_image(path, target_size=(150, 150)):
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Warning: Could not load image from {path}. Skipping.")
        return None
    
    # 2. Convert BGR to RGB 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Crop to square
    img = crop_img(img)

    # print(f"image before resize")
    # print_img(img)

    # 5. Resize the image
    img = cv2.resize(img, target_size)

    # print(f"image after resize")
    # print_img(img)

    # 6. Normalize pixel values to [0, 1]
    img = img / 255.0

    # 7. Sharpening 
    img = sharpen_image_unsharp_mask(img, sigma=1.0, strength=1.0) # Tune sigma and strength
    # print(f"image after resize")
    # print_img(img)
    return img


"""
    takes the data from non face pictures and puts them into array
"""
def load_neg_data():
    non_face_1_path = "archive/non_faces"
    non_face_2_path = "archive/non_face2/test"
    non_face_3_path = "archive/non_face2/train"
    
     
    
    neg_data = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff') 

    print("[INFO] Loading sample non-faces from:", non_face_2_path)
    non_faces_2 = os.listdir(non_face_2_path)

    for image_name in non_faces_2:
        img_path = os.path.join(non_face_2_path, image_name)
        
        # Check if it's a file and has a valid image extension
        if os.path.isfile(img_path) and image_name.lower().endswith(image_extensions):
            processed_img = preprocess_image(img_path)
            if processed_img is not None:
                neg_data.append(processed_img)
        
            

    print("[INFO] Loading sample non-faces from:", non_face_1_path)
    non_face_1 = os.listdir(non_face_1_path)
    for image_name in non_face_1:
        img_path = os.path.join(non_face_1_path, image_name)
        
        if os.path.isfile(img_path) and image_name.lower().endswith(image_extensions):
            processed_img = preprocess_image(img_path)
            if processed_img is not None:
                neg_data.append(processed_img)
        
    print("[INFO] Loading sample non-faces from:", non_face_3_path)     
    non_face_3 = os.listdir(non_face_3_path)

    for image_name in non_face_3:
        img_path = os.path.join(non_face_3_path, image_name)
        
        if os.path.isfile(img_path) and image_name.lower().endswith(image_extensions):
            processed_img = preprocess_image(img_path)
            if processed_img is not None:
                neg_data.append(processed_img)

    return neg_data
        
"""
    takes the data from face pictures and puts them into array
"""
def load_face_data():
    

    face_path_1 = "archive/faces_1"
    face_path_2 = "archive/faces_2"

    face_data = []
    
    faces_1 = os.listdir(face_path_1)
    faces_1 = [ # this filters for non directories hidden, that aren;t wanted
        person for person in os.listdir(face_path_1)
        if os.path.isdir(os.path.join(face_path_1, person))
    ]

    faces_2 = os.listdir(face_path_2)
    faces_2 = [ # this filters for non directories hidden, that aren;t wanted
        person for person in os.listdir(face_path_2)
        if os.path.isdir(os.path.join(face_path_2, person))
    ]

    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff') # Added common image extensions
    print("[INFO] Loading sample faces from:", face_path_2)

    for person in faces_2:
        person_dir = os.path.join(face_path_2, person) # makes a path for the person's folder content
        
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            if os.path.isfile(img_path) and image_name.lower().endswith(image_extensions):
                
                processed_img = preprocess_image(img_path)
                
                if processed_img is not None:
                    face_data.append(processed_img)
                    found_image = True
                    
        
    print("[INFO] Loading sample faces from:", face_path_1)
    for person in faces_1:
        person_dir = os.path.join(face_path_1, person) # makes a path for the person's folder content
        
        # Iterate through files in the person's directory to find an image
        found_image = False
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            if os.path.isfile(img_path) and image_name.lower().endswith(image_extensions):
                processed_img = preprocess_image(img_path)
                if processed_img is not None:
                    face_data.append(processed_img)
                    found_image = True
                    
        


    return face_data

"""
    put pos and neg data and put it into x array, build y label array
"""     
def load_data():
    face_data = load_face_data()
    neg_data = load_neg_data()

    print(f"number of face data: {len(face_data)}")
    print(f"number of non_face data: {len(neg_data)}")


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