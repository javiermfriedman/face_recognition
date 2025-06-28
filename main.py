from image_preprocess import load_data
from image_preprocess import load_test_image
from cnn_classifier import build_cnn_model
from cnn_classifier import compile_and_train_model
from cnn_classifier import plot_loss_tf
from cnn_classifier import load_trained_model
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    input_str = "would you like to make prediction[1] or train[2]: "
    user_data = input(input_str)

    if user_data == '1':
        predict_nexus()
    elif user_data == '2':
        face_recog_nexus()

    else:
        print("idk")

    return

def predict_nexus():
    model_save_path = 'trained_face_classifier_model.keras'

    trained_model = load_trained_model(model_save_path)

    if trained_model is None:
        print("[INFO] No pre-trained model found or failed to load\n")
        return # Added return to exit if model not loaded

    test_data = load_test_image("archive/test_cases")
    if len(test_data) == 0: # Added check for empty test_data
        print("[INFO] No test images found. Exiting prediction.")
        return

    for i in range(len(test_data)):
        img = test_data[i]

        # print(f" predicting if the following image is a face or not")
        # Ensure img is 2D for imshow if it's (H,W,1) or (H,W,3)
        if img.shape[-1] == 1: # Grayscale, remove channel for display
            plt.imshow(img.squeeze(), cmap='gray')
        else: # Color image
            plt.imshow(img)
        plt.title("Prediction") # Changed title
        plt.axis("off")
        plt.show()

        img_for_prediction = np.expand_dims(img, axis=0)
        prediction = trained_model.predict(img_for_prediction)  # prediction
        
        # Get the single probability value
        probability = prediction[0][0] 

        # Define your threshold
        threshold = 0.5 # You can adjust this value if needed

        # Apply the threshold to determine True or False
        is_face = probability > threshold

        print(f" is it a face: --> {is_face} (Probability: {probability:.6f}) <-- ")




def face_recog_nexus():
    print("[INFO] Starting face recognition pipeline...")

    # Step 1: Load data
    X, y = load_data("archive/faces", "archive/non_faces")

    print ('The shape of X is: ' + str(X.shape))
    print ('The shape of y is: ' + str(y.shape)) 

    # Step 2: create the achitecture or the CNN
    # Pass the correct input shape (3 for color images)
    model = build_cnn_model(input_shape=(180, 180, 3))
   
    # step 2.5: create a path saver
    model_save_path = 'trained_face_classifier_model.keras'
    
    # Create the directory if it doesn't exist (only the directory part, not the file)
    model_save_directory = os.path.dirname(model_save_path)
    if model_save_directory and not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)
        print(f"[INFO] Created directory: {model_save_directory}")

    # Step 3: train the model
    trained_model = compile_and_train_model(
                model, X, y, 15, 32, 0.2, model_save_path) # Pass the full path with extension
    
    if trained_model is None: # If training failed
        print("[ERROR] Model training failed. Exiting.")
        return

    # Step 4: test data
    # get test data and pipe it through 
    test_data = load_test_image("archive/test_cases")
    if len(test_data) == 0: # Added check for empty test_data
        print("[INFO] No test images found. Exiting testing.")
        return
    
    for i in range(len(test_data)):
        img = test_data[i]

        # print(f" predicting if the following image is a face or not")
        # Ensure img is 2D for imshow if it's (H,W,1) or (H,W,3)
        if img.shape[-1] == 1: # Grayscale, remove channel for display
            plt.imshow(img.squeeze(), cmap='gray')
        else: # Color image
            plt.imshow(img)
        plt.title("Prediction") # Changed title
        plt.axis("off")
        plt.show()
        img_for_prediction = np.expand_dims(img, axis=0)
        prediction = trained_model.predict(img_for_prediction)  # prediction
        
        # Get the single probability value
        probability = prediction[0][0] 

        # Define your threshold
        threshold = 0.5 # You can adjust this value if needed

        # Apply the threshold to determine True or False
        is_face = probability > threshold

        print(f" is it a face: --> {is_face} (Probability: {probability:.6f}) <-- ")

    print("[INFO] Pipeline completed.")

if __name__ == "__main__":
    main()