from image_preprocess import load_data
from cnn_classifier import train_face_classifier

def main():
    print("[INFO] Starting face recognition pipeline...")

    # Step 1: Load and display sample faces for inspection
    #load_and_display_sample_faces("archive/lfw_funneled", num_people=5, num_images_per_person=5)

    X, y = load_data("archive/faces", "archive/non_faces")
#     print(f"[INFO] Total samples: {len(X_data)}")
#     print(f"[INFO] X_data shape: {X_data.shape}")
#     print(f"[INFO] y_data shape: {y_data.shape}")
#     print(f"[INFO] Positive samples: {sum(y_data)} | Negative samples: {len(y_data) - sum(y_data)}")

    print ('The shape of X is: ' + str(X.shape))
    print ('The shape of y is: ' + str(y.shape))  
    # train_face_classifier(X, y)

    # Next steps (to implement):
    # - Clean and filter dataset
    # - Train TensorFlow CNN
    # - Evaluate model and save predictions

    print("[INFO] Pipeline completed.")

if __name__ == "__main__":
    main()