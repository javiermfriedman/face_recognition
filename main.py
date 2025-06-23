from image_preprocess import load_data

def main():
    print("[INFO] Starting face recognition pipeline...")

    # Step 1: Load and display sample faces for inspection
    #load_and_display_sample_faces("archive/lfw_funneled", num_people=5, num_images_per_person=5)

    load_data("archive/lfw_funneled", "archive/non_faces")

    # Next steps (to implement):
    # - Clean and filter dataset
    # - Train TensorFlow CNN
    # - Evaluate model and save predictions

    print("[INFO] Pipeline completed.")

if __name__ == "__main__":
    main()