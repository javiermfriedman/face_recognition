from image_preprocess import load_data
from cnn_classifier import build_cnn_model
from cnn_classifier import compile_and_train_model
from cnn_classifier import plot_loss_tf

def main():
    print("[INFO] Starting face recognition pipeline...")

    # Step 1: Load data
    X, y = load_data("archive/faces", "archive/non_faces")
#     print(f"[INFO] Total samples: {len(X_data)}")
#     print(f"[INFO] X_data shape: {X_data.shape}")
#     print(f"[INFO] y_data shape: {y_data.shape}")
#     print(f"[INFO] Positive samples: {sum(y_data)} | Negative samples: {len(y_data) - sum(y_data)}")

    print ('The shape of X is: ' + str(X.shape))
    print ('The shape of y is: ' + str(y.shape))  
    
    # Step 2: create the achitecture or the CNN
    model = build_cnn_model()
   
    # Step 3: train the model
    history = compile_and_train_model(model, X, y, 40, 32, 0.2)
    
    plot_loss_tf(history)
    

    print("[INFO] Pipeline completed.")

if __name__ == "__main__":
    main()