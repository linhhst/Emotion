# evaluate.py

import os
import numpy as np
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model: Model, test_images: np.ndarray, test_labels: np.ndarray, emotion_table: dict):
    """
    Evaluate the model and calculate metrics.

    Parameters:
        model (Model): Trained Keras model.
        test_images (np.ndarray): Test set images.
        test_labels (np.ndarray): Test set labels (one-hot encoded).
        emotion_table (dict): Mapping of emotion names to indices.
    """
    # Evaluate the model on the test set
    print("Evaluating the model...")
    score = model.evaluate(test_images, test_labels, verbose=1)
    print(f"Test Loss: {score[0]}")
    print(f"Test Accuracy: {score[1]}")

    # Predict labels
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Get true labels
    true_labels = np.argmax(test_labels, axis=1)

    # Generate confusion matrix and classification report
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(
        true_labels, predicted_labels, target_names=list(emotion_table.keys()), zero_division=0
    )

    # Print classification report
    print("Classification Report:")
    print(report)

    # Create results directory if it doesn't exist
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=list(emotion_table.keys()),
        yticklabels=list(emotion_table.keys()),
        cbar=False
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Save the confusion matrix image
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()  # Close plot to free up memory