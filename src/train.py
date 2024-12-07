import numpy as np
import logging
import os
import argparse
from tensorflow.keras.utils import to_categorical
from models import build_model
from evaluate import evaluate_model
import seaborn as sns
import matplotlib.pyplot as plt

# Define emotion table (emotion labels)
emotion_table = {
    'neutral'  : 0, 
    'happiness': 1, 
    'surprise' : 2, 
    'sadness'  : 3, 
    'anger'    : 4, 
    'disgust'  : 5, 
    'fear'     : 6, 
    'contempt' : 7
}

def check_and_filter_labels(labels, emotion_table):
    """
    Check and filter any invalid labels (if any) in the data.
    """
    invalid_labels = labels[labels >= len(emotion_table)]
    if len(invalid_labels) > 0:
        logging.warning(f"Found {len(invalid_labels)} invalid labels. These labels will be replaced with 0.")
        # Filter invalid labels (e.g., replace with class 0)
        labels[labels >= len(emotion_table)] = 0  
    else:
        logging.info("All labels are valid.")
    return labels

def main(base_folder="C:/Users/vuong/Documents/Code/Team12/results", 
         training_mode='majority', 
         model_name='EfficientNet-B3', 
         max_epochs=100):
    # Set up the model output folder
    output_model_path = os.path.join(base_folder, 'models')
    output_model_folder = os.path.join(output_model_path, f"{model_name}_{training_mode}")
    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    # Configure logging
    logging.basicConfig(filename=os.path.join(output_model_folder, "train.log"), filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())  # Add stream handler for console logs

    logging.info(f"Starting training with mode {training_mode}, model {model_name}, max epochs {max_epochs}.")

    # Load the dataset from .npy files
    try:
        logging.info("Loading data...")
        train_images = np.load('./saved/train_data_images.npy')
        train_labels = np.load('./saved/train_data_labels.npy')
        valid_images = np.load('./saved/valid_data_images.npy')
        valid_labels = np.load('./saved/valid_data_labels.npy')
        test_images = np.load('./saved/test_data_images.npy')
        test_labels = np.load('./saved/test_data_labels.npy')
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Filter labels to remove any invalid ones
    train_labels = check_and_filter_labels(train_labels, emotion_table)
    valid_labels = check_and_filter_labels(valid_labels, emotion_table)
    test_labels = check_and_filter_labels(test_labels, emotion_table)

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=len(emotion_table))
    valid_labels = to_categorical(valid_labels, num_classes=len(emotion_table))
    test_labels = to_categorical(test_labels, num_classes=len(emotion_table))

    # Verify data shape
    logging.info(f"Train images shape: {train_images.shape}")
    logging.info(f"Train labels shape: {train_labels.shape}")
    logging.info(f"Validation images shape: {valid_images.shape}")
    logging.info(f"Validation labels shape: {valid_labels.shape}")
    logging.info(f"Test images shape: {test_images.shape}")
    logging.info(f"Test labels shape: {test_labels.shape}")

    # Build the model
    model = build_model(len(emotion_table), model_name)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    logging.info("Training the model...")
    history = model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), 
                        epochs=max_epochs, batch_size=128)

    # Save the trained model
    model_save_path = os.path.join(output_model_folder, f"{model_name}_final_model.h5")
    model.save(model_save_path)
    logging.info(f"Model saved to {model_save_path}.")

    # Evaluate the model and calculate metrics
    evaluate_model(model, test_images, test_labels, emotion_table)

if __name__ == "__main__":
    # Default parameters
    base_folder = "C:/Users/vuong/Documents/Code/Team12/results"  # Default base folder
    training_mode = 'majority'  # Default training mode
    model_name = 'EfficientNet-B3'  # Default model name
    max_epochs = 2  # Default number of epochs

    main(base_folder, training_mode, model_name, max_epochs)