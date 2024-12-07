# generate_training_data.py

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import skimage.filters

# Hàm loại bỏ nền (Background removal) sử dụng thresholding
def remove_background(img_array):
    gray = tf.image.rgb_to_grayscale(img_array)
    threshold = tf.reduce_mean(gray)
    img_array = tf.where(gray > threshold, img_array, tf.zeros_like(img_array))
    return img_array

# Hàm lọc và phân đoạn (filtering and segmentation)
def filter_and_segment(img_array):
    # Apply Gaussian filter to each channel directly in TensorFlow to avoid NumPy overhead
    img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)  # Ensure image is float32
    img_filtered = tf.image.adjust_contrast(img_array, contrast_factor=2.0)  # Simple contrast adjustment
    return img_filtered

# Hàm tải và tiền xử lý dữ liệu
def load_data(data_dir, label_file, batch_size=32):
    labels_df = pd.read_csv(label_file)
    image_paths = [os.path.join(data_dir, img_file) for img_file in os.listdir(data_dir) if img_file.endswith('.png')]

    # Debug: Print number of images found
    print(f"Found {len(image_paths)} images in {data_dir}")

    # Trích xuất nhãn từ cột 'happiness' (hoặc cột cảm xúc khác mà bạn muốn sử dụng)
    image_labels = {img_file: labels_df[labels_df['Image name'] == img_file]['happiness'].values[0] 
                    for img_file in os.listdir(data_dir) if img_file.endswith('.png')}

    def process_image(img_path):
        img_path = tf.strings.as_string(img_path)  # Convert Tensor to string
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (48, 48))
        img = tf.cast(img, tf.float32) / 255.0  # Normalize image

        # Apply background removal and filtering
        img = remove_background(img)
        img = filter_and_segment(img)

        # Get label (image_labels expects a string, so we use `os.path.basename` correctly)
        label = image_labels[os.path.basename(img_path.numpy().decode('utf-8'))]  # Decode the tensor to a string
        return img, label

    # Load dataset with map and batch (using prefetch for performance)
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(lambda img_path: tf.py_function(func=process_image, inp=[img_path], Tout=(tf.float32, tf.int32)),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.cache()  # Cache dataset to memory after first epoch

    # Debug: Print the number of batches
    print(f"Dataset có {len(list(dataset))} batches.")
    
    return dataset

# Hàm chuẩn bị dữ liệu cho huấn luyện, kiểm thử và kiểm tra
def preprocess_data(train_dir, valid_dir, test_dir, label_file):
    print("Preprocessing data...")
    train_data = load_data(train_dir, label_file)
    valid_data = load_data(valid_dir, label_file)
    test_data = load_data(test_dir, label_file)

    return train_data, valid_data, test_data

# Hàm main để chạy quá trình tiền xử lý
if __name__ == '__main__':
    # Define directories and label file
    train_dir = './data/FER2013Train'
    valid_dir = './data/FER2013Valid'
    test_dir = './data/FER2013Test'
    label_file = './data/label.csv'

    # Tiền xử lý dữ liệu
    try:
        train_data, valid_data, test_data = preprocess_data(train_dir, valid_dir, test_dir, label_file)
        print("Data preprocessing completed.")

        # Đảm bảo thư mục lưu
        saved_dir = './saved'
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        # Lưu dữ liệu đã tiền xử lý vào các file npy
        def save_data(dataset, filename):
            images = []
            labels = []
            for image_batch, label_batch in dataset:
                images.append(image_batch.numpy())  # Convert tensor to numpy array
                labels.append(label_batch.numpy())  # Convert tensor to numpy array

            # Convert list of batches to NumPy arrays and save
            np.save(os.path.join(saved_dir, filename + '_images.npy'), np.concatenate(images, axis=0))
            np.save(os.path.join(saved_dir, filename + '_labels.npy'), np.concatenate(labels, axis=0))

        # Save training, validation, and test data
        save_data(train_data, 'train_data')
        save_data(valid_data, 'valid_data')
        save_data(test_data, 'test_data')

        print(f"Đã lưu {len(train_data)} ảnh trong thư mục {saved_dir}")
        print(f"Đã lưu {len(valid_data)} ảnh trong thư mục {saved_dir}")
        print(f"Đã lưu {len(test_data)} ảnh trong thư mục {saved_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")
