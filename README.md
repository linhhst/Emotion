Thư viện
XGBoost
Scikit-learn Intelex (Intel-optimized version of scikit-learn)
Intel Extension for TensorFlow (Optimized for Intel CPUs)
python -m venv myenv
myenv\Scripts\activate
pip install tensorflow==2.15
Modin (Ray backend for distributed DataFrame)
Neural Compressor (Model compression for AI models)
Sử dụng mô hình EfficientNet-B3 cho việc Xây dựng hệ thống phân loại cảm xúc qua khuôn mặt của con người
35,710 ảnh và data csv
Kết quả phân tích và đánh giá
Các độ so sánh:
Accuracy, Precision, Recall, F1-Score, Confusion Matrix
Training Time (Thời gian huấn luyện) loss
Inference Time (Thời gian dự đoán) ms/step loss
Vẽ 2D Density Plot
Thư mục lưu trữ mô hình sau huấn luyện
file h5
và biểu đồ Vẽ 2D Density Plot

import streamlit as st
emotion_classification/
│
├── data/                          # Dữ liệu huấn luyện, kiểm thử và kiểm tra
│   ├── FER2013Train/              # Dữ liệu huấn luyện (ảnh khuôn mặt và nhãn)
│   ├── FER2013Train.csv           # File CSV chứa nhãn và đường dẫn ảnh huấn luyện
│   ├── FER2013Valid/              # Dữ liệu kiểm tra (ảnh khuôn mặt và nhãn)
│   ├── FER2013Valid.csv           # File CSV chứa nhãn và đường dẫn ảnh kiểm tra
│   ├── FER2013Test/               # Dữ liệu kiểm thử (ảnh khuôn mặt và nhãn)
│   └── FER2013Test.csv            # File CSV chứa nhãn và đường dẫn ảnh kiểm thử
│
├── src/                           # Mã nguồn chính của dự án
│   ├── data_processing.py         # Tiền xử lý dữ liệu, tạo DataLoader
│   ├── model.py                   # Định nghĩa mô hình EfficientNet-B3
│   ├── train.py                   # Mã huấn luyện mô hình đánh giá mô hình, tính toán các chỉ số đánh giá
│   ├── evaluate.py   
│   ├── classification_report.py   # Tính toán và vẽ Classification Report
│   ├── plot_utils.py              # Các hàm vẽ biểu đồ (Confusion Matrix, 2D Density Plot, v.v.)
│   ├── inference.py               # Dự đoán (inference) trên dữ liệu mới
│   ├── utils.py                   # Các hàm tiện ích hỗ trợ chung (ví dụ: lưu mô hình, kiểm tra tình trạng hệ thống)
│
├── saved/                         # Thư mục lưu trữ mô hình sau huấn luyện
│   ├── model_best.pth             # Mô hình tốt nhất theo validation loss
│   └── final_model.pth            # Mô hình cuối cùng sau huấn luyện
│
├── results/                       # Kết quả phân tích và đánh giá
│   ├── metrics.csv                # Các chỉ số đánh giá mô hình (accuracy, precision, recall, F1-score, v.v.)
│   ├── confusion_matrix.png       # Confusion Matrix dưới dạng hình ảnh
│   └── 2d_density_plot.png        # Biểu đồ phân bố 2D (Density Plot)
│
└── README.md                      # Hướng dẫn sử dụng và mô tả dự án

from numba import jit

# Emotion Classification Using EfficientNet-B3

## Giới thiệu
Hệ thống phân loại cảm xúc qua khuôn mặt con người sử dụng mô hình EfficientNet-B3, thư viện Fastai và JIT compiler (Numba) để tối ưu hóa quá trình inference.

## Cài đặt
1. Cài đặt các thư viện cần thiết:
pip install fastai efficientnet_pytorch numba scikit-learn matplotlib seaborn

2. Chạy huấn luyện:
python src/train.py

3. Đánh giá mô hình:
python src/evaluate.py

## Cấu trúc dự án
- `data/`: Dữ liệu huấn luyện, kiểm thử, kiểm tra.
├── data/                          # Dữ liệu huấn luyện, kiểm thử và kiểm tra
│   ├── FER2013Train/              # Dữ liệu huấn luyện (ảnh khuôn mặt và nhãn)
│   ├── FER2013Train.csv           # File CSV chứa nhãn và đường dẫn ảnh huấn luyện
│   ├── FER2013Valid/              # Dữ liệu kiểm tra (ảnh khuôn mặt và nhãn)
│   ├── FER2013Valid.csv           # File CSV chứa nhãn và đường dẫn ảnh kiểm tra
│   ├── FER2013Test/               # Dữ liệu kiểm thử (ảnh khuôn mặt và nhãn)
│   └── FER2013Test.csv            # File CSV chứa nhãn và đường dẫn ảnh kiểm thử
- `src/`: Mã nguồn chính của dự án.
- `results/`: Các kết quả phân tích và đánh giá.
- `saved/`: Thư mục lưu trữ mô hình sau huấn luyện
Kết quả của mô hình nhận dạng hình ảnh thường được thể hiện qua các chỉ số như:
Accuracy, Precision, Recall, F1-Score, và Confusion Matrix
Accuracy: Tỷ lệ đúng.
Precision: Độ chính xác (Chỉ số của các dự đoán đúng trên tổng số dự đoán).
Recall: Độ nhạy (Chỉ số của các dự đoán đúng trên tổng số thực tế).
F1-Score: Trung bình điều hòa giữa Precision và Recall
Confusion Matrix: Là một bảng tóm tắt được sử dụng để kiểm tra hiệu quả của mô hình phân loại
Ma trận nhầm lẫn gồm các thành phần:
True Positives (TP): Dự đoán đúng là dương tính.
False Positives (FP): Dự đoán sai là dương tính.
True Negatives (TN): Dự đoán đúng là âm tính.
False Negatives (FN): Dự đoán sai là âm tính.
Time (Thời gian):
Thời gian huấn luyện mô hình (training time).
Thời gian dự đoán (inference time).

project/
│
cho dữ liệu .\data\ # Dữ liệu huấn luyện, kiểm thử và kiểm tra
   ├── FER2013Train/  # Dữ liệu huấn luyện (ảnh khuôn mặt)
   ├── FER2013Valid/  # Dữ liệu kiểm tra (ảnh khuôn mặt)
   ├── FER2013Test/   # Dữ liệu kiểm thử (ảnh khuôn mặt)
   └── label.csv      # File CSV chứa Usage Image name neutral happiness surprise sadness anger disgust fear contempt contempt NF code
   
    ├── src/                          # Mã nguồn chính của hệ thống
│   ├── preprocess.py             # Tiền xử lý dữ liệu ảnh
│   ├── model.py                  # Định nghĩa mô hình CNN
│   ├── train.py                  # Huấn luyện mô hình
│   ├── predict.py                # Dự đoán cảm xúc
│   ├── face_detection.py     # Phát hiện khuôn mặt
│   ├── image_utils.py        # Các hàm xử lý ảnh
│   └── emotion_utils.py      # Các hàm tiện ích về cảm xúc
│
├── results/                       # Các kết quả phân tích và đánh giá
│   ├── accuracy.png               # Biểu đồ accuracy theo thời gian hoặc epoch
│   ├── precision_recall.png       # Biểu đồ precision và recall
│   ├── confusion_matrix.png       # Ma trận nhầm lẫn
│   └── evaluation_report.txt      # Báo cáo đánh giá chi tiết các chỉ số
│
├── saved/                         # Thư mục lưu trữ mô hình sau huấn luyện
│   ├── model_final.h5             # Mô hình đã huấn luyện và lưu trữ
│   └── model_checkpoint.h5        # Mô hình lưu trong quá trình huấn luyện (nếu có)
│
└── README.md                      # Tài liệu hướng dẫn và mô tả dự án




1. data_preprocessing.py – Xử lý dữ liệu: tải dữ liệu và tiền xử lý ảnh
File này chứa các hàm để tải và tiền xử lý dữ liệu trước khi đưa vào mô hình học sâu. Những bước chính thường có trong file này bao gồm:

Tải dữ liệu: Đọc và tải các bộ dữ liệu (ví dụ như từ thư mục, từ các tập tin hình ảnh hoặc từ các cơ sở dữ liệu trực tuyến).
Tiền xử lý ảnh: Áp dụng các phương pháp như chuẩn hóa (normalization), thay đổi kích thước (resizing), chuyển đổi màu sắc, hoặc cắt xén (cropping) để chuẩn bị ảnh phù hợp với yêu cầu của mô hình học sâu. Ngoài ra, có thể sử dụng kỹ thuật augmentations (tăng dữ liệu) để tạo ra nhiều biến thể của ảnh nhằm làm phong phú dữ liệu huấn luyện.
Chia tách dữ liệu: Phân chia dữ liệu thành các tập huấn luyện (training), kiểm tra (validation), và kiểm tra cuối cùng (test).
2. model.py – Định nghĩa mô hình mạng nơ-ron
File này định nghĩa cấu trúc của mô hình học sâu (deep learning model), bao gồm:

Mạng nơ-ron tích chập (CNN): Đây là loại mạng phổ biến trong nhận dạng ảnh, với các lớp tích chập (convolutional layers) giúp mô hình học được các đặc trưng (features) của ảnh, và các lớp pooling giúp giảm kích thước ảnh.
Các lớp khác: Tùy thuộc vào bài toán, file này có thể định nghĩa các lớp Dense (fully connected) hoặc lớp Dropout (để giảm overfitting).
Kỹ thuật tối ưu (Optimizer): Để huấn luyện mô hình, sẽ sử dụng các thuật toán tối ưu như Adam, SGD (Stochastic Gradient Descent), v.v.
Hàm mất mát (Loss function): File này cũng có thể bao gồm hàm mất mát, ví dụ như categorical_crossentropy cho bài toán phân loại đa lớp.
Các tham số khác: Chỉ định các tham số như số lớp, số nơ-ron trong mỗi lớp, và các tham số huấn luyện.
3. train.py – Huấn luyện mô hình
File này chứa các bước thực hiện quá trình huấn luyện mô hình, bao gồm:

Chuẩn bị dữ liệu: Tải dữ liệu đã được xử lý từ data_preprocessing.py, và tạo các lô dữ liệu (batches).
Cấu hình mô hình: Khởi tạo mô hình từ model.py với các tham số cần thiết.
Huấn luyện: Thực hiện quá trình huấn luyện mô hình với dữ liệu huấn luyện. Quá trình này bao gồm việc tính toán gradient và cập nhật các trọng số của mô hình.
Theo dõi tiến trình: Trong quá trình huấn luyện, file này có thể theo dõi độ chính xác và tổn thất của mô hình trên dữ liệu huấn luyện và dữ liệu kiểm tra, giúp đánh giá hiệu quả huấn luyện.
Lưu mô hình: Sau khi huấn luyện xong, mô hình sẽ được lưu lại để sử dụng trong các bước tiếp theo (ví dụ như đánh giá hoặc sử dụng trong thực tế).
4. evaluate.py – Đánh giá mô hình
File này có nhiệm vụ đánh giá chất lượng của mô hình đã được huấn luyện, thường bao gồm:

Đánh giá mô hình trên dữ liệu kiểm tra: Sử dụng dữ liệu kiểm tra để đánh giá mô hình đã được huấn luyện. Các chỉ số đánh giá phổ biến bao gồm:
Accuracy: Tỷ lệ đúng của mô hình.
Precision: Độ chính xác của mô hình khi dự đoán là dương tính.
Recall: Độ nhạy của mô hình, tức là tỷ lệ những mẫu dương tính thực sự được phát hiện.
F1-score: Là trung bình hài hòa của Precision và Recall, dùng để đánh giá khi có sự mất cân đối giữa các lớp.
Confusion Matrix: Cung cấp cái nhìn chi tiết về hiệu suất phân loại của mô hình với các lớp dương tính và âm tính.
5. utils.py – Các hàm tiện ích
File này chứa các hàm hỗ trợ hoặc tiện ích cho các bước khác trong dự án. Các hàm có thể bao gồm:

Vẽ biểu đồ: Ví dụ như vẽ confusion matrix hoặc các đồ thị thể hiện độ chính xác và tổn thất của mô hình trong quá trình huấn luyện.
Lưu và tải mô hình: Các hàm hỗ trợ lưu trữ mô hình đã huấn luyện hoặc tải mô hình từ file.
Khởi tạo mô hình: Một số phương thức để khởi tạo mô hình từ các tham số hoặc cấu hình cụ thể.
Chuyển đổi dữ liệu: Các hàm để chuyển đổi dữ liệu đầu vào như thay đổi kích thước ảnh, hoặc chuẩn hóa đầu vào.



xgboost:

Là một thư viện mạnh mẽ dùng để xây dựng các mô hình học máy, đặc biệt là cây quyết định nâng cao (Gradient Boosted Trees). XGBoost thường được sử dụng trong các bài toán phân loại, hồi quy, và các bài toán dự đoán.
scikit-learn-intelex:

Là một phiên bản tối ưu hóa của thư viện scikit-learn dành cho các bộ vi xử lý Intel. Nó tận dụng các cải tiến phần cứng của Intel (như Intel MKL-DNN) để tăng tốc các thuật toán học máy trong scikit-learn.
intel-extension-for-tensorflow[cpu]==2.15:

Đây là một extension cho TensorFlow, giúp tối ưu hóa các phép toán trên CPU sử dụng các phần cứng Intel. Cấu hình [cpu] chỉ định rằng phiên bản này được tối ưu hóa cho việc sử dụng CPU (không phải GPU). Phiên bản 2.15 là phiên bản cụ thể được yêu cầu.
modin[ray]==0.31.0:

Modin là một thư viện Python giúp tăng tốc các tác vụ xử lý dữ liệu (giống như pandas), nhưng với khả năng phân tán xử lý dữ liệu trên nhiều lõi CPU hoặc các máy tính phân tán. Phần [ray] chỉ định rằng Modin sẽ sử dụng Ray (một hệ thống phân tán) để tăng tốc xử lý. Phiên bản được yêu cầu là 0.31.0.
neural-compressor==2.5.1:

Neural Compressor là một công cụ tối ưu hóa mô hình học sâu (deep learning), giúp giảm thiểu kích thước mô hình và tăng tốc quá trình suy luận (inference). Nó hỗ trợ nhiều framework như TensorFlow, PyTorch, và ONNX. Phiên bản 2.5.1 được yêu cầu ở đây.



1. Tạo môi trường ảo (Virtual Environment)
Trước tiên, bạn tạo môi trường ảo để tránh xung đột giữa các phiên bản thư viện khác nhau.

bash
Sao chép mã
python -m venv myenv
myenv\Scripts\activate  # Windows
source myenv/bin/activate  # Linux/MacOS
2. Cài đặt các thư viện cần thiết
a. TensorFlow
Dùng TensorFlow để huấn luyện mô hình, trong trường hợp này là EfficientNet-B3.

bash
Sao chép mã
pip install tensorflow==2.15
b. Keras (cho mô hình EfficientNet-B3)
EfficientNet-B3 là một mô hình có sẵn trong Keras (một API của TensorFlow), bạn không cần phải cài đặt riêng biệt nếu đã cài đặt TensorFlow.

c. Scikit-learn
Thư viện này hỗ trợ các công cụ tiền xử lý dữ liệu và các thuật toán học máy như phân loại và đánh giá mô hình.

bash
Sao chép mã
pip install scikit-learn
d. Intel Optimizations
Các thư viện tối ưu hoá cho Intel CPUs giúp tăng tốc độ tính toán, bạn có thể sử dụng các phiên bản tối ưu hoá của scikit-learn và TensorFlow.

Intel Extension for Scikit-learn (scikit-learn-Intel)
bash
Sao chép mã
pip install scikit-learn-intelex
Intel Extension for TensorFlow
bash
Sao chép mã
pip install intel-tensorflow
e. Modin (Tăng tốc xử lý dữ liệu)
Modin là một thư viện tăng tốc xử lý dữ liệu DataFrame, có thể sử dụng Ray backend để phân tán tính toán.

bash
Sao chép mã
pip install modin[ray]
f. Neural Compressor (Compression for AI models)
Thư viện này giúp nén mô hình để giảm kích thước và tăng tốc độ dự đoán mà không làm mất quá nhiều độ chính xác.

bash
Sao chép mã
pip install neural-compressor
g. Các thư viện xử lý hình ảnh và CSV
Bạn cũng cần cài đặt một số thư viện để xử lý ảnh và dữ liệu CSV.

bash
Sao chép mã
pip install pandas numpy matplotlib opencv-python pillow
3. Tiền xử lý dữ liệu
Dưới đây là cách bạn có thể tiền xử lý ảnh và dữ liệu CSV trước khi đưa vào mô hình.

Đọc dữ liệu CSV
Giả sử dữ liệu CSV của bạn chứa các chỉ số cảm xúc và tên tệp ảnh:

python
Sao chép mã
import pandas as pd

# Đọc dữ liệu CSV chứa thông tin về ảnh và nhãn cảm xúc
data = pd.read_csv("path_to_data.csv")
Đọc và xử lý ảnh
Dùng OpenCV hoặc Pillow để đọc ảnh và chuyển thành định dạng mà mô hình EfficientNet-B3 yêu cầu (ví dụ: 224x224 pixels, chuẩn hóa giá trị pixel).

python
Sao chép mã
import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize về kích thước 224x224
    img = np.array(img) / 255.0  # Chuẩn hóa giá trị pixel (0-1)
    return img
4. Xây dựng mô hình EfficientNet-B3
EfficientNet-B3 có sẵn trong Keras và TensorFlow. Bạn chỉ cần tải mô hình này và huấn luyện nó với dữ liệu của mình.

python
Sao chép mã
import tensorflow as tf
from tensorflow.keras import layers, models

# Tải mô hình EfficientNet-B3 đã được huấn luyện trước (pre-trained)
base_model = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Tạo mô hình mới với EfficientNet-B3 là backbone
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 cảm xúc
])

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# In summary của mô hình
model.summary()
5. Huấn luyện mô hình
Giả sử bạn đã có ảnh và nhãn trong dữ liệu, bạn có thể chia tập huấn luyện và kiểm tra, sau đó huấn luyện mô hình như sau:

python
Sao chép mã
from sklearn.model_selection import train_test_split

# Giả sử X là danh sách các ảnh đã được tiền xử lý và y là nhãn (one-hot encoding)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
history = model.fit(np.array(X_train), np.array(y_train), epochs=10, validation_data=(np.array(X_val), np.array(y_val)))
6. Đánh giá mô hình
Sau khi huấn luyện xong, bạn có thể đánh giá mô hình trên tập kiểm tra hoặc sử dụng các chỉ số như độ chính xác, độ nhớ, F1-score.

python
Sao chép mã
from sklearn.metrics import classification_report

# Dự đoán và đánh giá
y_pred = model.predict(np.array(X_val))
y_pred = np.argmax(y_pred, axis=1)  # Lấy nhãn với xác suất cao nhất

print(classification_report(np.argmax(y_val, axis=1), y_pred))
7. Nén mô hình với Neural Compressor
Cuối cùng, bạn có thể sử dụng Neural Compressor để nén mô hình và tăng tốc độ dự đoán.

python
Sao chép mã
from neural_compressor import Quantization

# Nén mô hình
quantizer = Quantization(model)
quantized_model = quantizer.fit()
Với những bước trên và các thư viện đã nêu, bạn có thể xây dựng hệ thống phân loại cảm xúc qua khuôn mặt của con người bằng mô hình EfficientNet-B3.


import os
import Modin
import argparse
import numpy as np
from itertools import islice
from PIL import Image

một tệp CSV duy nhất fer2013new.csv chứa dữ liệu từ ba tệp CSV khác nhau (FER2013Train.csv, FER2013Valid.csv, FER2013Test.csv)

.\data\ # Dữ liệu huấn luyện, kiểm thử và kiểm tra
   ├── FER2013Train/  # Dữ liệu huấn luyện (ảnh khuôn mặt)
   ├── FER2013Valid/  # Dữ liệu kiểm tra (ảnh khuôn mặt)
   ├── FER2013Test/   # Dữ liệu kiểm thử (ảnh khuôn mặt)
   └── label.csv      # File CSV chứa Usage Image name neutral happiness surprise sadness anger disgust fear contempt contempt NF