import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.applications import EfficientNetB3  # Import EfficientNetB3 đúng cách

def build_model(num_classes, model_name, img_size=(48, 48)):
    if model_name == "EfficientNet-B3":
        return EfficientNetB3Model(num_classes, img_size).model
    else:
        raise ValueError(f"Model {model_name} not found")

class EfficientNetB3Model:
    @property
    def learning_rate(self):
        return 0.001  # Learning rate bạn muốn sử dụng

    @property
    def input_width(self):
        return 48  # Kích thước đầu vào (tương ứng với EfficientNet-B3)

    @property
    def input_height(self):
        return 48

    @property
    def input_channels(self):
        return 3  # Hình ảnh RGB

    @property
    def model(self):
        return self._model

    def __init__(self, num_classes, img_size=(48, 48)):
        self._model = self._create_model(num_classes, img_size)

    def _create_model(self, num_classes, img_size):
        # Sử dụng EfficientNet-B3 với weights pre-trained từ ImageNet
        base_model = EfficientNetB3(
            include_top=False,  # Không bao gồm phần lớp phân loại gốc của EfficientNet
            weights="imagenet",  # Sử dụng trọng số đã huấn luyện sẵn từ ImageNet
            input_shape=(img_size[0], img_size[1], 3),  # Kích thước hình ảnh đầu vào
            pooling="max"  # Lấy pooling tối đa để làm giảm độ phân giải
        )
        base_model.trainable = False  # Giữ nguyên weights của base model, không huấn luyện lại phần này

        # Thêm các lớp mới để phân loại
        x = base_model.output
        x = BatchNormalization()(x)  # Chuẩn hóa đầu ra của base model
        x = Dense(1024, activation="relu")(x)  # Lớp Dense với 1024 neurons
        x = Dropout(0.3)(x)  # Dropout để tránh overfitting
        x = Dense(512, activation="relu")(x)  # Lớp Dense với 512 neurons
        x = Dropout(0.3)(x)  # Dropout
        x = Dense(128, activation="relu")(x)  # Lớp Dense với 128 neurons
        x = Dropout(0.3)(x)  # Dropout
        outputs = Dense(num_classes, activation="softmax")(x)  # Lớp đầu ra với activation 'softmax' cho phân loại đa lớp

        # Tạo mô hình hoàn chỉnh
        model = Model(inputs=base_model.input, outputs=outputs)

        # Biên dịch mô hình
        model.compile(optimizer=Adamax(learning_rate=self.learning_rate),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])  # Thêm 'accuracy' vào metrics

        return model