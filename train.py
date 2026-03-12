import os
import math
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU,
    GlobalAveragePooling2D, Dense, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# ============================
# CONFIG
# ============================

DATASET_PATH = "synthetic_dataset"
LETTERS = ["OMEGA", "PHI", "PSI", "EMPTY"]
IMG_SIZE = 128

# ============================
# LOAD DATASET
# ============================

def load_dataset():
    X, y = [], []
    for idx, letter in enumerate(LETTERS):
        folder = os.path.join(DATASET_PATH, letter)
        if not os.path.isdir(folder):
            print(f"Cartella mancante: {folder}")
            continue
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, -1)
            X.append(img)
            y.append(idx)
    X = np.array(X)
    y = np.array(y)
    print("Dataset shape:", X.shape, "Labels shape:", y.shape)
    return X, y

X, y = load_dataset()

# split con stratify sui label interi (NON one-hot)
X_train, X_test, y_train_int, y_test_int = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# one-hot dopo lo split
y_train = to_categorical(y_train_int, len(LETTERS))
y_test = to_categorical(y_test_int, len(LETTERS))

# ============================
# CUSTOM GAUSSIAN BLUR LAYER
# ============================

class GaussianBlur(Layer):
    def __init__(self, kernel_size=3, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def build(self, input_shape):
        ax = np.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * self.sigma**2))
        kernel = kernel / np.sum(kernel)
        kernel = kernel[:, :, np.newaxis, np.newaxis]
        self.kernel = tf.constant(kernel, dtype=tf.float32)

    def call(self, x):
        return tf.nn.depthwise_conv2d(
            x, self.kernel, strides=[1, 1, 1, 1], padding="SAME"
        )

# ============================
# DATA AUGMENTATION LEGGERA
# ============================

data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.1),
    GaussianBlur(3)
], name="data_augmentation")

# ============================
# MODELLO TIPO MOBILENET
# ============================

def depthwise_block(x, filters, stride=1):
    x = DepthwiseConv2D(3, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, 1, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

inp = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = data_aug(inp)

x = Conv2D(32, 3, strides=2, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = depthwise_block(x, 64)
x = depthwise_block(x, 128, stride=2)
x = depthwise_block(x, 128)
x = depthwise_block(x, 256, stride=2)
x = depthwise_block(x, 256)

x = GlobalAveragePooling2D()(x)
out = Dense(len(LETTERS), activation="softmax")(x)

model = Model(inp, out)

# ============================
# LEARNING RATE SCHEDULER
# ============================

initial_lr = 1e-3
warmup_epochs = 5
total_epochs = 80

def lr_schedule(epoch):
    # warmup lineare
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    # cosine decay puro in Python (niente tensori)
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    lr = 0.5 * initial_lr * (1 + math.cos(math.pi * progress))
    return lr

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# ============================
# TRAINING
# ============================

early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
lr_sched = LearningRateScheduler(lr_schedule)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=total_epochs,
    batch_size=64,
    callbacks=[early, lr_sched]
)

# ============================
# TFLITE INT8 EXPORT
# ============================

def representative_dataset():
    # qualche centinaio di esempi basta per la calibrazione
    for i in range(200):
        idx = np.random.randint(0, len(X_train))
        img = X_train[idx:idx+1].astype(np.float32)
        yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("greek_letters_model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("Modello TFLite INT8 salvato come greek_letters_model_int8.tflite")
print("Classi nell'ordine:", LETTERS)