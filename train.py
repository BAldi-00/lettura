import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

# ============================
# CONFIG
# ============================

DATASET_PATH = "synthetic_dataset"   # cartella con OMEGA/PHI/PSI/EMPTY
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
            print(f"Attenzione: cartella {folder} non trovata, salto.")
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
y = to_categorical(y, len(LETTERS))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# ============================
# MODELLO CNN
# ============================

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(LETTERS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    "best_model.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max"
)

early = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=32,
    callbacks=[checkpoint, early, reduce_lr]
)

# ============================
# SALVATAGGIO MODELLO TF
# ============================

model.save("greek_letters_model_folder", save_format="tf")

# ============================
# ESPORTAZIONE TFLITE
# ============================

converter = tf.lite.TFLiteConverter.from_saved_model("greek_letters_model_folder")
# opzionale: ottimizzazioni per embedded
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("greek_letters_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Salvati:")
print("- best_model.keras")
print("- cartella SavedModel: greek_letters_model_folder")
print("- modello TFLite: greek_letters_model.tflite")
print("Classi nell'ordine:", LETTERS)
