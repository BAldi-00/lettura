import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

DATASET_PATH = "dataset_greek"

# Solo le tre lettere che ti servono
greek_letters = ["PHI", "PSI", "OMEGA"]

IMG_SIZE = 128

def load_dataset():
    X = []
    y = []

    for idx, letter in enumerate(greek_letters):
        folder = os.path.join(DATASET_PATH, letter)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=-1)

            X.append(img)
            y.append(idx)

    X = np.array(X)
    y = np.array(y)
    return X, y


print("Caricamento dataset...")
X, y = load_dataset()
print("Dataset:", X.shape, y.shape)

# One-hot encoding
y = to_categorical(y, num_classes=len(greek_letters))

# Train/test split stratificato
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# CNN migliorata
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(greek_letters), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("Inizio training...")
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

model.save("greek_letters_model_folder")
print("Modello salvato come greek_3letters_model")
