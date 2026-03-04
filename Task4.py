import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ============================
# SETTINGS
# ============================

DATA_PATH = r"C:\Users\HP\Downloads\archive (5)\leapGestRecog\leapGestRecog"
IMG_SIZE = 128
BATCH_SIZE = 32

# ============================
# LOAD DATA
# ============================

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="both",
    seed=42
)

train_ds, val_ds = dataset

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# Normalize
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))

# ============================
# BUILD CNN MODEL
# ============================

model = models.Sequential([
    
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================
# TRAIN MODEL
# ============================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ============================
# SAVE MODEL
# ============================

model.save("hand_gesture_model.h5")
