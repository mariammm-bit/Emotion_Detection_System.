"""
This module handles data loading and augmentation
for the Emotion Recognition model.

Added:
- Canny Edge Detection
- Data Augmentation
"""

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# =========================================
# CUSTOM PREPROCESSING FUNCTION
# =========================================

def custom_preprocessing(image):

    # Convert image to uint8
    image = image.astype(np.uint8)

    # =====================================
    # CANNY EDGE DETECTION
    # =====================================

    edges = cv2.Canny(image, 100, 200)

    # Convert edges back to RGB
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # Apply VGG16 preprocessing
    processed = preprocess_input(edges_rgb)

    return processed

# =========================================
# DATA GENERATORS
# =========================================

def get_data_generators():

    # Training Generator
    train_datagen = ImageDataGenerator(

        preprocessing_function=custom_preprocessing,

        rotation_range=20,

        zoom_range=0.2,

        horizontal_flip=True,

        validation_split=0.2
    )

    # Validation/Test Generator
    test_datagen = ImageDataGenerator(

        preprocessing_function=custom_preprocessing
    )

    # =====================================
    # LOAD TRAINING DATA
    # =====================================

    train_data = train_datagen.flow_from_directory(

        'Data/train',

        target_size=(48, 48),

        batch_size=64,

        color_mode='rgb',

        class_mode='categorical',

        subset='training'
    )

    # =====================================
    # LOAD VALIDATION DATA
    # =====================================

    val_data = train_datagen.flow_from_directory(

        'Data/train',

        target_size=(48, 48),

        batch_size=64,

        color_mode='rgb',

        class_mode='categorical',

        subset='validation'
    )

    return train_data, val_data