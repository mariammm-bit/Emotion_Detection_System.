from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization

def build_model():
    # استدعاء الموديل الأساسي بدون الطبقات الأخيرة
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(48, 48, 3)
    )

    # السماح بتدريب آخر 4 طبقات فقط لتتأقلم مع ملامح الوجوه
    base_model.trainable = True
    for layer in base_model.layers[:-4]: 
        layer.trainable = False

    # بناء النظام المتكامل (Unified System)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(), # لتحسين استقرار التدريب
        Dropout(0.5), # لتقليل الـ Overfitting
        Dense(7, activation='softmax') # 7 فئات للمشاعر
    ])

    return model