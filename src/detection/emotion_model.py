import numpy as np
import cv2
from Notebooks.model import build_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# تم تصليح المسار هنا عشان ميضربش Error
MODEL_PATH = os.path.join(BASE_DIR, "Models", "emotion_model.h5")

model = build_model()
model.load_weights(MODEL_PATH)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 1. رجعنا فلتر تحسين الإضاءة اللي الموديل بيحبه
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def predict_emotion(face_img):
    # 2. تحويل الوش لأبيض وأسود وتحسين إضاءته قبل التنبؤ
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    roi_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    # 3. التجهيز للموديل
    face = cv2.resize(roi_rgb, (48, 48))
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face)

    preds = model.predict(face, verbose=0)[0]
    
    # 4. رجعنا الـ Boost factors عشان نعلي دقة المشاعر الصعبة
    boost_factors = np.array([1.5, 2.0, 1.5, 0.8, 0.6, 1.5, 1.0])
    adjusted_preds = preds * boost_factors
    adjusted_preds = adjusted_preds / np.sum(adjusted_preds) # Normalization

    emotion = emotion_labels[np.argmax(adjusted_preds)]
    confidence = float(np.max(adjusted_preds))

    return emotion, confidence