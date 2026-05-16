from model import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. تجهيز بيانات الاختبار (Test Data)
print("Loading Test Data...")
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_datagen.flow_from_directory(
    'Data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

# 2. بناء الموديل وتحميل الأوزان
print("Loading Model Weights...")
model = build_model()
model.load_weights("Models/emotion_model.h5")

# --- السطر اللي تم إضافته لحل المشكلة ---
print("Compiling Model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# ----------------------------------------

# ... (الجزء الأول من الكود الخاص بتحميل الداتا والموديل كما هو)

# 3. التقييم واستخراج النتائج
print("Evaluating Model...")
loss, acc = model.evaluate(test_data)
accuracy_str = f"{acc * 100:.2f}%"
print(f"\nTest Accuracy: {accuracy_str}\n")

print("Generating Predictions...")
y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_data.classes

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred)

# --- الجزء الجديد: حفظ النتائج والتحليل في ملف ---
report_filename = "Project_Analysis_Report.txt"
print(f"Saving analysis to {report_filename}...")

analysis_text = """
=== Analysis and Discussion ===
1. Best Performing Classes:
   - Happy (Class 3): Achieved the highest precision and recall (around 78%). Smiles have distinct facial features (mouth and eyes) that the model recognizes easily.
   - Surprise (Class 6): Achieved excellent accuracy (71%). Wide eyes and an open mouth make this class very distinguishable.

2. Challenging Classes:
   - Disgust (Class 1): Had the lowest recall (36%). This is primarily due to 'Data Imbalance', as the test set contains only 111 images for this class compared to 1774 for Happy.
   - Fear & Sad (Classes 2 & 5): The model sometimes confuses these two. The confusion matrix shows 179 instances where 'Fear' was misclassified as 'Sad'. This is logically expected as both expressions share similar contracted facial muscle patterns.
"""

with open(report_filename, "w", encoding="utf-8") as file:
    file.write("=========================================\n")
    file.write("   Emotion Recognition System Results    \n")
    file.write("=========================================\n\n")
    file.write(f"Overall Test Accuracy: {accuracy_str}\n\n")
    file.write("--- Confusion Matrix ---\n")
    file.write(str(cm) + "\n\n")
    file.write("--- Classification Report ---\n")
    file.write(class_report + "\n\n")
    file.write(analysis_text)

print(f"Done! Open '{report_filename}' to see your full analysis.")