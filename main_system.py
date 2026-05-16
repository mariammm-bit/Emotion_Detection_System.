import cv2
import numpy as np

# ===== Detection =====
from src.detection.emotion_model import predict_emotion
from src.detection.face_mesh import FaceMeshDetector

# ===== Tracking =====
from src.tracking.multi_face_tracker import CentroidTracker
from src.tracking.identity_tracker import assign_identity

# ===== Analysis =====
from src.analysis.emotion_memory import update_memory, get_history
from src.analysis.trend import get_trend
from src.analysis.heatmap import update_heatmap, overlay_heatmap

# ===== AI =====
from src.ai.recommendation import recommend

# ===== UI =====
from src.ui.hud import draw_box, draw_text

# =========================
# 🚀 INIT SYSTEM
# =========================
detector = FaceMeshDetector()
tracker = CentroidTracker()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found")
    exit()

print("🚀 Emotion AI System Running... Press ESC to exit")

# FPS tracking
prev_time = 0

# =========================
# 🎥 MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # =========================
    # 👁️ DETECT FACES
    # =========================
    faces = detector.detect(frame)

    rects = []

    if faces:
        for face in faces:
            xs = [int(lm.x * w) for lm in face.landmark]
            ys = [int(lm.y * h) for lm in face.landmark]

            x, y = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)

            rects.append((x, y, x2 - x, y2 - y))

    # =========================
    # 🔁 TRACK
    # =========================
    objects = tracker.update(rects)

    # =========================
    # 🧠 PROCESS
    # =========================
    for (person_id, (cx, cy)), rect in zip(objects.items(), rects):
        x, y, w_box, h_box = rect

        x = max(0, x)
        y = max(0, y)
        w_box = max(1, w_box)
        h_box = max(1, h_box)

        face_crop = frame[y:y+h_box, x:x+w_box]

        if face_crop.size == 0:
            continue

        emotion, conf = predict_emotion(face_crop)

        assign_identity(person_id, face_crop)

        update_memory(person_id, emotion)
        history = get_history(person_id)

        trend = get_trend(history)

        rec = recommend(emotion)

        update_heatmap(x, y, w_box, h_box)

        draw_box(frame, x, y, w_box, h_box)
        draw_text(frame, x, y, emotion, conf, trend, rec)

        cv2.putText(frame,
                    f"ID: {person_id}",
                    (x, y + h_box + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2)

    # =========================
    # 🔥 HEATMAP
    # =========================
   # frame = overlay_heatmap(frame)

    # =========================
    # ⚡ FPS
    # =========================
    curr_time = cv2.getTickCount() / cv2.getTickFrequency()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    # =========================
    # 🎬 SHOW
    # =========================
    cv2.imshow("Emotion AI System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# 🧹 CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()