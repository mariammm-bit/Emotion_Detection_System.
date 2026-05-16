import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh

class FaceMeshDetector:
    def __init__(self):
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        return results.multi_face_landmarks