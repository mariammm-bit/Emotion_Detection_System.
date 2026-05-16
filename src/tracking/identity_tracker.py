# Simple placeholder (advanced version uses embeddings)

identity_db = {}

def assign_identity(person_id, face_img):
    identity_db[person_id] = face_img
    return person_id