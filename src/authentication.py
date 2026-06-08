import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

app = FaceAnalysis()
app.prepare(ctx_id=-1)

THRESHOLD = 0.70

def authenticate(frame, username):

    embedding_path = f"data/embeddings/{username}.npy"

    if not os.path.exists(embedding_path):
        return False, 0.0, "User not found"

    stored_embedding = np.load(embedding_path)

    faces = app.get(frame)

    if len(faces) == 0:
        return False, 0.0, "No face detected"

    if len(faces) > 1:
        return False, 0.0, "Multiple faces detected"

    live_embedding = faces[0].embedding

    similarity = cosine_similarity(
        stored_embedding.reshape(1, -1),
        live_embedding.reshape(1, -1)
    )[0][0]

    authenticated = similarity > THRESHOLD

    return authenticated, similarity, "OK"