import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# USERNAME
# ==========================================

username = input("Enter username: ").strip()

embedding_path = f"data/embeddings/{username}.npy"

if not os.path.exists(embedding_path):
    print(f"User '{username}' not found")
    exit()

# ==========================================
# LOAD STORED EMBEDDING
# ==========================================

stored_embedding = np.load(embedding_path)

print("Stored embedding loaded")
print("Shape:", stored_embedding.shape)

# ==========================================
# LOAD INSIGHTFACE
# ==========================================

print("\nLoading InsightFace...")

app = FaceAnalysis()
app.prepare(ctx_id=-1)

print("InsightFace loaded")

# ==========================================
# OPEN CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

print("\nPress S to authenticate")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    display_frame = frame.copy()

    cv2.putText(
        display_frame,
        f"User: {username}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        "Press S to Authenticate",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Authentication", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s"):

        print("\nCapturing face...")
        print("Generating embedding...")

        faces = app.get(frame)

        if len(faces) == 0:
            print("No face detected")
            continue

        if len(faces) > 1:
            print("Multiple faces detected")
            continue

        live_embedding = faces[0].embedding

        similarity = cosine_similarity(
            stored_embedding.reshape(1, -1),
            live_embedding.reshape(1, -1)
        )[0][0]

        print("\n========================")
        print(f"Similarity: {similarity:.4f}")

        THRESHOLD = 0.60

        if similarity > THRESHOLD:
            print("AUTHENTICATED")
        else:
            print("REJECTED")

        print("========================\n")

cap.release()
cv2.destroyAllWindows()