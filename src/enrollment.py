import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# ==========================================
# USERNAME
# ==========================================

username = input("Enter username: ").strip()

if not username:
    print("Invalid username")
    exit()

# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs("data/embeddings", exist_ok=True)

embedding_path = f"data/embeddings/{username}.npy"

# ==========================================
# LOAD INSIGHTFACE
# ==========================================

print("Loading InsightFace...")

app = FaceAnalysis()
app.prepare(ctx_id=-1)

print("InsightFace loaded successfully")

# ==========================================
# OPEN CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

print("\nPress S to capture face")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    display_frame = frame.copy()

    cv2.putText(
        display_frame,
        "Press S to Capture",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"User: {username}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("Face Enrollment", display_frame)

    key = cv2.waitKey(1) & 0xFF

    print("Key:", key)

    # Quit
    if key == ord("q"):
        print("Enrollment cancelled")
        break

    # Capture
    if key == ord("s"):

        print("\nCapturing face...")
        print("Running InsightFace...")

        faces = app.get(frame)

        if len(faces) == 0:

            print("No face detected")
            continue

        if len(faces) > 1:

            print("Multiple faces detected")
            continue

        face = faces[0]

        embedding = face.embedding

        np.save(
            embedding_path,
            embedding
        )

        print("\n================================")
        print("Enrollment Successful")
        print(f"Saved: {embedding_path}")
        print(f"Embedding Shape: {embedding.shape}")
        print("================================")

        break

cap.release()
cv2.destroyAllWindows()