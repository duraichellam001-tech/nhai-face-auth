# Offline Facial Authentication System

## Features

- Face Detection
- Face Mesh Tracking
- Blink Detection
- Head Pose Estimation
- Active Liveness Verification
- Face Enrollment
- Face Authentication using InsightFace

## Tech Stack

- Python
- OpenCV
- MediaPipe
- InsightFace
- ONNX Runtime
- NumPy
- Scikit-learn

## Workflow

Enrollment
→ Generate 512D Face Embedding
→ Save Embedding

Authentication
→ Blink Detection
→ Head Left
→ Head Right
→ Generate Embedding
→ Cosine Similarity
→ Access Granted / Denied