import cv2
import mediapipe as mp
import urllib.request
import os
import math

# Download face landmarker model
model_path = "models/face_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("Download complete")

# Eye landmarks for blink detection
LEFT_EYE = [33, 160, 158, 133, 153, 144]

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )

# Setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

cap = cv2.VideoCapture(0)

print("Blink Detection Test")
print("Showing eye landmarks only")
print("Press Q to quit")

with FaceLandmarker.create_from_options(options) as landmarker:

    blink_count = 0
    eye_closed_frames = 0

    EAR_THRESHOLD = 0.20
    MIN_CLOSED_FRAMES = 2

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = landmarker.detect(mp_image)

        if results.face_landmarks:

            landmarks = results.face_landmarks[0]

            eye_points = []

            # Draw eye landmarks and store coordinates
            for idx in LEFT_EYE:

                landmark = landmarks[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                eye_points.append((x, y))

                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                cv2.putText(
                    frame,
                    str(idx),
                    (x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1
                )

            # Unpack eye points
            p1, p2, p3, p4, p5, p6 = eye_points

            # Draw eye shape
            cv2.line(frame, p1, p4, (255, 0, 0), 2)
            cv2.line(frame, p2, p6, (255, 0, 0), 2)
            cv2.line(frame, p3, p5, (255, 0, 0), 2)

            # Calculate EAR
            vertical_1 = distance(p2, p6)
            vertical_2 = distance(p3, p5)

            horizontal = distance(p1, p4)

            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        if ear < EAR_THRESHOLD:

            eye_closed_frames += 1

            status = "EYE CLOSED"
            color = (0, 0, 255)

        else:

            if eye_closed_frames >= MIN_CLOSED_FRAMES:
                blink_count += 1

            eye_closed_frames = 0

            status = "EYE OPEN"
            color = (0, 255, 0)

        print(f"EAR={ear:.3f}, ClosedFrames={eye_closed_frames}, Blinks={blink_count}")
        cv2.putText(
                        frame,
                        status,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2
                    )

        cv2.putText(
                        frame,
                        f"Blinks: {blink_count}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

        cv2.imshow("Blink Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Done")