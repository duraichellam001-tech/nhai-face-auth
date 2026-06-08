import cv2
import mediapipe as mp
import urllib.request
import os

# Download face landmarker model
model_path = "models/face_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("Download complete")

# Head pose landmarks
NOSE = 1
LEFT_FACE = 234
RIGHT_FACE = 454

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

print("Head Pose Test")
print("Turn your head left and right")
print("Press Q to quit")

with FaceLandmarker.create_from_options(options) as landmarker:

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

            # Get landmark coordinates
            nose = landmarks[NOSE]
            left_face = landmarks[LEFT_FACE]
            right_face = landmarks[RIGHT_FACE]

            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            left_x = int(left_face.x * w)
            left_y = int(left_face.y * h)

            right_x = int(right_face.x * w)
            right_y = int(right_face.y * h)

            # Draw landmarks
            cv2.circle(frame, (nose_x, nose_y), 8, (0, 0, 255), -1)
            cv2.circle(frame, (left_x, left_y), 8, (255, 0, 0), -1)
            cv2.circle(frame, (right_x, right_y), 8, (0, 255, 0), -1)

            # Draw lines
            cv2.line(
                frame,
                (left_x, left_y),
                (nose_x, nose_y),
                (255, 255, 0),
                2
            )

            cv2.line(
                frame,
                (nose_x, nose_y),
                (right_x, right_y),
                (255, 255, 0),
                2
            )

            # Calculate distances
            left_distance = abs(nose_x - left_x)
            right_distance = abs(right_x - nose_x)

            # Display values
            cv2.putText(
                frame,
                f"Left: {left_distance}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right: {right_distance}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Ratio
            ratio = left_distance / (left_distance + right_distance)

            cv2.putText(
                frame,
                f"Ratio: {ratio:.2f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # Head direction
            if ratio < 0.35:

                direction = "LOOKING RIGHT"

            elif ratio > 0.65:

                direction = "LOOKING LEFT"

            else:

                direction = "LOOKING CENTER"
                color = (255, 255, 255)

            cv2.putText(
                frame,
                direction,
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        else:

            cv2.putText(
                frame,
                "NO FACE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("Head Pose Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Done")