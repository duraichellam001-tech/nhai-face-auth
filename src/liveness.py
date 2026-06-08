import cv2
import mediapipe as mp
import urllib.request
import os
import math
from authentication import authenticate
import time

# =====================================================
# MODEL
# =====================================================

model_path = "models/face_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )
    print("Download complete")

# =====================================================
# LANDMARKS
# =====================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]

NOSE = 1
LEFT_FACE = 234
RIGHT_FACE = 454

# =====================================================
# HELPERS
# =====================================================

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )

# =====================================================
# MEDIAPIPE
# =====================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# =====================================================
# CAMERA
# =====================================================

username = input("Enter username: ").strip()
cap = cv2.VideoCapture(0)

print("Liveness Test")
print("1. Blink")
print("2. Turn Left")
print("3. Turn Right")
print("Press Q to quit")

# =====================================================
# LIVENESS VARIABLES
# =====================================================

blink_count = 0
eye_closed_frames = 0

EAR_THRESHOLD = 0.20
MIN_CLOSED_FRAMES = 2

blink_done = False
left_done = False
right_done = False

liveness_passed = False
authenticated = False
similarity = 0.0
auth_message = ""

auth_started = False

# =====================================================
# MAIN LOOP
# =====================================================

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

            # =========================================
            # BLINK DETECTION
            # =========================================

            eye_points = []

            for idx in LEFT_EYE:

                landmark = landmarks[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                eye_points.append((x, y))

                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            p1, p2, p3, p4, p5, p6 = eye_points

            vertical_1 = distance(p2, p6)
            vertical_2 = distance(p3, p5)

            horizontal = distance(p1, p4)

            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

            if ear < EAR_THRESHOLD:

                eye_closed_frames += 1

                eye_status = "EYE CLOSED"
                eye_color = (0, 0, 255)

            else:

                if eye_closed_frames >= MIN_CLOSED_FRAMES:

                    blink_count += 1

                    if not blink_done:
                        blink_done = True

                eye_closed_frames = 0

                eye_status = "EYE OPEN"
                eye_color = (0, 255, 0)

            # =========================================
            # HEAD DIRECTION
            # =========================================

            nose = landmarks[NOSE]
            left_face = landmarks[LEFT_FACE]
            right_face = landmarks[RIGHT_FACE]

            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            left_x = int(left_face.x * w)
            left_y = int(left_face.y * h)

            right_x = int(right_face.x * w)
            right_y = int(right_face.y * h)

            cv2.circle(frame, (nose_x, nose_y), 6, (0, 0, 255), -1)
            cv2.circle(frame, (left_x, left_y), 6, (255, 0, 0), -1)
            cv2.circle(frame, (right_x, right_y), 6, (0, 255, 0), -1)

            left_distance = abs(nose_x - left_x)
            right_distance = abs(right_x - nose_x)

            ratio = left_distance / (
                left_distance + right_distance
            )

            if ratio < 0.35:

                direction = "LOOKING RIGHT"

            elif ratio > 0.65:

                direction = "LOOKING LEFT"

            else:

                direction = "LOOKING CENTER"

            # =========================================
            # LIVENESS STATE MACHINE
            # =========================================

            if blink_done:

                if direction == "LOOKING LEFT":
                    left_done = True

                if left_done and direction == "LOOKING RIGHT":
                    right_done = True

            if (
                blink_done
                and left_done
                and right_done
                and not liveness_passed
            ):

                liveness_passed = True

                print("\n================================")
                print("LIVENESS PASSED")
                print("PLEASE LOOK STRAIGHT")
                print("Authentication in 2 seconds...")
                print("================================")

                auth_message = "LOOK CENTER"

                cv2.putText(
                    frame,
                    "LOOK CENTER",
                    (20, 420),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3
                )

                cv2.imshow("Liveness Test", frame)

                cv2.waitKey(1)

                time.sleep(2)

                # Capture fresh frame after user returns to center
                ret, frame = cap.read()

                if not ret:
                    print("Failed to capture frame")
                    break

                print("Authenticating...")

                authenticated, similarity, auth_message = authenticate(
                    frame,
                    username
                )

                print(f"Similarity: {similarity:.4f}")

                if authenticated:

                    auth_message = "ACCESS GRANTED"

                    print("\nACCESS GRANTED")
                    cv2.imshow("Liveness Test", frame)
                    cv2.waitKey(3000)
                    break

                else:

                    auth_message = "ACCESS DENIED"

                    print("\nACCESS DENIED")
                    cv2.imshow("Liveness Test", frame)
                    cv2.waitKey(3000)
                    break

            # if (
            #     blink_done
            #     and left_done
            #     and right_done
            #     and not liveness_passed
            # ):

            #     liveness_passed = True

            #     print("\n================================")
            #     print("LIVENESS PASSED")
            #     print("Authenticating...")
            #     print("================================")

            #     authenticated, similarity, auth_message = authenticate(
            #         frame,
            #         username
            #     )

            #     print(f"Similarity: {similarity:.4f}")

            #     if authenticated:

            #         auth_message = "ACCESS GRANTED"

            #         print("\nACCESS GRANTED")

            #         cv2.imshow("Liveness Test", frame)
            #         cv2.waitKey(3000)
            #         break

            #     else:

            #         auth_message = "ACCESS DENIED"

            #         print("\nACCESS DENIED")

            #         cv2.imshow("Liveness Test", frame)
            #         cv2.waitKey(3000)
            #         break

            # =========================================
            # INSTRUCTION
            # =========================================

            if not blink_done:

                instruction = "BLINK ONCE"

            elif not left_done:

                instruction = "TURN HEAD LEFT"

            elif not right_done:

                instruction = "TURN HEAD RIGHT"

            elif auth_message == "":

                instruction = "AUTHENTICATING..."

            else:

                instruction = auth_message

            # =========================================
            # DISPLAY
            # =========================================

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                eye_status,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                eye_color,
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

            cv2.putText(
                frame,
                direction,
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                instruction,
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Blink:{blink_done}",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Left:{left_done}",
                (20, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right:{right_done}",
                (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Similarity: {similarity:.3f}",
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if auth_message:

                color = (0, 255, 0)

                if auth_message == "ACCESS DENIED":
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    auth_message,
                    (20, 390),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3
                )

        cv2.imshow("Liveness Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Done")