import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
else:
    print("Webcam opened successfully")
    print("You should see a window with your camera feed")
    print("Press Q to close")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        break
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed successfully")