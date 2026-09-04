import cv2
import os
import sys

if len(sys.argv) < 2:
    print("Please provide student name")
    sys.exit()

student_name = sys.argv[1]

folder_path = f"dataset/{student_name}"

os.makedirs(folder_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face_img = gray[y:y+h, x:x+w]

    cv2.putText(
        frame,
        f"Images Saved: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Register Student", frame)

    key = cv2.waitKey(1)

    if key == ord('s') and len(faces) > 0:

        count += 1

        face_img = cv2.resize(face_img, (200, 200))

        filename = os.path.join(
            folder_path,
            f"{count}.png"
        )

        cv2.imwrite(filename, face_img)

        print(f"Saved: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()