import cv2
import csv
import pickle
from datetime import datetime
import time

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Load labels
with open("labels.pkl", "rb") as f:
    names = pickle.load(f)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_start_time = {}

csv_file = "attendance/Attendance.csv"

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    current_time = time.time()

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < 80:

            name = names.get(label, "Unknown")
            color = (0, 255, 0)

            if name not in face_start_time:
                face_start_time[name] = current_time

            elapsed = current_time - face_start_time[name]

            attendance_status = ""

            if elapsed >= 3:

                now = datetime.now()

                date = now.strftime("%d-%m-%Y")
                time_now = now.strftime("%H:%M:%S")

                already_marked = False

                try:
                    with open(csv_file, "r", newline="") as file:

                        reader = csv.reader(file)

                        next(reader, None)

                        for row in reader:

                            if len(row) >= 2:

                                saved_name = row[0]
                                saved_date = row[1]

                                if saved_name == name and saved_date == date:
                                    already_marked = True
                                    break

                except FileNotFoundError:
                    pass

                if not already_marked:

                    with open(csv_file, "a", newline="") as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            name,
                            date,
                            time_now
                        ])

                    attendance_status = "Attendance Marked"
                    print(f"Attendance Marked for {name}")

                else:

                    attendance_status = "Present Today"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{name} ({int(elapsed)}s)",
                (x, y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            if attendance_status:

                cv2.putText(
                    frame,
                    attendance_status,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        else:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "Unknown",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()