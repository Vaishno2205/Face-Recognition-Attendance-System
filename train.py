import cv2
import os
import numpy as np
import pickle 

dataset_path = "dataset"

faces = []
labels = []

label_map = {}
current_id = 0

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_map[current_id] = person_name

    print(f"\nProcessing: {person_name}")

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        img = cv2.imread(image_path)

        if img is None:
            print(f"Could not read: {image_name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(50, 50)
        )

        print(f"{image_name} -> Faces Found: {len(detected_faces)}")

        for (x, y, w, h) in detected_faces:

            face = gray[y:y+h, x:x+w]

            # Resize face to fixed size
            face = cv2.resize(face, (200, 200))

            faces.append(face)
            labels.append(current_id)

    current_id += 1

print("\n------------------------")
print("Faces collected:", len(faces))
print("Labels collected:", len(labels))
print("------------------------")

if len(faces) == 0:
    print("\nERROR: No faces detected in dataset images.")
    print("Check your images or face detection settings.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(labels)
)

recognizer.save("trainer.yml")
with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("\nModel trained successfully!")
print("People found:", label_map)