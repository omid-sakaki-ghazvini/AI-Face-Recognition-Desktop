import os
import json
import cv2
import numpy as np

from config import (
    DATASET_DIR,
    TRAINER_DIR,
    MODEL_PATH,
    LABEL_PATH,
    FACE_SIZE,
    CASCADE_PATH
)


class FaceTrainer:

    def __init__(self):

        self.detector = cv2.CascadeClassifier(CASCADE_PATH)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8,
            threshold=80
        )

        self.faces = []

        self.labels = []

        self.label_dict = {}

    ############################################################

    def train(self):

        print("=" * 60)
        print("Face Training Started...")
        print("=" * 60)

        current_label = 0

        people = sorted(os.listdir(DATASET_DIR))

        for person in people:

            person_path = os.path.join(
                DATASET_DIR,
                person
            )

            if not os.path.isdir(person_path):
                continue

            print(f"\nPerson : {person}")

            self.label_dict[current_label] = person

            images = os.listdir(person_path)

            for image_name in images:

                image_path = os.path.join(
                    person_path,
                    image_name
                )

                image = cv2.imread(image_path)

                if image is None:
                    continue

                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

                faces = self.detector.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(80, 80)
                )

                if len(faces) == 0:

                    print(f"  No face : {image_name}")

                    continue

                x, y, w, h = max(
                    faces,
                    key=lambda f: f[2] * f[3]
                )

                roi = gray[y:y+h, x:x+w]

                roi = cv2.resize(
                    roi,
                    FACE_SIZE
                )

                self.faces.append(roi)

                self.labels.append(current_label)

                print(f"  Added : {image_name}")

            current_label += 1

        print("\nTraining Model...")

        if len(self.faces) == 0:

            raise Exception(
                "No faces found in dataset."
            )

        self.recognizer.train(
            self.faces,
            np.array(self.labels)
        )

        self.recognizer.save(
            MODEL_PATH
        )

        with open(
                LABEL_PATH,
                "w",
                encoding="utf-8"
        ) as f:

            json.dump(
                self.label_dict,
                f,
                indent=4,
                ensure_ascii=False
            )

        print()

        print("=" * 60)
        print("Training Finished Successfully")
        print("=" * 60)

        print(f"Faces   : {len(self.faces)}")
        print(f"Persons : {len(self.label_dict)}")

        print()

        print("Model Saved :")
        print(MODEL_PATH)

        print()

        print("Labels Saved :")
        print(LABEL_PATH)


###############################################################

if __name__ == "__main__":

    trainer = FaceTrainer()

    trainer.train()