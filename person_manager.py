import os
import shutil
import cv2

from config import DATASET_DIR


class PersonManager:

    def __init__(self):

        os.makedirs(DATASET_DIR, exist_ok=True)

    # ==========================================================
    # Create Person Folder
    # ==========================================================

    def create_person(self, person_name):

        person_name = person_name.strip()

        if person_name == "":
            return None

        person_path = os.path.join(
            DATASET_DIR,
            person_name
        )

        os.makedirs(person_path, exist_ok=True)

        return person_path

    # ==========================================================
    # Save Image
    # ==========================================================

    def save_image(self, person_name, frame):

        person_path = self.create_person(person_name)

        if person_path is None:
            return False

        image_count = len([
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        filename = os.path.join(
            person_path,
            f"{image_count + 1}.jpg"
        )

        cv2.imwrite(filename, frame)

        return True

    # ==========================================================
    # Save Multiple Images
    # ==========================================================

    def save_images(self,
                    person_name,
                    frames):

        saved = 0

        for frame in frames:

            if self.save_image(person_name, frame):

                saved += 1

        return saved

    # ==========================================================
    # Get Person List
    # ==========================================================

    def get_people(self):

        people = []

        for item in os.listdir(DATASET_DIR):

            path = os.path.join(
                DATASET_DIR,
                item
            )

            if os.path.isdir(path):

                people.append(item)

        people.sort()

        return people

    # ==========================================================
    # Delete Person
    # ==========================================================

    def delete_person(self, person_name):

        person_path = os.path.join(
            DATASET_DIR,
            person_name
        )

        if os.path.exists(person_path):

            shutil.rmtree(person_path)

            return True

        return False

    # ==========================================================
    # Count Images
    # ==========================================================

    def image_count(self,
                    person_name):

        person_path = os.path.join(
            DATASET_DIR,
            person_name
        )

        if not os.path.exists(person_path):

            return 0

        count = len([

            f for f in os.listdir(person_path)

            if f.lower().endswith(
                (
                    ".jpg",
                    ".png",
                    ".jpeg"
                )
            )

        ])

        return count

    # ==========================================================
    # Dataset Statistics
    # ==========================================================

    def dataset_info(self):

        info = {}

        for person in self.get_people():

            info[person] = self.image_count(person)

        return info