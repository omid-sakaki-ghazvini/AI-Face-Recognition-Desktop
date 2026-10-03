import os
import cv2
import pickle
import numpy as np

from config import (
    DATASET_DIR,
    DATABASE_DIR,
    DATABASE_FILE,
    SFACE_MODEL,
    FACE_SIZE
)

from face_detector import FaceDetector



class FaceDatabase:


    def __init__(self):

        self.detector = FaceDetector()


        self.recognizer = cv2.FaceRecognizerSF.create(

            SFACE_MODEL,

            ""

        )


        if not os.path.exists(DATABASE_DIR):

            os.makedirs(
                DATABASE_DIR
            )



    def extract_face_embedding(self, image):


        faces = self.detector.detect(
            image
        )


        if len(faces) == 0:

            return None



        face = faces[0]


        aligned_face = self.recognizer.alignCrop(

            image,

            face

        )


        embedding = self.recognizer.feature(

            aligned_face

        )


        return embedding



    def build_database(self):


        database = []


        print("="*50)

        print("Building Face Database")

        print("="*50)



        for person_name in os.listdir(DATASET_DIR):


            person_path = os.path.join(

                DATASET_DIR,

                person_name

            )


            if not os.path.isdir(person_path):

                continue



            print(
                "\nPerson:",
                person_name
            )


            images = os.listdir(
                person_path
            )


            count = 0



            for image_name in images:


                image_path = os.path.join(

                    person_path,

                    image_name

                )


                image = cv2.imread(
                    image_path
                )


                if image is None:

                    continue



                embedding = self.extract_face_embedding(

                    image

                )


                if embedding is not None:


                    database.append(

                        {

                            "name": person_name,

                            "embedding": embedding

                        }

                    )


                    count += 1


                    print(

                        " Added:",

                        image_name

                    )


            print(

                "Images processed:",

                count

            )



        if len(database) == 0:


            raise Exception(

                "No face embeddings found"

            )



        with open(
            DATABASE_FILE,
            "wb"
        ) as f:


            pickle.dump(

                database,

                f

            )



        print("\nDatabase saved:")

        print(DATABASE_FILE)

        print(

            "Total faces:",

            len(database)

        )




if __name__ == "__main__":


    db = FaceDatabase()

    db.build_database()