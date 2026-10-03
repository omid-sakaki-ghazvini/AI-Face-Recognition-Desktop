import cv2
import os
import pickle
import time

from config import (
    DATABASE_FILE,
    DATASET_DIR,
    SFACE_MODEL
)

from face_detector import FaceDetector



class FaceEnrollment:


    def __init__(self):

        self.detector = FaceDetector()


        self.recognizer = cv2.FaceRecognizerSF.create(

            SFACE_MODEL,

            ""

        )



    def capture_embeddings(self, name, samples=30):


        embeddings = []


        cap = cv2.VideoCapture(0)


        count = 0


        print("Starting enrollment...")


        while count < samples:


            ret, frame = cap.read()


            if not ret:
                break



            faces = self.detector.detect(frame)



            if len(faces) > 0:


                face = faces[0]



                aligned = self.recognizer.alignCrop(

                    frame,

                    face

                )


                embedding = self.recognizer.feature(

                    aligned

                )



                embeddings.append(

                    embedding

                )


                count += 1



                cv2.rectangle(

                    frame,

                    (
                        int(face[0]),
                        int(face[1])
                    ),

                    (
                        int(face[0]+face[2]),
                        int(face[1]+face[3])
                    ),

                    (0,255,0),

                    2

                )


                cv2.putText(

                    frame,

                    f"Captured {count}/{samples}",

                    (20,40),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1,

                    (0,255,0),

                    2

                )



            cv2.imshow(

                "Register Person",

                frame

            )


            if cv2.waitKey(1) == 27:

                break



        cap.release()

        cv2.destroyAllWindows()



        if len(embeddings) == 0:

            raise Exception(
                "No face detected"
            )


        self.save_person(

            name,

            embeddings

        )



    # --------------------------------

    def save_person(self, name, embeddings):


        database = []



        if os.path.exists(DATABASE_FILE):


            with open(

                DATABASE_FILE,

                "rb"

            ) as f:

                database = pickle.load(f)



        for emb in embeddings:


            database.append(

                {

                    "name": name,

                    "embedding": emb

                }

            )



        with open(

            DATABASE_FILE,

            "wb"

        ) as f:


            pickle.dump(

                database,

                f

            )


        print(
            "Person added:",
            name
        )


        print(
            "Total embeddings:",
            len(database)
        )