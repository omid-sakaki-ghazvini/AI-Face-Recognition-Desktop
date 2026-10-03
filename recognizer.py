import cv2
import os
import pickle
import time

from config import (
    SFACE_MODEL,
    DATABASE_FILE,
    COSINE_THRESHOLD
)

from face_detector import FaceDetector



class FaceRecognizer:


    def __init__(self):

        print("Loading Face Recognition System...")


        # Face Detector
        self.detector = FaceDetector()


        # SFace Model
        self.recognizer = cv2.FaceRecognizerSF.create(

            SFACE_MODEL,

            ""

        )


        # Load database

        self.database = self.load_database()



        # FPS

        self.prev_time = time.time()

        self.fps = 0



        print(
            "Recognition System Ready"
        )



    # ---------------------------------
    # Load Face Database
    # ---------------------------------

    def load_database(self):


        if not os.path.exists(DATABASE_FILE):

            raise Exception(

                "Database not found. Run face_database.py first."

            )



        with open(

            DATABASE_FILE,

            "rb"

        ) as f:


            database = pickle.load(f)



        print(

            "Loaded faces:",

            len(database)

        )


        return database




    # ---------------------------------
    # Extract Face Feature
    # ---------------------------------

    def extract_embedding(

            self,

            frame,

            face

    ):


        aligned_face = self.recognizer.alignCrop(

            frame,

            face

        )


        embedding = self.recognizer.feature(

            aligned_face

        )


        return embedding




    # ---------------------------------
    # Compare With Database
    # ---------------------------------

    def compare_face(

            self,

            embedding

    ):


        best_name = "Unknown"

        best_score = 0



        for item in self.database:


            db_embedding = item["embedding"]


            score = self.recognizer.match(

                embedding,

                db_embedding,

                cv2.FaceRecognizerSF_FR_COSINE

            )



            if score > best_score:


                best_score = score

                best_name = item["name"]




        if best_score < COSINE_THRESHOLD:


            best_name = "Unknown"



        return best_name, best_score




    # ---------------------------------
    # Recognition
    # ---------------------------------

    def recognize(

            self,

            frame

    ):


        results = []



        faces = self.detector.detect(

            frame

        )



        for face in faces:


            embedding = self.extract_embedding(

                frame,

                face

            )


            name, score = self.compare_face(

                embedding

            )



            results.append(

                {

                    "name": name,

                    "score": score,

                    "face": face

                }

            )



        # FPS Calculation

        current_time = time.time()


        diff = current_time - self.prev_time


        if diff > 0:


            self.fps = 1 / diff



        self.prev_time = current_time



        return results




    # ---------------------------------
    # Draw Result On Frame
    # ---------------------------------

    def draw_results(

            self,

            frame,

            results

    ):



        for result in results:



            face = result["face"]


            name = result["name"]


            score = result["score"]




            x = int(face[0])

            y = int(face[1])

            w = int(face[2])

            h = int(face[3])



            # Rectangle

            cv2.rectangle(

                frame,

                (x,y),

                (x+w,y+h),

                (0,255,0),

                2

            )



            text = f"{name} {score:.2f}"



            cv2.putText(

                frame,

                text,

                (x, y-10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0,255,0),

                2

            )




        # FPS

        cv2.putText(

            frame,

            f"FPS: {self.fps:.1f}",

            (20,30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0,255,0),

            2

        )



        # Number of faces

        cv2.putText(

            frame,

            f"Faces: {len(results)}",

            (20,65),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0,255,0),

            2

        )



        return frame