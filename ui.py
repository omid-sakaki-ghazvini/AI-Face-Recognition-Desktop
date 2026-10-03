import sys
import cv2
from enroll import FaceEnrollment

from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QApplication,
    QMessageBox
)

from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QFont
)

from PyQt6.QtCore import (
    Qt,
    QTimer
)


from camera import Camera
from recognizer import FaceRecognizer
from face_database import FaceDatabase



class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "AI Face Recognition"
        )

        self.setGeometry(
            200,
            100,
            1000,
            750
        )


        # -----------------------
        # Objects
        # -----------------------

        self.camera = Camera()

        self.recognizer = None

        self.running = False


        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_frame
        )


        self.init_ui()



    # =============================
    # UI
    # =============================

    def init_ui(self):


        self.video_label = QLabel()

        self.video_label.setFixedSize(
            850,
            550
        )


        self.video_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )


        self.video_label.setStyleSheet(
            """
            QLabel{
                background:#111;
                border:2px solid #555;
                border-radius:15px;
            }
            """
        )



        self.status_label = QLabel(
            "System Ready"
        )


        self.status_label.setFont(
            QFont(
                "Segoe UI",
                12
            )
        )



        self.start_camera_btn = QPushButton(
            "Start Camera"
        )


        self.start_recognition_btn = QPushButton(
            "Start Recognition"
        )


        self.build_database_btn = QPushButton(
            "Build Database"
        )


        self.stop_btn = QPushButton(
            "Stop"
        )

        self.register_btn = QPushButton(
            "Register Person"
        )

        buttons = [

            self.start_camera_btn,

            self.start_recognition_btn,

            self.register_btn,

            self.build_database_btn,

            self.stop_btn

        ]


        for btn in buttons:

            btn.setMinimumHeight(
                40
            )



        self.start_camera_btn.clicked.connect(
            self.start_camera
        )


        self.start_recognition_btn.clicked.connect(
            self.start_recognition
        )


        self.build_database_btn.clicked.connect(
            self.build_database
        )


        self.stop_btn.clicked.connect(
            self.stop_camera
        )

        self.register_btn.clicked.connect(
            self.register_person
        )



        # Layout

        main_layout = QVBoxLayout()


        main_layout.addWidget(
            self.video_label
        )


        main_layout.addWidget(
            self.status_label
        )


        button_layout = QHBoxLayout()


        for btn in buttons:

            button_layout.addWidget(
                btn
            )


        main_layout.addLayout(
            button_layout
        )


        self.setLayout(
            main_layout
        )



    # =============================
    # Camera
    # =============================

    def start_camera(self):

        self.camera.start()

        self.running = True


        self.timer.start(
            30
        )


        self.status_label.setText(
            "Camera Started"
        )



    # =============================
    # Recognition
    # =============================

    def start_recognition(self):

        try:

            self.recognizer = FaceRecognizer()


            self.status_label.setText(
                "Recognition Active"
            )


        except Exception as e:


            QMessageBox.critical(

                self,

                "Recognizer Error",

                str(e)

            )



    # =============================
    # Build Database
    # =============================

    def build_database(self):

        try:


            self.status_label.setText(
                "Building Database..."
            )


            QApplication.processEvents()



            database = FaceDatabase()


            database.build_database()



            QMessageBox.information(

                self,

                "Completed",

                "Face database created"

            )


            self.status_label.setText(
                "Database Ready"
            )


        except Exception as e:


            QMessageBox.critical(

                self,

                "Database Error",

                str(e)

            )



    # =============================
    # Frame Update
    # =============================

    def update_frame(self):


        if not self.running:

            return



        frame = self.camera.read()


        if frame is None:

            return



        frame = cv2.flip(
            frame,
            1
        )



        if self.recognizer is not None:


            results = self.recognizer.recognize(
                frame
            )


            frame = self.recognizer.draw_results(
                frame,
                results
            )



        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )


        h, w, ch = rgb.shape


        bytes_per_line = ch * w



        image = QImage(

            rgb.data,

            w,

            h,

            bytes_per_line,

            QImage.Format.Format_RGB888

        )



        pixmap = QPixmap.fromImage(
            image
        )



        self.video_label.setPixmap(

            pixmap.scaled(

                self.video_label.width(),

                self.video_label.height(),

                Qt.AspectRatioMode.KeepAspectRatio

            )

        )



    # =============================
    # Stop
    # =============================

    def stop_camera(self):


        self.running = False


        self.timer.stop()


        self.camera.stop()


        self.video_label.clear()


        self.status_label.setText(
            "Stopped"
        )



    # =============================
    # Close
    # =============================

    def closeEvent(self, event):


        self.stop_camera()


        event.accept()

    def register_person(self):

        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(

            self,

            "Register Person",

            "Enter name:"

        )

        if ok and name:

            try:

                enroll = FaceEnrollment()

                enroll.capture_embeddings(

                    name,

                    30

                )

                QMessageBox.information(

                    self,

                    "Success",

                    f"{name} added successfully"

                )

                # reload database

                self.recognizer = FaceRecognizer()



            except Exception as e:

                QMessageBox.critical(

                    self,

                    "Error",

                    str(e)

                )



# =============================
# Run
# =============================

if __name__ == "__main__":


    app = QApplication(sys.argv)


    window = MainWindow()


    window.show()


    sys.exit(
        app.exec()
    )