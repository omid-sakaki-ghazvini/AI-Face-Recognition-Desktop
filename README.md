# 🎯 AI Face Recognition Desktop

> A real-time face detection and recognition desktop application built with **Python**, **OpenCV** (YuNet + SFace), and **PyQt6**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv&logoColor=white)
![PyQt6](https://img.shields.io/badge/PyQt6-6.5%2B-brightgreen?logo=qt&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

---

## 📖 Overview

**AI Face Recognition Desktop** is a complete desktop solution for real-time face detection and identification. It uses OpenCV's state-of-the-art models — **YuNet** for face detection and **SFace** for face recognition — combined with a modern **PyQt6** graphical interface.

The application allows you to:
- Detect faces live from your webcam
- Enroll new people into a local database
- Identify known individuals in real time using cosine similarity

---

## ✨ Features

- 🎥 **Real-time face detection** using YuNet (ONNX model)
- 🧠 **Face recognition** using SFace embeddings
- 👤 **Person enrollment** directly from the webcam
- 🗂️ **Local embedding database** stored as a `.pkl` file
- 📊 **Live FPS counter** and face count overlay
- 🖥️ **Modern PyQt6 GUI** with intuitive controls
- 🔒 **Cosine similarity threshold** to reject unknown faces
- 🧵 **Threaded camera capture** for smooth, non-blocking performance
- 📁 **Dataset management** — add, delete, and inspect registered people

---

## 📸 Screenshots

> Add your own screenshots here.

| Main Window | Live Recognition |
|-------------|------------------|
| ![Main UI](screenshots/ui_preview.png) | ![Recognition](screenshots/recognition.png) |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-------------|
| Language | Python 3.9+ |
| GUI | PyQt6 |
| Computer Vision | OpenCV (`opencv-contrib-python`) |
| Face Detection | YuNet (ONNX) |
| Face Recognition | SFace (ONNX) |
| Numerical Computing | NumPy |
| Serialization | Pickle |

---
