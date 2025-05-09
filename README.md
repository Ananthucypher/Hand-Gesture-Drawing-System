# Hand Gesture Drawing System ✍️🤖

[![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-blue)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)](https://mediapipe.dev/)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](https://python.org)

A real-time hand gesture-controlled drawing application with **OCR text recognition** and **sketch classification** using computer vision and deep learning.

## Features 🎨
- ✋ **Air Drawing**: Draw on a virtual canvas using hand gestures (index finger tracking)
- 🔤 **Smart OCR**: Extract and autocorrect handwritten text with 85%+ accuracy
- 🖼️ **Image Recognition**: Classify simple sketches (apples, cars, etc.) using CNN
- ⚡ **Optimized Performance**: 30 FPS on low-end hardware with multithreading
- 🛠️ **User-Friendly UI**: Undo, clear canvas, grid overlay, and live camera preview

## Installation 🛠️
```bash
git clone https://github.com/Ananthucypher/Hand-Gesture-Drawing-System.git
cd Hand-Gesture-Drawing-System
pip install -r requirements.txt

Usage 🖐️
Run the application:
bash
python main.py

Drawing Mode:
Extend your index finger to start drawing
close your =palm to stop

OCR Mode:
Click "Recognize" to extract text from drawings

Image Recognition:
Click "Recognize Image" to classify sketchespython main.py

Component	          Technology Used
Hand                Tracking	MediaPipe
Drawing Engine	    OpenCV
OCR	                Tesseract
Sketch Recognition	TensorFlow/Keras
GUI	                Tkinter
