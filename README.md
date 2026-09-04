# Face Recognition Attendance System

A real-time face recognition based attendance system developed using Python and OpenCV. The system captures student faces, trains a face recognition model, recognizes registered students through a webcam, and automatically records their attendance.

## Features

- Student registration through webcam
- Face detection using Haar Cascade
- Face recognition using LBPH
- Real-time webcam-based recognition
- 3-second confirmation before marking attendance
- Prevents duplicate attendance on the same day
- Attendance stored in CSV format
- Excel report generation
- Simple Tkinter graphical user interface

## Technologies Used

- Python
- OpenCV
- LBPH Face Recognizer
- Haar Cascade Classifier
- Pandas
- NumPy
- OpenPyXL
- Tkinter
- CSV
- Pickle

## Project Workflow

```text
Register Student
       ↓
Capture Face Images
       ↓
Train Recognition Model
       ↓
Real-Time Face Recognition
       ↓
3-Second Confirmation
       ↓
Attendance Recorded
       ↓
CSV Attendance File
       ↓
Excel Report