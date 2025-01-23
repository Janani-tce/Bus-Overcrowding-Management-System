# Bus-Overcrowding-Management-System
This project aims to control crowding in buses using video analytics and image recognition. It detects and counts the number of people in a video feed. If the number of people exceeds a certain limit, the system triggers an alarm and sends notifications to bus management for allocating additional buses.

Features

Detect and count the number of people in a video feed.

Trigger an alarm when the number of people exceeds a set limit.

Notify bus management for better resource allocation.

Frontend interface to display the count and control the system.

Technology Stack

Backend: Python (OpenCV, NumPy, Pygame)

Frontend: HTML, CSS, JavaScript

Machine Learning: YOLO (You Only Look Once) object detection model

Installation and Setup

Prerequisites

Python 3.x installed on your system.

Libraries: Install required Python libraries using the command:

pip install opencv-python numpy pygame

Git installed on your system.

YOLO model files:

yolov3.weights

yolov3.cfg

coco.names

Steps to Run

Clone the repository:

git clone https://github.com/Janani-tce/Bus-Overcrowding-Management-System.git

Navigate to the project directory:

cd Bus-Overcrowding-Management-System

Ensure that the YOLO model files (yolov3.weights, yolov3.cfg, coco.names) are in the project directory.

Run the Python script:

python app.py

Open the index.html file in your browser to access the frontend.

File Structure

app.py: Main backend script for video processing.

index.html: Frontend interface for user interaction.

style.css: Stylesheet for the frontend.

yolov3.weights: Pre-trained YOLO weights.

yolov3.cfg: YOLO configuration file.

coco.names: Class names for YOLO (e.g., person, car, etc.).

alarm_sound.wav: Alarm sound file triggered when the limit is exceeded.

Usage

Update the app.py script to set the limit for the number of people.

Ensure the video file (videoplayback.mp4) is in the project directory, or replace it with your video feed.

Monitor the live video feed through the frontend interface.

Notes

The file yolov3.weights exceeds GitHub’s file size limit. Download it from Google Drive or another source and place it in the project directory.

For live video feeds, modify the video source in app.py to use a webcam:

cap = cv2.VideoCapture(0)

Future Enhancements

Add real-time notifications through email or SMS.

Improve the frontend UI for better visualization.

Integrate with cloud services for scalability.

License

This project is open-source and available under the MIT License.

