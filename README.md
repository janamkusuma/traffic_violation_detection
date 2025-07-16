Red Light Violation Detection System
Automatically detect vehicles that violate red traffic lights using YOLOv8 and Streamlit.

Project Overview
This project processes traffic videos to:

Detect the red light status in each frame.

Track vehicles crossing a predefined zone during a red signal.

Generate an annotated video with violation highlights.

Export a detailed CSV report for all detected violations.

How to Run Locally
Clone the Repository


git clone https://github.com/janamkusuma/traffic_violation_detection
cd traffic_violation_detection
Create Virtual Environment (optional but recommended)


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Requirements


pip install -r requirements.txt
Download YOLOv8 Model

Default: Uses yolov8n.pt for demonstration.

For better performance: Train or download a custom model for your country’s plates/lights and update the model path in load_model().

Example:


YOLO("path/to/custom-model.pt")
Run the App


streamlit run app.py

Features
Red light detection using HSV color filter.

Vehicle tracking with YOLOv8.

Annotated video output with violations marked.

Violation dashboard: total frames, violations, processing time.

Downloadable CSV report.

 
Original Video Upload

Detection Dashboard

Processed Video

CSV Report

Requirements
Python 3.8+

Install dependencies:


pip install -r requirements.txt


