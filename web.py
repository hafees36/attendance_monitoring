from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

KNOWN_FACES_DIR = 'known_faces'
attendance_file = 'webattendance.csv'

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(('.jpg', '.png')):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Initialize attendance file if not exists
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name', 'Date', 'Time' ,'Status'])
    df.to_csv(attendance_file, index=False)

# Start webcam
video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.45)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                now = datetime.now()
                date = now.strftime('%Y-%m-%d')
                time = now.strftime('%H:%M:%S')
                
                df = pd.read_csv(attendance_file)
                if not ((df['Name'] == name) & (df['Date'] == date)).any():
                    new_entry = {'Name': name, 'Date': date, 'Time': time,'Status':'Present'}
                    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                    df.to_csv(attendance_file, index=False)

                # Draw rectangle and name
                top, right, bottom, left = [v*4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    df = pd.read_csv(attendance_file)
    attendance_records = df.to_dict(orient='records')
    return render_template('index.html', attendance=attendance_records)

@app.route('/attendance_data')
def attendance_data():
    df = pd.read_csv(attendance_file)
    return jsonify(df.to_dict(orient='records'))

if __name__ == "__main__":
    app.run(debug=True)
