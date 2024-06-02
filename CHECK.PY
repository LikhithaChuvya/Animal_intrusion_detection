import os
import dlib
import numpy as np
import cv2
import face_recognition




# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []
    
    for file_name in os.listdir(directory):
        if file_name.endswith(".jpeg"):
            name = os.path.splitext(file_name)[0]
            image_path = os.path.join(directory, file_name)
            image = dlib.load_rgb_image(image_path)
            
            # Detect faces in the image
            face_rects = detector(image, 1)
            
            if len(face_rects) == 1:
                # Predict the facial landmarks for the face region
                landmarks = predictor(image, face_rects[0])
                
                # Compute the face descriptor using the facial landmarks
                face_encoding = np.array(face_recognizer.compute_face_descriptor(image, landmarks))
                
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
    
    return known_face_encodings, known_face_names

import face_recognition

def recognize_face(frame, known_face_encodings, known_face_names):
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces in the frame
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face is a match for any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # If there is a match, label the face with the name
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Unknown"

        # Draw a box around the face and label it with the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

# Load face detector and face recognition models
detector = dlib.get_frontal_face_detector()
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load known faces and their encodings
known_face_encodings, known_face_names = load_known_faces("/home/vamsi/Desktop/iot/pics")

# Start video capture
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    name = recognize_face(frame, known_face_encodings, known_face_names)
    if name is not None:
        cv2.putText(frame, str(name), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()

