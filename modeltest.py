import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model

# Load the trained emotion recognition model
model = load_model('emotion_model.h5')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the image for the model
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = np.reshape(img, (1, 48, 48, 1))
    return img

# Function to predict the emotion from the image
def predict_emotion(img):
    img = preprocess_image(img)
    emotion_prediction = model.predict(img)
    emotion_label = emotions[np.argmax(emotion_prediction)]
    return emotion_label

# Open the webcam
cap = cv2.VideoCapture(0)

# Create a folder to store images if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Set the time interval for capturing images (in seconds)
capture_interval = 5

# Initialize variables for capturing images
last_capture_time = time.time()
capture_count = 0

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces and predict emotions
    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        emotion = predict_emotion(face_roi)

        # Display the emotion on the frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Emotion Recognition', frame)

    # Capture and store images at regular intervals
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        last_capture_time = current_time

        # Save the current frame as an image
        capture_count += 1
        image_filename = f'data/image_{capture_count}.png'
        cv2.imwrite(image_filename, frame)
        print(f"Image captured and saved: {image_filename}")

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
