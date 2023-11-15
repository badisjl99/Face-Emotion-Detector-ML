import cv2
import time
import os
import socket

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Create a folder called 'data' if it doesn't exist
data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)

# Get local IP address
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Camera', frame)

    # Capture an image every 3 seconds
    time.sleep(3)

    # Save the captured image with local IP address in the 'data' folder
    img_name = os.path.join(data_folder, f"captured_image_({ip_address})_({time.strftime('%Y-%m-%d-%H-%M-%S')}).png")
    cv2.imwrite(img_name, frame)
    print(f"Image captured and saved: {img_name}")

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
