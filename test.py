import cv2
import numpy as np
import geocoder
import random
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

# Create a list of classes
classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

# Initialise the Firebase Admin SDK
cred = credentials.Certificate("/Users/shobithmallya/darknet/marine-debris-observatio-d453a-firebase-adminsdk-f37vf-5722ad755e.json")
firebase_admin.initialize_app(cred, {'databaseURL':'https://marine-debris-observatio-d453a-default-rtdb.asia-southeast1.firebasedatabase.app'})

# Create a reference to the database
ref = db.reference('objects')

# Create a VideoCapture object and connect it to the webcam
cap = cv2.VideoCapture("http://192.168.43.73:81/stream")

# Check if the connection was successful
if not cap.isOpened():
    raise RuntimeError("Could not open the webcam")

# Create a file to store the detected objects
output_file = open("detected_objects.txt", "w")

# Read frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame to a fixed width for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()

    # Process the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            class_label = classes[class_id]

            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([640, 480, 640, 480])
            (x, y, w, h) = box.astype("int")

            # Draw the bounding box and class label on the frame
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            label = f"{class_label}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Retrieve latitude and longitude
            location = geocoder.ip('me').latlng
            latitude, longitude = location[0], location[1]

            # Write the detected object and location to the file
            if class_label == "bottle":
                add_latitude = random.randrange(1000,4000)
                output_file.write(f"{class_label}, Latitude: {round(latitude)}.{add_latitude}, Longitude: {longitude}\n")

                #Write the detected object and location to the database
                data = {
                    'type': class_label,
                    'latitude': latitude,
                    'longitude': longitude
                }
                ref.push().set(data)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord("q"):
        break

# Release the VideoCapture object, close the file, and close windows
cap.release()
output_file.close()
cv2.destroyAllWindows()