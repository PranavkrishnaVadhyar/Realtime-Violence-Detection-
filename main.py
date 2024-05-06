import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import tensorflow as tf
import datetime
from message import send_sms
import os


# Define constants
SEQUENCE_LENGTH = 16  # Number of frames to consider for prediction
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
CLASSES_LIST = ["Non-Violence", "Violence"]  # Assuming binary classification

snapshot_folder = "violence_snapshots"
if not os.path.exists(snapshot_folder):
    os.makedirs(snapshot_folder)

# Load the TensorFlow model from the .pb file
model_path = "C:/Users/bpran/Desktop/ViolenceDetection/MoBiLSTM_model.h5"  # Update with your actual model path
model = load_model(model_path)

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for the default camera (usually front camera)

# Declare a queue to store video frames.
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the Frame to fixed Dimensions.
    resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Normalize the resized frame
    normalized_frame = resized_frame / 255.0  # Assuming pixel values are in [0, 255]

    # Appending the pre-processed frame into the frames list.
    frames_queue.append(normalized_frame)

    # Perform prediction when enough frames are in the queue
    if len(frames_queue) == SEQUENCE_LENGTH:
        # Convert frames_queue to a numpy array and expand dimensions
        input_data = np.expand_dims(np.array(frames_queue), axis=0)

        # Convert input_data to TensorFlow Tensor
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Perform inference using the TensorFlow model
        predictions = model(input_tensor)

        # Get the predicted class index
        predicted_label = np.argmax(predictions[0])

        # Get the class name using the retrieved index.
        predicted_class_name = CLASSES_LIST[predicted_label]

        # Write predicted class name on top of the frame.
        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Send email with timestamp and image
            image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

            # Save the snapshot of violence with timestamp as its name
            snapshot_filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(os.path.join(snapshot_folder, snapshot_filename), frame)
            # Send SMS with timestamp and image path
            send_sms(timestamp)

        else:
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)



    # Display the processed frame
    cv2.imshow('Real-time Violence Detection', frame)

    # Check for the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera, video writer, and close all windows
cap.release()
cv2.destroyAllWindows()
