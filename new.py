import cv2
import numpy as np
import mediapipe as mp
import datetime
import time
from cvzone.FaceMeshModule import FaceMeshDetector

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Function to measure size in centimeters
def measure_size(frame, points, pixel_per_cm):
    # Calculate the distance between points in pixels
    distance_pixels = calculate_distance(points[0], points[1])
    # Convert distance from pixels to centimeters
    distance_cm = distance_pixels / pixel_per_cm
    return distance_cm

# Initialize Pose estimator
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Open the default camera (usually the first connected camera)
cap = cv2.VideoCapture(0)

# Initialize Face Mesh Detector
detector = FaceMeshDetector(maxFaces=1)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize variables for countdown
countdown_start_time = time.time()
countdown_duration = 20

# Loop to continuously capture frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    try:
        # Convert the frame to RGB format
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB frame to get the result
        results = pose.process(RGB)

        # Find Face Mesh
        frame, faces = detector.findFaceMesh(frame, draw=False)

        # Check if face mesh is detected
        if faces:
            print("Face mesh detected")
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]
            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3

            # Finding the Focal Length
            d = 50
            f = (w * d) / W

            # Finding distance
            f = 840
            d = (W * f) / w

            # Check if the depth distance is out of the specified range (290 to 310 cm)
            if d < 295:
                cv2.putText(frame, "Go Back", (270, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif d > 310:
                cv2.putText(frame, "Move Forward", (270, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if results.pose_landmarks:
                    print("Pose landmarks detected")
                    # Extract landmark points for shoulder, hand, and waist for both sides
                    for landmark in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                     mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
                                     mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]:
                        landmark_point = np.array([results.pose_landmarks.landmark[landmark].x * frame.shape[1],
                                                   results.pose_landmarks.landmark[landmark].y * frame.shape[0]])

                        # Draw the landmark points
                        if landmark in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]:
                            color = (255, 0, 0)  # Blue for shoulders
                        elif landmark in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
                            color = (0, 0, 255)  # Red for wrists
                        else:
                            color = (0, 255, 0)  # Green for hips
                        
                        cv2.circle(frame, (int(landmark_point[0]), int(landmark_point[1])), 5, color, -1)

                    # Calculate shoulder, hand, and waist sizes in centimeters
                    shoulder_size_cm = measure_size(frame,
                                                     [np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1],
                                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]]),
                                                      np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
                                                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]])],
                                                     100 / 46)  # Assuming known shoulder size of 100 pixels is 46 cm

                    hand_size_cm = measure_size(frame,
                                                [np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
                                                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]]),
                                                 np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1],
                                                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]])],
                                                100 / 56)  # Assuming known hand size of 100 pixels is 56 cm

                    waist_size_cm = measure_size(frame,
                                                 [np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * frame.shape[1],
                                                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * frame.shape[0]]),
                                                  np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * frame.shape[1],
                                                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * frame.shape[0]])],
                                                 100 / 80)  # Assuming known waist size of 100 pixels is 80 cm

                    # Print calculated sizes for debugging
                    print(f"Shoulder Size: {shoulder_size_cm}, Hand Size: {hand_size_cm}, Waist Size: {waist_size_cm}")

                    # Determine shirt size based on shoulder size
                    if 36 <= shoulder_size_cm <= 38:
                        shirt_size = "S"
                    elif 39 <= shoulder_size_cm <= 41:
                        shirt_size = "M"
                    elif 42 <= shoulder_size_cm <= 45:
                        shirt_size = "L"
                    elif 46 <= shoulder_size_cm <= 49:
                        shirt_size = "XL"
                    elif 50 <= shoulder_size_cm <= 53:
                        shirt_size = "XXL"
                    else:
                        shirt_size = "Unknown"

                    # Draw measurements on the frame with a background for visibility
                    cv2.rectangle(frame, (10, 70), (300, 220), (255, 255, 255), -1)  # White rectangle for background
                    font_scale = 0.7  # Adjust font size here
                    cv2.putText(frame, f"Shoulder Size: {int(shoulder_size_cm)} cm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (0, 0, 0), 2)
                    cv2.putText(frame, f"Hand Size: {int(hand_size_cm)} cm", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (0, 0, 0), 2)
                    cv2.putText(frame, f"Waist Size: {int(waist_size_cm)} cm", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (0, 0, 0), 2)
                    cv2.putText(frame, f"Shirt Size: {shirt_size}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (0, 0, 0), 2)

                    # Capture the frame and save it as an image
                    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    filename = f"captured_photo_{current_time}.jpg"
                    cv2.imwrite(filename, frame)
                    print("Photo captured and saved as:", filename)

                    # Release the camera and close all OpenCV windows
                    cap.release()
                    cv2.destroyAllWindows()
                    break  # Exit the loop after capturing the photo

        else:
            print("No face mesh detected")

    except Exception as e:
        print("An error occurred:", e)

    # Display the captured frame
    cv2.imshow('Measurement', frame)

    # Check for key events
    key = cv2.waitKey(1)

    # Check for the 'q' key to exit the loop
    if key == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
