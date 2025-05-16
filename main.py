import cv2
import mediapipe as mp
import numpy as np 

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
 
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points
    Args:
        a: first point [x,y]
        b: mid point [x,y]
        c: end point [x,y]
    Returns:
        angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)
 
def calculate_vertical_angle(upper_point, lower_point):
    """
    Calculate angle from vertical line
    """
    vertical = np.array([0, -1])  # Vertical vector 
    actual = np.array([
        lower_point[0] - upper_point[0],
        lower_point[1] - upper_point[1]
    ])
    
    # Calculate angle using dot product
    dot = np.dot(vertical, actual)
    norms = np.linalg.norm(vertical) * np.linalg.norm(actual)
    
    angle = np.arccos(dot/norms)
    return np.degrees(angle)
 
# Initialize webcam
cap = cv2.VideoCapture(0)
 
# Set up the Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue
 
        # Convert the BGR image to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
 
        # Process the image and make detections
        results = holistic.process(image)
 
        # Convert the image color back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
        # Calculate angles if pose landmarks are detected
        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Calculate spine angle
            left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_hip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
            
            # Calculate midpoints
            mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2,
                          (left_shoulder[1] + right_shoulder[1])/2]
            mid_hip = [(left_hip[0] + right_hip[0])/2,
                      (left_hip[1] + right_hip[1])/2]
            
            # Calculate spine angle from vertical
            spine_angle = calculate_vertical_angle(mid_shoulder, mid_hip)
            
            # Calculate leg angles
            # Left leg
            left_knee = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE.value].y]
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            
            # Right leg
            right_knee = [landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE.value].y]
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Display angles on frame
            h, w, c = image.shape
            cv2.putText(image, f'Spine Angle: {int(spine_angle)}deg',
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(image, f'Left Leg Angle: {int(left_leg_angle)}deg',
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(image, f'Right Leg Angle: {int(right_leg_angle)}deg',
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
 
        # Draw landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
 
        # Display the image
        cv2.imshow('MediaPipe Holistic', image)
 
        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
 
# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()