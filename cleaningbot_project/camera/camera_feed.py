import cv2

def capture_frame():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None
    
    ret, frame = cap.read()
    
    if ret:
        return frame
    else:
        print("Error: Failed to capture frame.")
        return None
    
    cap.release()  # Ensure the camera is released outside the if block
