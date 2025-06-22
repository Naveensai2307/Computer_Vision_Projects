
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR messages

import cv2
from cvzone.HandTrackingModule import HandDetector

# Set up webcam capture (use 0 or 1 depending on your camera)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8)

# Load image once outside the loop
img1_original = cv2.imread("pic.jpg")
if img1_original is None:
    raise FileNotFoundError("Image 'pic.jpg' not found!")

# Variables for zoom gesture
startDist = None
scale = 0
cx, cy = 900, 500  # Initial center coordinates for image placement

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to read from webcam.")
        continue

    # Detect hands
    hands, img = detector.findHands(img)
    img1 = img1_original.copy()

    if len(hands) == 2:
        # Check for zoom gesture: both hands show [1, 1, 0, 0, 0]
        if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
           detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:

            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]

            # Use index fingertips (landmark 8) - only (x, y)
            point1 = lmList1[8][:2]
            point2 = lmList2[8][:2]

            # Get distance between fingers
            length, info, img = detector.findDistance(point1, point2, img)

            # Set startDist only once
            if startDist is None and length > 0:
                startDist = length

            # Compute scale based on distance change
            if startDist:
                scale = int((length - startDist) // 2)
                cx, cy = info[4:]  # midpoint (x, y)

    else:
        startDist = None  # Reset if not using two fingers

    try:
        h1, w1, _ = img1.shape

        # Prevent invalid (negative or zero) resize dimensions
        newH = max(2, ((h1 + scale) // 2) * 2)
        newW = max(2, ((w1 + scale) // 2) * 2)
        img1 = cv2.resize(img1, (newW, newH))

        # Calculate placement position
        top, bottom = cy - newH // 2, cy + newH // 2
        left, right = cx - newW // 2, cx + newW // 2

        # Only paste if within frame bounds
        if 0 <= top and bottom <= img.shape[0] and 0 <= left and right <= img.shape[1]:
            img[top:bottom, left:right] = img1

    except Exception as e:
        print("Image placement error:", e)

    # Display output
    cv2.imshow("Zoom Gesture Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
