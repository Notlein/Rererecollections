import cv2
import numpy as np
from pythonosc import udp_client

# Open a connection to the webcam (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Set up OSC client
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 12345)  # Replace with your OSC server address and port

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adjust the threshold value based on your lighting conditions
    threshold_value = 100
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the coordinates of the largest contour
    largest_contour = None
    largest_area = 0

    # Iterate through the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Set a threshold for the contour area to filter small contours
        if area > 50:
            # Update the largest contour if the current contour is larger
            if area > largest_area:
                largest_area = area
                largest_contour = contour

    # Draw a bounding box around the largest detected object
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Output the XY position of the largest object
        centroid_x = float(x + w / 2)
        centroid_y = float(y + h / 2)
        print(f"Object Position: X={centroid_x}, Y={centroid_y}")

        # Send the position via OSC
        osc_client.send_message("/detection", (centroid_x, centroid_y))

    # Display the result
    cv2.imshow('Object Detection', frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
