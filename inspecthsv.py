import cv2
import numpy as np
from pyniryo import NiryoRobot

def inspect_hsv(robot):
    """
    Capture an image from the robot's camera, display it, and save it for HSV calibration.
    """
    img = robot.get_img_compressed()
    if img is None:
        print("No image captured from the robot.")
        return

    # Decode the image
    frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("Failed to decode the image.")
        return

    # Convert the image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the original and HSV images
    cv2.imshow("Original Frame", frame)
    cv2.imshow("HSV Frame", hsv)

    # Save the images for analysis
    cv2.imwrite("board_image.jpg", frame)
    cv2.imwrite("board_image_hsv.jpg", hsv)
    print("Images saved as 'board_image.jpg' and 'board_image_hsv.jpg'.")

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your robot's IP address
    robot_ip = "10.10.10.10"
    robot = NiryoRobot(robot_ip)

    try:
        print("Capturing image from the robot's camera...")
        inspect_hsv(robot)
    finally:
        robot.close_connection()
        print("Robot connection closed.")