# tictactoe_vision.py

import cv2
import numpy as np
from pyniryo import NiryoRobot
from object_detection import detect_tokens

def uncompress_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

class BoardDetector:
    def __init__(self, robot, shrink_factor=0.7):
        """
        :param robot: Instance of NiryoRobot.
        :param shrink_factor: Fraction to shrink the detected board grid relative to the board bounding box.
        """
        self.robot = robot
        self.grid_cells = None
        self.shrink_factor = shrink_factor

    def get_grid_cells(self):
        return self.grid_cells

    def detect_board(self):
        """
        Use the robot's camera to detect the board. Press 'q' when the grid is properly visualized.
        Returns a list of 9 cells as tuples: [((x1, y1), (x2, y2)), ...].
        """
        while True:
            img_compressed = self.robot.get_img_compressed()
            if img_compressed is None:
                print("No image from the robot camera. Retrying...")
                continue

            frame = uncompress_image(img_compressed)
            if frame is None:
                print("Failed to decode camera image. Retrying...")
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Adjust thresholds as needed for your board color
            lower = np.array([0, 0, 100])
            upper = np.array([180, 50, 255])
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                cv2.imshow("Board Detection", frame)
                print("No board-like contour found. Check thresholds.")
            else:
                board_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(board_contour)
                board_size = min(w, h)
                grid_size = int(board_size * self.shrink_factor)
                # Center the grid in the detected board region
                x = x + (w - grid_size) // 2
                y = y + (h - grid_size) // 2
                w, h = grid_size, grid_size
                cell_size = w // 3
                self.grid_cells = []
                from itertools import product
                for row, col in product(range(3), repeat=2):
                        x1 = x + col * cell_size
                        y1 = y + row * cell_size
                        x2 = x1 + cell_size
                        y2 = y1 + cell_size
                        self.grid_cells.append(((x1, y1), (x2, y2)))
                # Draw the grid on the frame for visual feedback
                for (top_left, bottom_right) in self.grid_cells:
                    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.imshow("Board Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return self.grid_cells

    def detect_tokens_on_board(self, frame):
        return detect_tokens(frame)