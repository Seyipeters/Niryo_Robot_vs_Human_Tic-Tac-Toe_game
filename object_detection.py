import cv2
import numpy as np

def detect_tokens(frame, debug=False):
    """
    Detect RED and BLUE tokens based on HSV color segmentation.
    Returns a list of tuples: [(color, x, y, w, h), ...]
    :param frame: Input image frame (BGR format).
    :param debug: If True, displays intermediate masks and bounding boxes for debugging.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    tokens = []

    # --- RED TOKEN DETECTION ---
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = clean_mask(mask_red)
    tokens += get_token_bounding_boxes(mask_red, 'red', debug, frame)

    # --- BLUE TOKEN DETECTION ---
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue = clean_mask(mask_blue)
    tokens += get_token_bounding_boxes(mask_blue, 'blue', debug, frame)

    # Sort tokens by position (top-to-bottom, left-to-right)
    tokens = sorted(tokens, key=lambda t: (t[2], t[1]))  # Sort by y, then x

    return tokens

def clean_mask(mask):
    """
    Apply morphological operations to clean up the mask.
    :param mask: Binary mask.
    :return: Cleaned binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
    return mask

def get_token_bounding_boxes(mask, color_label, debug=False, frame=None):
    """
    Find bounding boxes for detected tokens in the mask.
    :param mask: Binary mask for the color.
    :param color_label: Label for the color ('red' or 'blue').
    :param debug: If True, displays bounding boxes on the frame.
    :param frame: Original frame for visualization (optional).
    :return: List of bounding boxes [(color, x, y, w, h), ...].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter out small blobs
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.8 <= aspect_ratio <= 1.2:  # Ensure the contour is roughly square
                boxes.append((color_label, x, y, w, h))
                if debug and frame is not None:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, color_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if debug and frame is not None:
        cv2.imshow(f"{color_label} Mask", mask)
        cv2.imshow(f"{color_label} Detection", frame)
        cv2.waitKey(1)

    return boxes