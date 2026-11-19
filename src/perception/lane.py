
import cv2
import numpy as np


def detect_lanes(frame):
    """
    Detect road lane markings using OpenCV.
    Returns the original frame with detected lane lines.
    """

    if frame is None or frame.size == 0:
        return frame

    height, width = frame.shape[:2]

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Focus on the road region
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.45), int(height * 0.55)),
        (int(width * 0.55), int(height * 0.55)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lane lines
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    # Draw detected lines
    if lines is not None:

        for line in lines:

            # Handle OpenCV output shapes:
            # (1, 4) or (4,)
            coordinates = np.asarray(line).reshape(-1)

            if coordinates.size != 4:
                continue

            x1, y1, x2, y2 = map(int, coordinates)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                4
            )

    return frame
