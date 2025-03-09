import cv2
import os
import numpy as np

image_path = r"D:/Project/Image/Shapes.jpg"

if not os.path.isfile(image_path):
    print(f"Image not found at the specified path: {image_path}")
else:
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 3:
                cv2.drawContours(img, [approx], 0, (0, 255, 0), -1)  # Green for triangles
            elif len(approx) == 4:
                cv2.drawContours(img, [approx], 0, (255, 0, 0), -1)  # Blue for rectangles/squares

        cv2.imshow("Detected Shapes and Lines", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
