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

        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
