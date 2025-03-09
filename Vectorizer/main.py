import cv2
import os

image_path = r"D:/Project/Image/Shapes.jpg"

if not os.path.isfile(image_path):
    print(f"Image not found at the specified path: {image_path}")
else:
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Failed to load image: {image_path}")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale Image", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
