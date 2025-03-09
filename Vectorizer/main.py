import cv2
import os
import numpy as np
from PIL import Image
import fitz  

image_path = r"C:\Users\Kakha\python3\Vectorizer\Shapes.jpg"

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
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2) 

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 3:
                cv2.drawContours(img, [approx], 0, (0, 255, 0), -1) 
            elif len(approx) == 4:
                cv2.drawContours(img, [approx], 0, (255, 0, 0), -1) 

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(img, (x, y), r, (255, 255, 0), 4)  

        cv2.imwrite("shapes_detected.jpg", img)

        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pdf = fitz.open()
        pdf_bytes = pil_image.convert("RGB").tobytes("jpeg", "RGB")
        rect = fitz.Rect(0, 0, pil_image.width, pil_image.height)
        page = pdf.new_page(width=pil_image.width, height=pil_image.height)
        page.insert_image(rect, stream=pdf_bytes)
        pdf.save("shapes_detected.pdf")
        pdf.close()
