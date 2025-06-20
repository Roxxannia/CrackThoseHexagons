import cv2
import numpy as np

def detect_hexagons(image_path, show_result=True):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or path is incorrect.")

    #equalized_image = cv2.equalizeHist(img)
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)

    #Threshold image to b&w
    # thresh = cv2.threshold(blurred, 100, 255,cv2.THRESH_BINARY)[1]
    th3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,19,5)

    #Edge detection
    edges = cv2.Canny(th3, 25,45)
    cv2.imshow("Edge Image", edges)
    cv2.waitKey(0)

    #thicken edge lines
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 1)
    cv2.imshow("Dilated Edges", dilation)
    cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    # Convert grayscale to BGR for visualization
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    hexagons = []
    for cnt in contours:
        # Approximate contour to polygon
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check for hexagon: 6 vertices, area threshold, and convexity
        if len(approx) >= 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 10 and area < 120: 
                intersects = False
                #check if the hexagon intersects with a pre-existing hexagons
                # Create a mask for the current hexagon
                mask = np.zeros(img.shape, dtype=np.uint8)
                cv2.drawContours(mask, [approx], -1, 255, -1)

                for prev_hex in hexagons:
                    prev_mask = np.zeros(img.shape, dtype=np.uint8)
                    cv2.drawContours(prev_mask, [prev_hex], -1, 255, -1)
                    # Check intersection
                    intersection = cv2.bitwise_and(mask, prev_mask)
                    if np.any(intersection):
                        intersects = True
                        break

                if not intersects:
                    hexagons.append(approx)
                    cv2.drawContours(output, [approx], 0, (0, 255, 0), 1)
    if show_result:
        print(f"Detected {len(hexagons)} hexagons.")
        print(hexagons[0], "\n", hexagons[-1])
        cv2.imshow("Detected Hexagons", output)
        cv2.waitKey(0)

    
        
    return hexagons

# Example usage:
if __name__ == "__main__":
    image_path = "hexagons_medium.png"  # Replace with your image path
    hexagons = detect_hexagons(image_path)
    


