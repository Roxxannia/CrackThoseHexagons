import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

# Show Image function
def showImage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)

# Convert pixels to nm with the scale bar
# This could be automated in the future
def conversion():
    # Scale bar 200nm, and 166 pixels, therefore about 1.205 nm/pixel
    scale = 200/166
    return round(float(scale),3)

def preProcessing (image_path):
    #   TODO
    #   - Find the size of the scale bar in pixels
    #   - Process the image to edit the contrast

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or path is incorrect.")

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    showImage("Blurred Image", blurred)

    # Threshold image to b&w
    th3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,19,5)
    
    # Edge detection
    edges = cv2.Canny(th3, 25,45)
    showImage("Edge Image", edges)

    # thicken edge lines
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 1)
    showImage("Dilated Edges", dilation)

    return dilation

# Takes in the pre-processed b&w outlined image from pre-processing
# Returns a list of coordinates of every polygon vertex and center
def detect_hexagons(image_path, outlines, show_result=True):
    # load original image in greyscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Find contours
    contours, _ = cv2.findContours(outlines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale to BGR for visualization
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    #Detect hexagons
    hexagons = []
    centroids = []
    for contour in contours:
        # Approximate contour to polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check for hexagon: min 4 vertices, area threshold, and convexity
        if len(approx) >= 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 10 and area < 120: 
                hexagons.append(approx)
                cX, cY = findCentroids(contours = approx)
                centroids.append(np.array([cX, cY]))
    
    # Check if any hexagons overlap. If so, remove the second one
    # Threshold is the min distance in pixels that centroids can be from each other before being considered "overlapping"
    threshold = 3.5
    filtered_hexagons, filtered_centroids = remove_duplicate_hexagons(hexagons, centroids, threshold)

    for c in filtered_hexagons:
        cv2.drawContours(output, [c], -1, (0, 255, 0), 1)

    for h in filtered_centroids:
        cv2.circle(output, h, 0, (0, 255, 0), -1)

    if show_result:
        print(f"Detected {len(filtered_hexagons)} hexagons.")
        showImage("Detected Hexagons", output)
    
    # return hexagons, centroids, output
    return filtered_hexagons, filtered_centroids, output

def findCentroids(contours):
    # Moment is the weighted average of image pixel intensities
    M = cv2.moments(contours)
                
    # Preventing getting errors   
    if M["m00"] != 0:
        # Find the centroid of the image, convert it to binary format and then find its center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    return cX, cY

def remove_duplicate_hexagons(hexagons, centroids, threshold):
    n = len(centroids)
    # Create a matrix of boolean value that's the same size as centroids
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            # Calculate if there are centroids that are too close to each other, if there is then its possible that its duplicated
            if keep[j] and np.linalg.norm(centroids[i] - centroids[j]) < threshold:
                # Change corresponding boolean value in keep to False
                # Only keeping the first found value
                keep[j] = False  

    # Filter out the hexagons and centroids with the keep matrix
    filtered_hexagons = [hexagons[i] for i in range(n) if keep[i]]
    filtered_centroids = [centroids[i] for i in range(n) if keep[i]]

    return filtered_hexagons, filtered_centroids
