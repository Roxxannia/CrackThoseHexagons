import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

# Show Image function
def showImage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)

# Convert pixels to nm with the scale bar. This could be automated in the future
def conversion():
    # Scale bar 200nm, and 167 pixels, therefore about 1.205 nm/pixel
    scale = 200/167
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

# Displays an image with the detected polygons and centroids
def drawHexagons(image, polygons, centroids):
    for c in polygons:
        cv2.drawContours(image, [c], -1, (0, 255, 0), 1)

    for h in centroids:
        cv2.circle(image, h, 0, (0, 255, 0), -1)

    print(f"Detected {len(polygons)} hexagons.")
    showImage("Detected Hexagons", image)

    return

# Takes in a list of the vertices of each polygon, returns a list of center coordinates of each polygon
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

# Removes the hexagons that overlap
def removeDuplicateHexagons(hexagons, centroids, threshold):
    # Create a matrix of boolean value that's the same size as centroids
    n = len(centroids)
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
    filteredHexagons = [hexagons[i] for i in range(n) if keep[i]]
    filteredCentroids = [centroids[i] for i in range(n) if keep[i]]

    return filteredHexagons, filteredCentroids

# Takes in the pre-processed b&w outlined image from pre-processing // Returns a list of coordinates of every polygon vertex and center
def detectHexagons(image_path, outlines):
    # load original image in greyscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Find contours
    contours, _ = cv2.findContours(outlines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale to RGB for visualization
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
    filteredHexagons, filteredCentroids = removeDuplicateHexagons(hexagons, centroids, threshold)

    # Draw the hexagons on the original image, and display the number of hexagons found
    for c in filteredHexagons:
        cv2.drawContours(output, [c], -1, (0, 255, 0), 1)
    for h in filteredCentroids:
        cv2.circle(output, h, 0, (0, 255, 0), -1)
    print(f"Detected {len(filteredHexagons)} hexagons.")
    showImage("Detected Hexagons", output)

    return filteredHexagons, filteredCentroids, output

# Remove the duplicated lines from the nearest neighbour list
def filterNearestNeighbours(temp):
    x = 0
    keep = [True] * len(temp)
    for i in range(len(temp)):
        for h in range(i, len(temp)):
            if i != h \
            and (temp[i][0][0] == temp[h][1][0]) \
            and (temp[i][0][1] == temp[h][1][1]) \
            and (temp[i][1][0] == temp[h][0][0]) \
            and (temp[i][1][1] == temp[h][0][1]):
                keep[i] =False

    filtered_data = [temp[i] for i in range(len(temp)) if keep[i]]


    for i in range(len(filtered_data)):
        cv2.line(output, filtered_data[i][0], filtered_data[i][1], (255, 100, 0), 1)

    showImage("New Nearest Neighbors", output)
    strainCalc(filtered_data)

    print(len(temp))
    print(len(filtered_data))
    return

def nearestNeighbours(centroid, k, image):
    neighbours = {}
    neighbours_temp = {}
    x = 0
    for index, c in enumerate(centroid):
        # Calculate euclidean distances between each centroid and all other centroids
        EuDistance = [np.linalg.norm(np.array(x) - np.array(c)) for x in centroid]
        # Exclude the centroid itself from the nearest distance
        EuDistance[index] = float('inf')  # exclude self

        # Sort data points by distance (smallest to largest) and get first K numbers of nearest neighbors
        # N_distance is just a group of indices of the closest points
        N_distance = np.argsort(EuDistance, kind='stable')[:k]  

        kNN_temp = [EuDistance[n] for n in N_distance]
        avg = np.mean(kNN_temp)
        # for dist in N_distance:
        #     sum += EuDistance[dist]
        
        # avg = sum/6

        standardDev = np.std(kNN_temp)  
        # print(standardDev)
        # Get the target values of the K nearest neighbors as long as they are under certain distance (pixels)
        nearest = []
        
        for dist in N_distance:
            # EuDistance[dist] < np.sqrt(15**2+15**2) and
            if EuDistance[dist] <= avg or standardDev <= 2:
                nearest.append(dist)
                neighbours_temp[x] = ([centroid[index],centroid[dist]])
                x += 1

 
        kNN = [centroid[i] for i in nearest]

        neighbours[index] = kNN

    for i in range(len(centroid)):
        for n in neighbours[i]:
            cv2.line(image, centroid[i], n, (255, 0, 0), 1)

    showImage("Nearest Neighbors", image)
    # If you want to save the image, uncomment this line
    # cv2.imwrite('Nearest_neigh.png',image)

    return neighbours, EuDistance, neighbours_temp

def strainCalc(centroids):
    sizes = []
    for datapoint in centroids:
        sizes.append(np.linalg.norm(np.array(datapoint[0]) - np.array(datapoint[1])))
        print(datapoint)
    print("number of lines: ", len(sizes))
    averageSize = sum(sizes)/len(sizes)
    print(averageSize)
    return averageSize

if __name__ == "__main__":

    # start_time = time.time()

    # image_path = "C:/Users/roxxa/OneDrive/University/Masters/Code/CrackThoseHexagons/hexagons_lightRoom.jpg"  
    image_path = "C:/Users/Owner/OneDrive/Documents/School/Masters/Research/Code/hexagons_git/CrackThoseHexagons/vat3-processed.jpg"
    outline = preProcessing(image_path)
    hexagons, centroids, output = detectHexagons(image_path, outline)

    # scale = conversion()

    neighbours, distance, temp = nearestNeighbours(centroids, 6, output) 
    filterNearestNeighbours(temp)
    
    # print("filtered data: \n", filtered_data)