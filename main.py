import cv2
import numpy as np

def showImage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)

# Convert pixels to nm with the scale bar. This could be automated in the future
def conversion():
    # Scale bar 200nm, and 167 pixels, therefore about 1.205 nm/pixel
    scale = 200/167
    return round(float(scale),3)

def findThresholds(hexSize_nm):
    hexSize_pixels = hexSize_nm / conversion()
    avgArea = (hexSize_pixels/2)**2 * np.pi
    maxArea = int(avgArea) + 10
    minArea = int(0.1 * avgArea)
    return minArea, maxArea

def preProcessing (image_path, blurInt):
    #   TODO
    #   - Find the size of the scale bar in pixels
    #   - Process the image to edit the contrast

    # 07/13 Roxxannia edit
    # For now, the image needs to be processed (i.e. adjusted in lightroom) before loading in
    # The image does not need to be cropped

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or path is incorrect.")
    
    # Resizing image
    # Define new width and height
    new_width = 710
    new_height = 510
    
    # Resize the image
    img = cv2.resize(img, (new_width, new_height))

    # Cropping image
    cropped_height = 474
    cropped_image = img[0:cropped_height, 0:new_width]

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(cropped_image, (blurInt,blurInt), 0)
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

    return dilation, blurred

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

# Removes hexagons that overlap with another
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
def detectHexagons(image_path, blurredImage, outlines, minArea, maxArea, distanceThreshold):
    # load original image in greyscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Find contours
    contours, _ = cv2.findContours(outlines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale to RGB for visualization
    blurredImage = cv2.cvtColor(blurredImage, cv2.COLOR_GRAY2BGR)
    
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
            if area > minArea and area < maxArea: 
                hexagons.append(approx)
                cX, cY = findCentroids(contours = approx)
                centroids.append(np.array([cX, cY]))
    
    # Check if any hexagons overlap. If so, remove the second one
    # Threshold is the min distance in pixels that centroids can be from each other before being considered "overlapping"
    filteredHexagons, filteredCentroids = removeDuplicateHexagons(hexagons, centroids, distanceThreshold)

    # Draw the hexagons on the original image, and display the number of hexagons found
    for c in filteredHexagons:
        cv2.drawContours(blurredImage, [c], -1, (0, 255, 0), 1)
    for h in filteredCentroids:
        cv2.circle(blurredImage, h, 1, (0, 0, 255), -1)
    print("Number of hexagons: ", len(filteredHexagons))
    showImage("Detected Hexagons", blurredImage)

    return filteredHexagons, filteredCentroids, blurredImage

# Remove the duplicated lines from the nearest neighbour list
def removeDuplicateNeighbours(temp):
    keep = [True] * len(temp)
    for i in range(len(temp)):
        for h in range(i, len(temp)):
            if i != h \
            and (temp[i][0][0] == temp[h][1][0]) \
            and (temp[i][0][1] == temp[h][1][1]) \
            and (temp[i][1][0] == temp[h][0][0]) \
            and (temp[i][1][1] == temp[h][0][1]):
                keep[i] = False

    filteredNeighbours = [temp[i] for i in range(len(temp)) if keep[i]]

    for i in range(len(filteredNeighbours)):
        cv2.line(output, filteredNeighbours[i][0], filteredNeighbours[i][1], (255, 100, 0), 1)
    showImage("New Nearest Neighbors", output)
    print("Number of lines: ", len(filteredNeighbours))
    return filteredNeighbours

#This threshold value might be something we need to change depending on the sample/hexagon size
def nearestNeighbours(centroid, threshold = 2):
    #dictionary of startpoint (centroid) and endpoint (vertex) of each neighbour
    startPointEndPoint_list = {} 
    x = 0
    for index, c in enumerate(centroid):
        # Calculate euclidean distances between each centroid and all other centroids
        EuDistance = [np.linalg.norm(np.array(x) - np.array(c)) for x in centroid]
        # Exclude the centroid itself from the nearest distance
        EuDistance[index] = float('inf')  # exclude self

        # Sort data points by distance (smallest to largest) and get first K numbers of nearest neighbors
        # nearestIndices is a dictionary of indices in the EuDistance list of the nearest neighbours
        nearestIndices = np.argsort(EuDistance, kind='stable')[:6]  

        # Save the distances of the 6 nearest neighbours to the current centroid
        nearestDistances = [EuDistance[n] for n in nearestIndices]
        avg = np.mean(nearestDistances)
        standardDev = np.std(nearestDistances)  

        # Get the target values of the 6 nearest neighbors
        nearest = []
        for dist in nearestIndices:
            # As long as the point is relatively close to the centroid (distance<avg) OR if all points are equally close (stdDev is small)
            if EuDistance[dist] <= avg or standardDev <= threshold:
                nearest.append(dist)
                startPointEndPoint_list[x] = ([centroid[index],centroid[dist]])
                x += 1
                
    return startPointEndPoint_list

def strainCalc(centroids):
    sizes = []
    for datapoint in centroids:
        sizes.append(np.linalg.norm(np.array(datapoint[0]) - np.array(datapoint[1])))
    averageSize = sum(sizes)/len(sizes)
    sizeDeviation = np.std(sizes)
    print("Average size: ", averageSize)
    print("Standard deviation", sizeDeviation)
    return averageSize

def dislocationCalc(img, centroids, lines, squareSize, step):
    # 07/14 Roxxannia's Edit
    # Dislocation Calculation logistics
    # - create a square that's 30x30 (to start with), and move by 15px every time (both horizontally and vertically)
    # - when creating the square, make sure theres no overhang
    # - count number of centroids in the square, and the number of lines (that has a start point and an end point)
    # - 6*number of centroid - number of lines --> this is the dislocation within the square 
    # - then we can find the dislocation density by (# of dislocations /area of the square) --> save this number 
    # - average the dislocation density after iterating through all the squares
    # - calculate the standard deviation --> how uniform the sample surface is

    # For colour image, its a tuple
    h, w, _ = img.shape
    
    densities = []
    # For grayscale, its just a 2D array
    # h, w = im_gray.shape

    
    # Drawing a "30 x 30 square"
    for y in range(0, h + 1, step):
        border = False
        for x in range(0, w + 1, step):
            
            x0, y0 = x, y
            x1, y1 = x + squareSize, y + squareSize

            # Check if the square is overhanging
            if x1>= w: 
                x1 = w
                border = True
                
            if y1 >= h:
                y1 = h

            # If the square has reached the border, then start the next row of iteration
            if border == True:
                break

            # Finding centroids inside square
            centroidsInSquare = [cen for cen in centroids if x0 <= cen[0] < x1 and y0 <= cen[1] < y1]
            numCentroids = len(centroidsInSquare)
            
            
            # Finding lines that start or end in square
            linesInSquare = [line for line in lines if 
                            (x0 <= line[0][0] < x1 and y0 <= line[0][1] < y1) and
                            (x0 <= line[1][0] < x1 and y0 <= line[1][1] < y1)]
            
            numLines = len(linesInSquare)
            

            # Dislocation calculation
            dislocations = 6 * numCentroids - numLines
            
            # Dislocation density calculation
            area = (x1-x0)*conversion() * (y1-y0) * conversion()
            density = (dislocations / area)  #density in /nm^2
            densities.append(density)

        
    densities = np.array(densities)
    avgDensity = np.mean(densities) * 800 * 500
    stdDensity = np.std(densities) * 800 * 500
    print(densities)
    print("Average Density: ", avgDensity, " dislocations/nm2")
    print("Standard Deviation of Density: ", stdDensity)

    return densities, avgDensity, stdDensity

    # return True

if __name__ == "__main__":
    # Roxxannia's path
    # imagePath = "C:/Users/roxxa/OneDrive/University/Masters/Code/CrackThoseHexagons/VAT4-TESTING.jpg"
    # Sophie's path  
    imagePath = "VAT4-TESTING.jpg"

    # Estimated by hand
    predictedHexagonSize = 16 #nm

    minArea, maxArea = findThresholds(predictedHexagonSize)
    # print("min area: ", minArea)
    # print("max Area: ", maxArea)

    # Min distance for the centroids
    # This value could be calculated by the program based on hexagon size
    distanceThreshold = 7

    # an ODD int for Gaussian blur. Higher = more blur. (typically around 5 - 11)
    blurInt = 11
    if blurInt % 2 == 0:
        raise ValueError("blurInt must be odd!")

    outline, blurredImage = preProcessing(imagePath, blurInt)

    hexagons, centroids, output = detectHexagons(imagePath, blurredImage, outline, minArea, maxArea, distanceThreshold)

    startPointEndPoint = nearestNeighbours(centroids) 
    startPointEndPoint_clean = removeDuplicateNeighbours(startPointEndPoint)

    strainCalc(startPointEndPoint_clean)


    # The size of the square that's used to check the centroids
    squareSize = 30
    # How much the square slides every iteration
    # Tested for 5 and 15, the values dont seem to change much
    stepSize = 15
    densities, avgDensity, stdDensity = dislocationCalc(output, centroids, startPointEndPoint_clean, squareSize, stepSize)

    # Roxxannia's Note 07/13
    # I am kinda unsure if this calculation is right oop
    # Result for VAT4
    # Average Density:  0.014860760450946883
    # Standard Deviation of Density:  0.0057735345773644205
    
    