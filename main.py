import cv2
import numpy as np

def showImage(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)

def detect_hexagons(image_path, show_result=True):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or path is incorrect.")

    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    showImage("Blurred Image", blurred)

    #Threshold image to b&w
    # thresh = cv2.threshold(blurred, 100, 255,cv2.THRESH_BINARY)[1]
    th3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,19,5)


    #Edge detection
    edges = cv2.Canny(th3, 25,45)
    showImage("Edge Image", edges)

    #thicken edge lines
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 1)
    showImage("Dilated Edges", dilation)

    # Find contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    # Convert grayscale to BGR for visualization
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    hexagons = []
    centroids = []

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
                    cX, cY = findCentroids(contours = approx)
                    cv2.circle(output, (cX, cY), 0, (0, 255, 0), -1)
                    centroids.append((cX, cY))
                    #Need to record the cX and cY to compare later

                    cv2.drawContours(output, [approx], 0, (0, 255, 0), 1)

    if show_result:
        print(f"Detected {len(hexagons)} hexagons.")
        showImage("Detected Hexagons", output)

    
    return hexagons, centroids, output

def findCentroids(contours):
    M = cv2.moments(contours)
                
    # Preventing getting errors   
    if M["m00"] != 0:
        # Find the centroid of the image, convert it to binary format and then find its center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    return cX, cY

#k=6
def nearestNeighbours(centroid, k, image):
    neighbours = {}

    for index, c in enumerate(centroid):
        # Calculate euclidean distances between each centroid and all other centroids
        EuDistance = [np.linalg.norm(np.array(x) - np.array(c)) for x in centroid]
        # Exclude the centroid itself from the nearest distance
        EuDistance[index] = float('inf')  # exclude self

        # Sort data points by distance (smallest to largest) and get first K numbers of nearest neighbors
        # N_distance is just a group of indices of the closest points
        N_distance = np.argsort(EuDistance, kind='stable')[:k]  
          
        # Get the target values of the K nearest neighbors as long as they are under certain distance (pixels)
        nearest = []
        for dist in N_distance:
            if EuDistance[dist] < np.sqrt(15**2+15**2):
                nearest.append(dist)
 
        kNN = [centroid[i] for i in nearest]

        neighbours[index] = kNN

    for i in range(len(centroid)):
        for n in neighbours[i]:
            cv2.line(output, centroid[i], n, (255, 0, 0), 1)

    showImage("Nearest Neighbors", image)

    return neighbours

# Example usage:
if __name__ == "__main__":
    image_path = "C:/Users/roxxa/OneDrive/University/Masters/Code/CrackThoseHexagons/hexagons_medium.png"  # Replace with your image path
    hexagons, centroids, output = detect_hexagons(image_path)
    nearestNeighbours(centroids, 6, output)

    


