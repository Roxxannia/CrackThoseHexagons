import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

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
    # print(contours[0])

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
            if area > 10 and area < 80: 
                # area < 120 with previous image used
                # intersects = False
                
                #check if the hexagon intersects with a pre-existing hexagons
                # Create a mask for the current hexagon

                # mask = np.zeros(img.shape, dtype=np.uint8)
                # cv2.drawContours(mask, [approx], -1, 255, -1)

                # for prev_hex in hexagons:
                #     prev_mask = np.zeros(img.shape, dtype=np.uint8)
                #     cv2.drawContours(prev_mask, [prev_hex], -1, 255, -1)
                #     # Check intersection
                #     intersection = cv2.bitwise_and(mask, prev_mask)
                #     if np.any(intersection):
                #         intersects = True
                #         break

                # if not intersects:
                #     hexagons.append(approx)
                #     cX, cY = findCentroids(contours = approx)
                #     cv2.circle(output, (cX, cY), 0, (0, 255, 0), -1)
                #     centroids.append((cX, cY))
                #     #Need to record the cX and cY to compare later

                #     cv2.drawContours(output, [approx], 0, (0, 255, 0), 1)

                #---------New method------
                hexagons.append(approx)
                cX, cY = findCentroids(contours = approx)
                centroids.append(np.array([cX, cY]))
    
    filtered_hexagons, filtered_centroids = remove_duplicate_hexagons(hexagons, centroids, threshold = 3)
    for c in filtered_hexagons:
        cv2.drawContours(output, [c], -1, (0, 255, 0), 1)

    for h in filtered_centroids:
        cv2.circle(output, h, 0, (0, 255, 0), -1)

    

    if show_result:
        print(f"Detected {len(filtered_hexagons)} hexagons.")
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

def remove_duplicate_hexagons(hexagons, centroids, threshold):
    n = len(centroids)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if keep[j] and np.linalg.norm(centroids[i] - centroids[j]) < threshold:
                keep[j] = False  # Remove duplicate

    filtered_hexagons = [hexagons[i] for i in range(n) if keep[i]]
    filtered_centroids = [centroids[i] for i in range(n) if keep[i]]

    return filtered_hexagons, filtered_centroids



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
            cv2.line(image, centroid[i], n, (255, 0, 0), 1)

    showImage("Nearest Neighbors", image)
    # cv2.imwrite('Nearest_neigh.png',image)

    return neighbours

def conversion():
    # Scale bar 200nm 
    # Scale bar = 136 pixels
    # about 1.47 nm/pixel

    scale = 200/136

    return round(float(scale),3)

def contourMap(output, centroids, neighbours):
    # Create blank with the dimension of the original image
    height, width = output.shape[:2]
    strain_map = np.zeros((height, width))
    count_map = np.zeros((height, width))

    

    for i, center in enumerate(centroids):
        neighbour_pts = neighbours[i]
        if len(neighbour_pts) == 0:
            continue

        # Mean distance to neighbors
        distances = [np.linalg.norm(np.array(center) - np.array(n)) for n in neighbour_pts]

        strain = 1/np.mean(distances)  #Smaller distance, less strain, 

        # Accumulate strain at the centroid location
        cx, cy = center
        if 0 <= cy < height and 0 <= cx < width:
            strain_map[cy, cx] += strain
            count_map[cy, cx] += 1

    # y = row 
    # x = column

    # Avoid division by zero
    count_map[count_map == 0] = 1
    strain_map = strain_map / count_map

    # Optional: smooth the strain map
    smooth_strain = gaussian_filter(strain_map, sigma=10)

    # Step 4: Plot the strain contour map
    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, smooth_strain, levels=50, cmap='plasma')
    plt.colorbar(contour, label='Estimated Local Strain')
    plt.scatter([x for x, y in centroids], [y for x, y in centroids], color='white', s=3, label='Centroids')
    plt.title("Contour Map of Relaxation Based on Dislocation")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    # plt.savefig('contourmap.png')


# Example usage:
if __name__ == "__main__":

    # start_time = time.time()

    image_path = "C:/Users/roxxa/OneDrive/University/Masters/Code/CrackThoseHexagons/hex-2.png"  # Replace with your image path
    hexagons, centroids, output = detect_hexagons(image_path)
    # print(centroids)
    
    # Obtain the neighbours
    # neighbours = nearestNeighbours(centroids, 6, output) 

    # contourMap(output, centroids, neighbours)
    
    conversion()

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(elapsed_time)
    


