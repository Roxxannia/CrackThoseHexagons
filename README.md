# CrackThoseHexagons




dimension of the picture is 710 x 474 

- create a square that's 30x30 (to start with), and move by 15px every time (both horizontally and vertically)
    - when creating the square, make sure theres no overhang
- count number of centroids in the square, and the number of lines (that has a start point and an end point)
- 6*number of centroid - number of lines --> this is the dislocation within the square 
- then we can find the dislocation density by (# of dislocations /area of the square) --> save this number 
- average the dislocation density after iterating through all the squares
- calculate the standard deviation --> how uniform the sample surface is