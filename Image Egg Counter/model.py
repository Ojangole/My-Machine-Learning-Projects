# Image processing using OpenCV to count number of items in image

#Step 1: Reading image and coverting from RGB to HSV color space
import numpy as np
import cv2
import math
from google.colab.patches import cv2_imshow

img = cv2.imread('inputImage.png') #read image
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert to hsv

#Step 2: Get red color range
 # range for lower red
lower_red = np.array([0,70,0])
upper_red = np.array([40,255,255])
mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

 #range for upper red
lower_red = np.array([170,70,0])
upper_red = np.array([180,255,255])
mask2 = cv2.inRange(hsv_img, lower_red, upper_red)

 # mask for lower and upper red
mask = mask1 + mask2

 # get image in red pixel only
redImage = cv2.bitwise_and(img.copy(), img.copy(), mask=mask)

#Step 3: Gray Scale, Gaussian Blur and Image thresholding
gray = cv2.cvtColor(redImage, cv2.COLOR_BGR2GRAY)
blured = cv2.GaussianBlur(gray, (5,5),0)

ret, thresh = cv2.threshold(blured,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#Step 4: Remove small object usig Morphological Transformation operation Closing
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#Step 5: Find contour area, average area and remove too small areas
 #Find contour area and calculate average gpa
negative = cv2.bitwise_not(closing) # negative image
contours, hierarchy = cv2.findContours(negative, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #np.negative or negative?
hierarchy = hierarchy[0]
max_area = cv2.contourArea(contours[0])
total = 0 # total contour size
for con in contours:
     area = cv2.contourArea(con) # get contour size
     total += area
     if area > max_area:
        max_area = area
diff = 0.1 # smallest contour have to bigger than (diff * max_area)
max_area = int(max_area * diff) # smallest contour have to bigger
average = int(total / (len(contours))) # average size for contour
radius_avg = int(math.sqrt(average / 3.14)) # average radius 

average = int(average * diff)

 #remove small area
# Remove small object
mask = np.zeros(negative.shape[:2],dtype=np.uint8)
for component in zip(contours, hierarchy):
     currentContour = component[0]
     currentHierarchy = component[1]
     area = cv2.contourArea(currentContour)
     if currentHierarchy[3] < 0 and area > average:
          cv2.drawContours(mask, [currentContour], 0, (255), -1)

#Step 6: Find result by using Distance Transform and and Watershed Algorithm or find circle in each contour by using HoughCircles
 #Using HoughCircles in each contour because we know eggs is circle shape
res1 = img.copy()
count = 0 #result
for con in contours:
     area = cv2.contourArea(con)
     radian = int(math.sqrt(area / 3.14))
     minRad = int(radian * 0.3)
     maxRad = int(radian * 2)
     mask_temp = np.zeros(mask.shape[:2],dtype=np.uint8)
     cv2.drawContours(mask_temp, [con], 0, (255), -1)
     circles = cv2.HoughCircles(mask_temp,cv2.HOUGH_GRADIENT,1, 1.2 * radian, param1=100,param2=10,minRadius=minRad,maxRadius=maxRad)
     if circles is not None: 
          circles = np.uint16(np.around(circles))
          for i in circles[0, :]:
               radius = i[2]
               if radius > radius_avg:
                    count += 1
                    center = (i[0], i[1]) # circle center
                    cv2.putText(res1, str(count), center,      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Put text at center    
                    cv2.circle(res1, center, radius, (0, 0, 255), 3) 
print('number of object is', count)
cv2_imshow(res1)

# Using Distance Transform and Watershed Algorithm
 # sure background area
sure_bg = cv2.erode(mask, kernel)
 # Finding sure foreground area
dist_transform = cv2.distanceTransform(mask,cv2.DIST_L2,5)
cv2_imshow(dist_transform)
 # Draw sure figure from distance transform
ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0) 
 # 0.2 is important, the bigger it is, the object is smaller (to the object center)
sure_fg = np.uint8(sure_fg)
 #Find contour for sure figure
contours, hierarchy = cv2.findContours(sure_fg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 0
result = img.copy()
contours_poly = [None]*len(contours)
centers = [None]*len(contours)
radius = [None]*len(contours)
for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

totalRadius = 0
for i in range(len(contours)):
  totalRadius += radius[i]
averageRadius = totalRadius / len(contours)
diff_average_radius = 0.3

for i in range(len(contours)):
  if radius[i] > averageRadius * diff_average_radius:  
    count += 1
    cv2.circle(result, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), (0,255,0), 2) # Draw circle
    cv2.putText(result, str(count), (int(centers[i][0]), int(centers[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) # Put text
  
cv2_imshow(result)
