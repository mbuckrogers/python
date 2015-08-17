import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('D:/temp/yatzee_images/IMAG0239.jpg',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

kernel = np.ones((5,5),np.float32)/25
img = cv2.filter2D(img,-1,kernel)


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Set up the detector with default parameters.

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.7

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.10

detector = cv2.SimpleBlobDetector(params)
 
# Detect blobs.
keypoints = detector.detect(img)
print keypoints
img = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 


titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]


"""
for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
"""

plt.subplot(1,1,0),plt.imshow(images[0],'gray')
plt.title(titles[0])
plt.xticks([]),plt.yticks([])

plt.show()