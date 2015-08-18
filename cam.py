import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import glob


def detectAndMark(img):
	kernel = np.ones((5,5),np.float32)/25
	img = cv2.filter2D(img,-1,kernel)
	#img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	
	
	img = cv2.blur(img,(3,3))
	#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C,11,2)
	
	#img = cv2.Canny(img,100,200)
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

	for kp in keypoints:
		print str(kp.pt) + "  " + str(kp.size)
	
	
	img = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	return img

def toggle_images(event):
	if event.key != 'n': return
	global counter, files
	counter += 1
	print files[counter]
	localImg = cv2.imread(files[counter],0)
	localImg = detectAndMark(localImg)
	plotter.set_data(localImg)
	plt.draw()
	
img = cv2.imread("C:\dev\yatzee_images\IMAG0171.jpg",0)
img = detectAndMark(img)
	
plotter = plt.imshow(img)
plt.title("Guckste")
plt.xticks([]),plt.yticks([])	

counter = 0	
plt.connect('key_press_event', toggle_images)


directory = os.path.join("C:\dev\yatzee_images", "*.jpg")
files = glob.glob(directory)	

plotter = plt.imshow(img,'gray')
plt.show()





