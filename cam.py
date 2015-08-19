import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import math

fac = 3

class DiceEye:
	def __init__(self):
		self.x = 0
		self.y = 0
		self.diameter = 0
	
	def __init__(self, x, y, diamter):
		self.x = int(x)
		self.y = int(y)
		self.diameter = int(diamter)
	
	
	def toString(self):
		return "x=" + str(self.x) + " y=" + str(self.y) + " d=" + str(self.diameter)

	def distance(self, other):
		return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2))
		
def average(diceeyes):
	sum = 0
	for diceye in diceeyes:
		sum += diceye.diameter
	return sum / len(diceeyes)

def unify(neighbourMap):
	for key in neighbourMap.iterkeys(): 
			b = neighbourMap[key]
			for values in neighbourMap.itervalues(): 	
				if len(b.intersection(values)) > 0:
					b = b.union(values)
					neighbourMap[key] = b
			
	res = []
	for values in neighbourMap.itervalues(): 	
		if values not in res:
			res.append(values)

	return [kp for kp in res if len(kp)> 0 ]

	
def distance(p1, p2):
	return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))
	

def isInTheHodd(can, hood, distance):
	tmp = de for de in hood if can.distance(de) <= distance ]
	return len(tmp) > 0
	
def findMyNeigbourhood(me, potentialNeighbours, distance):
		return set([pn for pn in potentialNeighbours if me.distance(pn) <= distance])
	
def collectN(res, pot, distance):
	r = []
	if len(pot) > 0:
		can = pot.pop()
		r = collectN(res, pot, distance)
	res.union(r)
	return res


def detectAndMark(img):
	kernel = np.ones((5,5),np.float32)/25
	img = cv2.filter2D(img,-1,kernel)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


	img = cv2.blur(img,(3,3))
	#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C,11,2)

	#img = cv2.Canny(img,100,200)
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()
	# Set up the detector with default parameters.
	# Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	# Filter by Area.
	params.filterByArea = True
	params.minArea = 100

	# Filter by Circularity
	params.filterByCircularity = True
	params.minCircularity = 0.7

	# Filter by Convexity
	params.filterByConvexity = True
	params.minConvexity = 0.87

	# Filter by Inertia
	params.filterByInertia = True
	params.minInertiaRatio = 0.4

	#params.minDistBetweenBlobs = 400.0

	detector = cv2.SimpleBlobDetector(params)
	# Detect blobs.
	keypoints = detector.detect(img)
	diceeyes = map(lambda kp: DiceEye(kp.pt[0], kp.pt[1], kp.size), keypoints)
	avg = average(diceeyes)
	
	
	diceeyes = [de for de in diceeyes if de.diameter < 1.5 * avg ]

	
	print collectN(set(), diceeyes, fac * avg)
	

	
	keypoints = [kp for kp in keypoints if kp.size < 1.5 * avg ]

	myNeighbours = {}
	nall = set

	

	for kp in keypoints:
		neighbours = filter(lambda kpOther:  distance(kpOther.pt, kp.pt) <= fac *avg, keypoints)
		assert len(neighbours) > 0
		myNeighbours[kp] = set(neighbours)
		print myNeighbours[kp]
		
		de = DiceEye(kp.pt[0], kp.pt[1], kp.size)
		print de.toString()
		
		#nall.add(myNeighbours[kp])

	res = unify(myNeighbours)

	#assert len(nall) == len(keypoints), "Difference 1 " + str(len(nall) - len(nall))

	i = 10
	for kps in res:
		if len(kps) <= 2:
			for kp in kps:
				cv2.circle(img, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size), (i,i, i), -1)
				print kp.pt[0], kp.pt[1]
			i += 60

	total = 0
	results = ""
	for kp in res:
		print len(kp)
		total += len(kp)
		results = results + str(len(kp)) +  ", "

	assert total == len(keypoints), "Difference " + str(total - len(keypoints))
	'''
		if(distance(kp1.pt, kp.pt) < 2 *avg):
			cv2.circle(img, (int(kp1.pt[0]), int(kp1.pt[1])), int(kp1.size), (255,0, 255), -1)
			cv2.circle(img, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size), (255,0, 255), -1)
	'''
	for kp in keypoints:
		#print str(kp.pt) + "  " + str(kp.size)
		#cv2.circle(img, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size), (255,0, 255), -1)
		cv2.circle(img, (int(kp.pt[0]), int(kp.pt[1])), int(avg * fac), (0,255, 0), 1, 4)
		#cv2.circle(img, (int(kp.pt[0]), int(kp.pt[1])), int(avg * 4), (0,0, 255), 1)

	
	cv2.putText(img, results,(30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
	cv2.line(img, (10,10), (10 + int(avg), 10), (0,255,0))
	
	img = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	return img

def toggle_images(event):
	global counter, files
	if event.key == 'n':
		counter += 1
	elif event.key == 'b':
		counter -= 1
	else: return
	
	print files[counter]
	localImg = cv2.imread(files[counter],1)
	localImg = cv2.cvtColor(localImg, cv2.COLOR_BGR2RGB)
	localImg = detectAndMark(localImg)
	plotter.set_data(localImg)
	plt.draw()
	
img = cv2.imread("E:\git\python\python-master\images\IMAG0171.jpg", 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = detectAndMark(img)
	
plotter = plt.imshow(img)
plt.title("Guckste")
plt.xticks([]),plt.yticks([])	

counter = 0	
plt.connect('key_press_event', toggle_images)


directory = os.path.join("E:\git\python\python-master\images", "*.jpg")
files = glob.glob(directory)	

plotter = plt.imshow(img)
plt.show()





