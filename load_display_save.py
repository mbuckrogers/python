import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
face_cascade = cv2.CascadeClassifier('D:/dev/opencv/build/share/OpenCV/haarcascades/haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('D:/dev/opencv/build/share/OpenCV/haarcascades/haarcascade_eye.xml')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
print "Found {0} faces!".format(len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

		
height, width = img.shape[:2]
res = cv2.resize(img,(width /2, height / 2), interpolation = cv2.INTER_CUBIC)		
cv2.imshow("Faces found" ,res)
cv2.waitKey(0)

