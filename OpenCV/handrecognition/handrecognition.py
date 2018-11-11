import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while True:
	# grabbing frame of live video feed
	ret, frame = cap.read()
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	template = cv2.imread('opencv_frame_1.png', 0)

	orb = cv2.ORB_create()

	kp1, des1 = orb.detectAndCompute(template, None)
	kp2, des2 = orb.detectAndCompute(gray_frame, None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

	matches = bf.match(des1, des2)
	matches = sorted(matches, key=lambda x:x.distance)

	img3 = cv2.drawMatches(template, kp1, gray_frame, kp2, matches[:10], None, flags=2)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cv2.destroyAllWindows()
cap.release()
