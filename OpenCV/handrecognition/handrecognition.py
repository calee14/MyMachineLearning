import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while True:
	img1 = cv2.imread('opencv_frame_0.png', 0)
	_, img2 = cap.read()
	# img2 = cv2.imread('opencv-feature-matching-image.jpg', 0)

	orb = cv2.ORB_create()

	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

	matches = bf.match(des1, des2)

	matches = sorted(matches, key=lambda x:x.distance)
	# Initialize lists
	list_kp1 = []
	list_kp2 = []

	# For each match...
	for mat in matches:

	    # Get the matching keypoints for each of the images
	    img1_idx = mat.queryIdx
	    img2_idx = mat.trainIdx

	    # x - columns
	    # y - rows
	    # Get the coordinates
	    (x1,y1) = kp1[img1_idx].pt
	    (x2,y2) = kp2[img2_idx].pt

	    # Append to each list
	    list_kp1.append((x1, y1))
	    list_kp2.append((x2, y2))
	    
	print(list_kp1[0], list_kp2[0])

	img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
	cv2.circle(img3, (int(list_kp1[0][0]), int(list_kp1[0][1])), 4, (0,0,255), -1)
	plt.imshow(img3)
	plt.show()
	break

cap.release()
cv2.destroyAllWindows()

'''
# import the necessary modules
import numpy as np
import argparse
import imutils
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
	help="Path to images where template will be matched")
ap.add_argument("-v", "--visulalize",
	help="Flag indicating whether or not to visulalize each iteration")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

cap = cv2.VideoCapture(0)

while True:
	# load the image, convert it to grayscale, and initialize the
	# book keeping variable to keep track of the matched region
	ret, image = cap.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	found = None

	# loop over the scales of the image
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])

		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the imge
		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

		# check to see if the iteration should be visualized
		# if args.get("visulalize", False):
		# 	# draw a bounding box around the detected region
		# 	clone = np.dstack([edged, edged, edged])
		# 	cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
		# 		(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
		# 	cv2.imshow("Visualize", clone)
		# 	cv2.waitKey(0)

		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)

	# unpack the book keeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

	# draw a bounding bo around the detected result and display the image
	cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
	cv2.imshow("Image", image)
	# cv2.waitKey(0)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
'''
'''
# loop over the images to find the template in
for imagePath in glob.glob(args["images"] + "/*.png"):
	# load the image, convert it to grayscale, and initialize the
	# book keeping variable to keep track of the matched region
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None

	# loop over the scales of the image
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])

		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the imge
		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

		# check to see if the iteration should be visualized
		if args.get("visulalize", False):
			# draw a bounding box around the detected region
			clone = np.dstack([edged, edged, edged])
			cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
			cv2.imshow("Visualize", clone)
			cv2.waitKey(0)

		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)

	# unpack the book keeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

	# draw a bounding bo around the detected result and display the image
	cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
# To run enter this command:
# python handrecognition.py --template cod_logo.png --images images
'''