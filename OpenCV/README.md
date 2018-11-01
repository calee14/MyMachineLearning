# OpenCV
This will be where I store my OpenCV work for image analyse.

# Notes:
- Image Preparation:
	- Get images of a hand and remove background and only include the the shape of the hand
	- Remove color; Gray out image of hand. (this will help the program focus only on the shape of the hand)
	- Could also get the edges of the hand.
	- Live steam video data needs to be grayed out the same way as the hand image data
- Hand Detection:
	- Use tne template matching method to match the hands on the screen. If the method doesn't work with smaller images of the give image then we can use it.
	- Use the feature matching method. The feature matching method will match features from both images that are similar. Since we have only one hand in the image given then it should lock down on what is paired. If this method isn't effective, there are parameters to change how percise it has to be.
- Hand Gestures and Motion:
	- To understand the gesture of the hand we need to watch where the hand is being moved. If the hand is moving accross from left to right we know the use wants to move to the previous display. Vice Versa with the hand moving from right to left.
	- The zoom in gesture, will have two hands held close together and then move away from each other.