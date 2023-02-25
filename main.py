import imutils
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

winStride = (4, 4)
padding = (0, 0)
scale = 1.1
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_human(image):
	# detect people in the image
	rects, weights = hog.detectMultiScale(image, winStride=winStride,
											padding=padding, scale=scale)

	pick = non_max_suppression(rects, probs=weights, overlapThresh=0.65)

	for (x, y, w, h) in pick:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	return image


def detectByPathVideo(path):
	video = cv2.VideoCapture(path)
	check, frame = video.read()

	if check == False:
		print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
		return
	print('Detecting people...')

	while video.isOpened():
		# check is True if reading was successful
		check, frame = video.read()
		if check:
			frame = imutils.resize(frame, width=min(800, frame.shape[1]))
			frame = detect_human(frame)
			cv2.imshow('Human detection', frame)

			key = cv2.waitKey(10)
			if key == ord('q'):
				break
		else:
			break
	video.release()
	cv2.destroyAllWindows()
	print("Saved")

detectByPathVideo("videos/vid1.mp4")


