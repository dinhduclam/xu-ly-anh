import datetime
import imutils
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt

image_path = "images/img_2.png"
winStride = (1, 1)
padding = (8, 8)
scale = 1.01

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

a= cv2.HOGDescriptor_getDefaultPeopleDetector()
# load the image and resize it
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
image = imutils.resize(image, width=min(400, image.shape[1]))

# detect people in the image
start = datetime.datetime.now()
(rects, weights) = hog.detectMultiScale(image, winStride=winStride,
	padding=padding, scale=scale)
end = datetime.datetime.now()

print("Total seconds = " + str((end-start).total_seconds()))

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

origin_image = image.copy()
non_nms_image = image.copy()
nms_image = image.copy()

# draw the original bounding boxes
for (x, y, w, h) in rects:
	cv2.rectangle(non_nms_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

for (x, y, w, h) in pick:
	cv2.rectangle(nms_image, (x, y), (x + w, y + h), (0, 255, 0), 1)


# show the output image
plt.figure(figsize = (16, 4))
plt.subplot(1, 3, 1)
plt.imshow(origin_image)
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(non_nms_image)
plt.title('Before Non-Maximum Suppression')
plt.subplot(1, 3, 3)
plt.imshow(nms_image)
plt.title('After Non-Maximum Suppression')

plt.show()