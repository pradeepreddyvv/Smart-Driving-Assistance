import cv2
import pafy
from laneDetector import laneDetector, ModelType

# model_path = "models/tusimple_18.pth"
model_path = "models/culane_18.pth"
# model_type = ModelType.TUSIMPLE
model_type = ModelType.CULANE
use_gpu = True

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

'''
videoUrl = 'https://youtu.be/2CIxM7x-Clc'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
'''

# cap = cv2.VideoCapture('lane.mp4')
cap = cv2.VideoCapture('traffic.mp4')
# cap = cv2.VideoCapture('Nice Road Bangalore - Tumkur Road to Hosur Road 2018.mp4')

# Initialize lane detection model
lane_detector = laneDetector(model_path, model_type, use_gpu)

cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	

while cap.isOpened():
	try:
		# Read frame from the video
		ret, frame = cap.read()
	except:
		continue

	if ret:	

		# Detect the lanes
		output_img = lane_detector.detect_lanes(frame,"Yes")
		

		cv2.imshow("Detected lanes", output_img)

	else:
		break

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()