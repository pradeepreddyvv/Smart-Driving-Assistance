import cv2
import pafy
from laneDetector import laneDetector, ModelType
import numpy as np
import time
from grabscreen import grab_screen
from tensorflow.keras.models import load_model
import pyttsx3
from collections import defaultdict
from playsound import playsound
import threading
engine = pyttsx3.init()
engine.setProperty('rate',350)

soundmap = {}

#Speak
def speakSign(label):
    
	# engine.say(label)
	# engine.runAndWait()
	filename = soundmap[label]
	playsound("sounds/"+filename+".mp3")

def soundDictCreation():
	i=1
	with open("signs_classes.txt", "r") as f:
		for line in f:
			line = line.strip()
			soundmap[line] = str(i)
			i+=1


def solve(lane_sound ,traffic_sound ,road_lanes ,traffic_sign):
	soundDictCreation()
	# cap = cv2.VideoCapture('A_both_test_1.mp4')
	# cap = cv2.VideoCapture('A_Lane Detection Test Video 01.mp4')
	# cap = cv2.VideoCapture('A_lane_traffic.mp4')
	cap = cv2.VideoCapture('Bangalore Electronic City Elevated Tollway.mp4')
	# cap = cv2.VideoCapture('Mumbai-Agra_NH-60_lane_traffic_3_l.mp4')
	# cap = cv2.VideoCapture('Nice_Road_Bangalore_lane_switch_test.mp4')
	# cap = cv2.VideoCapture('Night Drive - Delhi 4K - India.mp4')
	# cap = cv2.VideoCapture('Odhisa to Kolkata Raw Dash cam Video Nh 60.mp4')
	# cap = cv2.VideoCapture('test4.mp4')
	# cap = cv2.VideoCapture('traffic.mp4')
	if (traffic_sign=="No" and traffic_sound=="No" and road_lanes=="No" and  lane_sound =="No"):
		return 0
	elif (traffic_sign=="No" and traffic_sound=="Yes"):
		return 0
	elif (traffic_sign=="Yes" and traffic_sound=="Yes" and road_lanes=="No" and  lane_sound =="No"):
		voice_out = "No"
		if(lane_sound=="Yes"):
			voice_out = "Yes"
		else:
			voice_out = "No"

		# model_path = "models/tusimple_18.pth"
		model_path = "models/culane_18.pth"
		# model_type = ModelType.TUSIMPLE
		model_type = ModelType.CULANE
		use_gpu = True
		# print("solve = ",lane_sound ,traffic_sound ,road_lanes ,traffic_sign)


		# Initialize video
		# cap = cv2.VideoCapture("video.mp4")

		'''
		videoUrl = 'https://youtu.be/2CIxM7x-Clc'
		videoPafy = pafy.new(videoUrl)
		print(videoPafy.streams)
		cap = cv2.VideoCapture(videoPafy.streams[-1].url)
		'''
		'''Input size expected by the classification_model'''
		HEIGHT = 32
		WIDTH = 32

		# yolo_net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_training.cfg")
		yolo_net = cv2.dnn.readNet("yolov4-tiny_training_last.weights", "yolov4-tiny_training.cfg")
		# yolo_net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_training.cfg")

		classes = []
		with open("signs.names.txt", "r") as f:
			classes = [line.strip() for line in f.readlines()]

		#get last layers names
		layer_names = yolo_net.getLayerNames()
		output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
		## colors = np.random.uniform(0, 255, size=(len(classes), 3))
		## check_time = True
		confidence_threshold = 0.5
		font = cv2.FONT_HERSHEY_SIMPLEX
		## start_time = time.time()
		frame_count = 0
		frame_count_voice = 0
		## detection_confidence = 0.5
		# cap = cv2.VideoCapture(0)
		font = cv2.FONT_HERSHEY_SIMPLEX
		'''Load Classification Model'''
		classification_model = load_model('traffic.h5') #load mask detection model
		classes_classification = []
		with open("signs_classes.txt", "r") as f:
			classes_classification = [line.strip() for line in f.readlines()]

		'''Test: Input File'''
		# cap = cv2.VideoCapture('lane.mp4')
		# cap = cv2.VideoCapture('traffic.mp4')
		# cap = cv2.VideoCapture('lane_switch_test.mp4')
		# cap = cv2.VideoCapture('videoplay.mp4')
		'''Test: WEBCAM '''
		# ret, img = video_capture.read()
		# ret, img = cap.read()
		'''Test: SCREEN '''
		# frame = grab_screen(region=(0,70,800,680))
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
		dd = defaultdict(int)
		prev = None

		cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	
		'''Test: SCREEN '''
		# while True:
      
		# Test: Input File
		while cap.isOpened():
			try:
				# Read frame from the video
				'''Test: input file'''
				ret, frame = cap.read()
				'''Test: Screen'''
		
				frame_count +=1
				if frame_count%2==0:
					continue
			except:
				continue
			# Test: Input File
			if ret:	
			# if True:
				#get image shape

				height, width, channels = frame.shape
				window_width = width

				# Detecting objects (YOLO)
				blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
				yolo_net.setInput(blob)
				outs = yolo_net.forward(output_layers)

				# Showing informations on the screen (YOLO)
				class_ids = []
				confidences = []
				boxes = []
				for out in outs:
					for detection in out:
						scores = detection[5:]
						class_id = np.argmax(scores)
						confidence = scores[class_id]
						if confidence > confidence_threshold:
							# Object detected
							center_x = int(detection[0] * width)
							center_y = int(detection[1] * height)
							w = int(detection[2] * width)
							h = int(detection[3] * height)
							# Rectangle coordinates
							x = int(center_x - w / 2)
							y = int(center_y - h / 2)
							boxes.append([x, y, w, h])
							confidences.append(float(confidence))
							class_ids.append(class_id)
				indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
				count = 0

			
				for i in range(len(boxes)):
					if i in indexes:
						x, y, w, h = boxes[i]
						label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
						cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
						'''crop the detected signs -> input to the classification model'''
						crop_img = frame[y:y+h, x:x+w]
						if h>0 and w>0 and y>0 and x>0:
							if len(crop_img)>0:
								crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
								crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
								prediction = np.argmax(classification_model.predict(crop_img))
								label = str(classes_classification[prediction])
								# if label in dq:
								#     continue
								# else:
								#     dq.append(label)
								####
								# if(frame_count_voice%2==0):
								# 	threading.Thread(target=speakSign,args=(label,)).start()
								# frame_count_voice+=1

								if label not in dd:
									# threading.Thread(target=speakSign,args=(label,)).start()
									dd[label]+=1
								else:
									dd[label]+=1
									if dd[label]>2 and label!=prev:
										threading.Thread(target=speakSign,args=(label,)).start()
										# engine.say(label)
										# engine.runAndWait()
										prev=label
										dd.clear()
								# engine.say(label)
								# engine.runAndWait()
								cv2.putText(frame, label, (x, y), font, 0.5, (255,0,0), 2)
				cv2.imshow("Detected lanes", frame)

			else:
				break

			# Press key q to stop
			if cv2.waitKey(1) == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()
	elif traffic_sign=="Yes":
		voice_out = "No"
		if(lane_sound=="Yes"):
			voice_out = "Yes"
		else:
			voice_out = "No"

		# model_path = "models/tusimple_18.pth"
		model_path = "models/culane_18.pth"
		# model_type = ModelType.TUSIMPLE
		model_type = ModelType.CULANE
		use_gpu = True
		# print("solve = ",lane_sound ,traffic_sound ,road_lanes ,traffic_sign)


		# Initialize video
		# cap = cv2.VideoCapture("video.mp4")

		'''
		videoUrl = 'https://youtu.be/2CIxM7x-Clc'
		videoPafy = pafy.new(videoUrl)
		print(videoPafy.streams)
		cap = cv2.VideoCapture(videoPafy.streams[-1].url)
		'''
		'''Input size expected by the classification_model'''
		HEIGHT = 32
		WIDTH = 32

		# yolo_net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_training.cfg")
		yolo_net = cv2.dnn.readNet("yolov4-tiny_training_last.weights", "yolov4-tiny_training.cfg")
		# yolo_net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_training.cfg")

		classes = []
		with open("signs.names.txt", "r") as f:
			classes = [line.strip() for line in f.readlines()]

		#get last layers names
		layer_names = yolo_net.getLayerNames()
		output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
		## colors = np.random.uniform(0, 255, size=(len(classes), 3))
		## check_time = True
		confidence_threshold = 0.5
		font = cv2.FONT_HERSHEY_SIMPLEX
		## start_time = time.time()
		frame_count = 0

		## detection_confidence = 0.5
		# cap = cv2.VideoCapture(0)
		font = cv2.FONT_HERSHEY_SIMPLEX
		'''Load Classification Model'''
		classification_model = load_model('traffic.h5') #load mask detection model
		classes_classification = []
		with open("signs_classes.txt", "r") as f:
			classes_classification = [line.strip() for line in f.readlines()]

		'''Test: Input File'''
		# cap = cv2.VideoCapture('lane.mp4')
		# cap = cv2.VideoCapture('traffic.mp4')
		# cap = cv2.VideoCapture('lane_switch_test.mp4')
		# cap = cv2.VideoCapture('videoplay.mp4')
		'''Test: WEBCAM '''
		# ret, img = video_capture.read()
		# ret, img = cap.read()
		'''Test: SCREEN '''
		# frame = grab_screen(region=(0,70,800,680))
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
		dd = defaultdict(int)
		prev = None
		# Initialize lane detection model
		if road_lanes=="Yes":
			lane_detector = laneDetector(model_path, model_type, use_gpu)

		cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	
		'''Test: SCREEN '''
		# while True:
      
		# Test: Input File
		while cap.isOpened():
			try:
				# Read frame from the video
				'''Test: input file'''
				ret, frame = cap.read()
				'''Test: Screen'''
		
				frame_count +=1
				if frame_count%2==0:
					continue
			except:
				continue
			# Test: Input File
			if ret:	
			# if True:
				#get image shape

				height, width, channels = frame.shape
				window_width = width

				# Detecting objects (YOLO)
				blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
				yolo_net.setInput(blob)
				outs = yolo_net.forward(output_layers)

				# Showing informations on the screen (YOLO)
				class_ids = []
				confidences = []
				boxes = []
				for out in outs:
					for detection in out:
						scores = detection[5:]
						class_id = np.argmax(scores)
						confidence = scores[class_id]
						if confidence > confidence_threshold:
							# Object detected
							center_x = int(detection[0] * width)
							center_y = int(detection[1] * height)
							w = int(detection[2] * width)
							h = int(detection[3] * height)
							# Rectangle coordinates
							x = int(center_x - w / 2)
							y = int(center_y - h / 2)
							boxes.append([x, y, w, h])
							confidences.append(float(confidence))
							class_ids.append(class_id)
				indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
				count = 0

			
				for i in range(len(boxes)):
					if i in indexes:
						x, y, w, h = boxes[i]
						label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
						cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
						'''crop the detected signs -> input to the classification model'''
						crop_img = frame[y:y+h, x:x+w]
						if h>0 and w>0 and y>0 and x>0:
							if len(crop_img)>0:
								crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
								crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
								prediction = np.argmax(classification_model.predict(crop_img))
								label = str(classes_classification[prediction])
								# if label in dq:
								#     continue
								# else:
								#     dq.append(label)
								####
        
								# threading.Thread(target=speakSign,args=(label,)).start()
								if label not in dd:
									dd[label]=1
								else:
									dd[label]+=1
									if dd[label]>1 and label!=prev:
										if traffic_sound=="Yes":
											threading.Thread(target=speakSign,args=(label,)).start()
											# engine.say(label)
											# engine.runAndWait()
										prev=label
										dd.clear()
          
								# engine.say(label)
								# engine.runAndWait()
								cv2.putText(frame, label, (x, y), font, 0.5, (255,0,0), 2)
				if road_lanes=="Yes":
					# Detect the lanes
					frame = lane_detector.detect_lanes(frame,voice_out)
				cv2.imshow("Detected lanes", frame)

			else:
				break

			# Press key q to stop
			if cv2.waitKey(1) == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()
	elif (traffic_sign=="No" and traffic_sound=="No" and road_lanes=="Yes" and lane_sound=="Yes"):
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
		# cap = cv2.VideoCapture('traffic.mp4')
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
	elif (traffic_sign=="No" and traffic_sound=="No" and road_lanes=="Yes" and lane_sound=="No"):
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
		# cap = cv2.VideoCapture('traffic.mp4')
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
				output_img = lane_detector.detect_lanes(frame,"No")
				

				cv2.imshow("Detected lanes", output_img)

			else:
				break

			# Press key q to stop
			if cv2.waitKey(1) == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()
	else:
		return 0