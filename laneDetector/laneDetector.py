import torch
import cv2

import scipy.special
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist
from playsound import playsound
from laneDetector.model import parsingNet
import img_directions
import threading

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
prev_x = -10000



left_img = img_directions.get_left_image()
right_img = img_directions.get_right_image()
straight_img = img_directions.get_straight_image()

def playSounds(path):
    playsound(path)

# left_img = cv2.imread('download.png')
# right_img=cv2.imread('right.png')
# straight_img=cv2.imread('stright.png')



# cv2.imshow('image',left_img)
# cv2.waitKey(0)
# print("left = ",left_img.shape)



def overlay_transparent(background, overlay, x, y):
    
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

class ModelConfig():

	def __init__(self, model_type):

		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18

class laneDetector():

	def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):

		self.use_gpu = use_gpu
		
		# Load model configuration based on the model type
		self.cfg = ModelConfig(model_type)

		# Initialize model
		self.model = self.initialize_model(model_path, self.cfg, use_gpu)

		# Initialize image transformation
		self.img_transform = self.initialize_image_transform()
  
  


	@staticmethod
	def initialize_model(model_path, cfg, use_gpu):

		# Load the model architecture
		net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
						use_aux=False) # we dont need auxiliary segmentation in testing


		# Load the weights from the downloaded model
		if use_gpu:
			if torch.backends.mps.is_built():
				net = net.to("mps")
				state_dict = torch.load(model_path, map_location='mps')['model'] # Apple GPU
			else:
				net = net.cuda()
				state_dict = torch.load(model_path, map_location='cuda')['model'] # CUDA
		else:
			state_dict = torch.load(model_path, map_location='cpu')['model'] # CPU

		compatible_state_dict = {}
		for k, v in state_dict.items():
			if 'module.' in k:
				compatible_state_dict[k[7:]] = v
			else:
				compatible_state_dict[k] = v

		# Load the weights into the model
		net.load_state_dict(compatible_state_dict, strict=False)
		net.eval()

		return net

	@staticmethod
	def initialize_image_transform():
		# Create transfom operation to resize and normalize the input images
		img_transforms = transforms.Compose([
			transforms.Resize((288, 800)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		return img_transforms

	def detect_lanes(self, image, voice_out, draw_points=True):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		output = self.inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

		# Draw depth image
		visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, voice_out, draw_points)

		return visualization_img

	def prepare_input(self, img):
		# Transform the image for inference
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_pil = Image.fromarray(img)
		input_img = self.img_transform(img_pil)
		input_tensor = input_img[None, ...]

		if self.use_gpu:
			if not torch.backends.mps.is_built():
				input_tensor = input_tensor.cuda()

		return input_tensor

	def inference(self, input_tensor):
		with torch.no_grad():
			output = self.model(input_tensor)

		return output

	@staticmethod
	def process_output(output, cfg):		
		# Parse the output of the model
		processed_output = output[0].data.cpu().numpy()
		processed_output = processed_output[:, ::-1, :]
		prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
		idx = np.arange(cfg.griding_num) + 1
		idx = idx.reshape(-1, 1, 1)
		loc = np.sum(prob * idx, axis=0)
		processed_output = np.argmax(processed_output, axis=0)
		loc[processed_output == cfg.griding_num] = 0
		processed_output = loc


		col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
		col_sample_w = col_sample[1] - col_sample[0]

		lanes_points = []
		lanes_detected = []

		max_lanes = processed_output.shape[1]
		for lane_num in range(max_lanes):
			lane_points = []
			# Check if there are any points detected in the lane
			if np.sum(processed_output[:, lane_num] != 0) > 2:

				lanes_detected.append(True)

				# Process each of the points for each lane
				for point_num in range(processed_output.shape[0]):
					if processed_output[point_num, lane_num] > 0:
						lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)

			lanes_points.append(lane_points)
		return np.array(lanes_points), np.array(lanes_detected)

	@staticmethod
	def draw_lanes(input_img, lanes_points, lanes_detected, cfg, voice_out, draw_points=True):
		# Write the detected line points in the image
		visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)
		
		
		change_colour = False
			
		#show the image of direction at the top corner
		# Draw a mask for the current lane
		if(lanes_detected[1] and lanes_detected[2]):
			lane_segment_img = visualization_img.copy()
			
			###
			# bottom_cen = (lanes_points[1][0][]+lanes_points[2][0])/2
			# print(lanes_points)
			# lanes_points[1][0]
			# if len(lanes_points)>=3:
			left_line =lanes_points[1]
			right_line =lanes_points[2]

			if(len(left_line)>=2 and len(right_line)>=2):
				bcenterx=(left_line[0][0]+right_line[0][0])/2
				# bcentery=(left_line[0][1]+right_line[0][1])/2

				tcenterx=(left_line[-1][0]+right_line[-1][0])/2
				# tcentery=(left_line[-1][1]+right_line[-1][1])/2
				
				# left_image='load a logo for left turn'
				# right_image='load a logo for right turn'

				
				#initial centers are same so skip 
				# direction="Right"
				deviation=tcenterx-bcenterx
				
				if deviation>180:
					# direction="Right"
					
					overlay = right_img
					visualization_img = overlay_transparent(visualization_img,overlay,10,10)
				
					# print("deviation=",deviation,direction)
				elif deviation<-180:
					# direction="Left"
					# print(visualization_img.shape)
					# cv2.imshow('image',visualization_img)
					# cv2.waitKey(0)
					overlay = left_img
					visualization_img = overlay_transparent(visualization_img,overlay,10,10)
     
					# overlay = cv2.resize(overlay,(12,12))
					# visualization_img = overlay_transparent(visualization_img,overlay,10,10)
					# print("deviation=",deviation,direction)
				else:
					# print("deviation= STRIGHsT",)
					overlay = straight_img
					visualization_img = overlay_transparent(visualization_img,overlay,10,10)

				#show deviation from center = {deviation}
				global prev_x
				if prev_x==-10000:
					prev_x=bcenterx
					change_colour = False
				else:
					if abs(prev_x-bcenterx)>200:
						cv2.putText(visualization_img,"Lane Switched",(550,60),cv2.FONT_HERSHEY_DUPLEX,1.5,(0,0,0),2,cv2.LINE_AA,False)
						change_colour = True
						if voice_out=="Yes":
							threading.Thread(target=playSounds,args=('laneDetector/sounds2.mp3',)).start()
					prev_x=bcenterx
			

			cur_color=(255,191,0)
			if change_colour==True:
				cur_color=(0,0,255)
			cv2.fillPoly(lane_segment_img, pts = [np.vstack((lanes_points[1][:25],np.flipud(lanes_points[2][:25])))], color =cur_color)
			visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)
			# prev = bottom_cen
		

		if(draw_points):
			for lane_num,lane_points in enumerate(lanes_points):
				temp = 0
				for lane_point in lane_points:
					cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)
					temp+=1
					if temp>25:
						break

		return visualization_img


'''
def findCenter(img):
    print(img.shape, img.dtype)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    #cv2.imshow("threshed", threshed);cv2.waitKey();cv2.destroyAllWindows()
    #_, cnts, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    M = cv2.moments(cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX,cY)

img1 = cv2.imread("img1.jpg")
img2 = cv2.resize(cv2.imread("img2.jpg"), None, fx=0.3, fy=0.3)

## (1) Find centers
pt1 = findCenter(img1)
pt2 = findCenter(img2)

## (2) Calc offset
dx = pt1[0] - pt2[0]
dy = pt1[1] - pt2[1]

## (3) do slice-op `paste`
h,w = img2.shape[:2]

dst = img1.copy()
dst[dy:dy+h, dx:dx+w] = img2

cv2.imwrite("res.png", dst)

'''

	







