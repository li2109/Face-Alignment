# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

# def detect_faces(image, down_scale=1.5):
#     image_scaled = cv2.resize(image, None, fx=1.0/down_scale, fy=1.0/down_scale,
#                               interpolation=cv2.INTER_LINEAR)
#     faces = face_detector(image_scaled, 0)
#   #  faces = [face.rect for face in faces]
#     faces = scale_faces(faces, down_scale)
#     return faces


dir_name = ["Aligned", "Aligned/camera1/", "Aligned/camera2/"]

if not os.path.exists(dir_name[0]):                                                                                                                                           
	os.makedirs(dir_name[0])
if not os.path.exists(dir_name[1]):                                                                                                                                           
	os.makedirs(dir_name[1])
if not os.path.exists(dir_name[2]):                                                                                                                                           
	os.makedirs(dir_name[2])

for cam in range(2):
	for folder in range(100):
		folder_number = str(folder+1).zfill(3)
		pic_dir = str(dir_name[cam+1])+folder_number
		if not os.path.exists(pic_dir):
			os.makedirs(pic_dir)

cam1 = []
cam2 = []

for folder in range(100):
	for picture in range(8):
		folder_number = str(folder+1).zfill(3)
		picture_number = str(picture).zfill(4)
		cam1_source = "camera1/"+str(folder_number)+"/image_"+str(picture_number)+".pnm.ppm"
		cam2_source = "camera2/"+str(folder_number)+"/image_"+str(picture_number)+".pnm.ppm"
		cam1.append(cam1_source)
		cam2.append(cam2_source)


# detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)


# initialize dlib's face detector and then create
# the facial landmark predictor and the face aligner


for source in cam1:

	# load the input image, resize it, and convert it to grayscale
	# image = cv2.imread(source)
	image = cv2.imread(source, cv2.IMREAD_COLOR)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# show the original input image and detect faces in the grayscale
	# image
	# cv2.imshow("Input", image)
	rects = cnn_face_detector(image, 2)

	# loop over the face detections
	rect_nums = len(rects)
	for i in range(rect_nums):
		x = rects[i].rect.left()
		y = rects[i].rect.top()
		w = rects[i].rect.right() - x
		h = rects[i].rect.bottom() - y

		rec = dlib.rectangle(rects[i].rect.left(),rects[i].rect.top(),rects[i].rect.right(),rects[i].rect.bottom())

		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
		faceAligned = fa.align(image, gray, rec)
		if not os.path.isfile("Aligned/"+str(source)): 
			cv2.imwrite("Aligned/"+str(source), faceAligned)
		else:
			cv2.imwrite("Aligned/"+str(source)+str(1)+".ppm", faceAligned)

	# for face in faces_cnn:
	# 	# extract the ROI of the *original* face, then align the face
	# 	# using facial landmarks
	# 	# (x, y, w, h) = rect_to_bb(rect)
		

	# 	x = face.rect.left()
	# 	y = face.rect.top()
	# 	w = face.rect.right() - x
	# 	h = face.rect.bottom() - y
		
	# 	faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
	# 	faceAligned = fa.align(image, gray, face)
	# 	if not os.path.isfile("Aligned/"+str(source)): 
	# 		cv2.imwrite("Aligned/"+str(source), faceAligned)
	# 	else:
	# 		cv2.imwrite("Aligned/"+str(source)+str(1)+".ppm", faceAligned)

		# display the output images
		# cv2.imshow("Original", faceOrig)
		# cv2.imshow("Aligned", faceAligned)
		# cv2.waitKey(0)

for source in cam2:

	# load the input image, resize it, and convert it to grayscale
	# image = cv2.imread(source)
	image = cv2.imread(source, cv2.IMREAD_COLOR)
	image = imutils.resize(image, width=800)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# show the original input image and detect faces in the grayscale
	# image
	# cv2.imshow("Input", image)
	rects = cnn_face_detector(image, 1)

	# loop over the face detections
	rect_nums = len(rects)
	for i in range(rect_nums):
		x = rects[i].rect.left()
		y = rects[i].rect.top()
		w = rects[i].rect.right() - x
		h = rects[i].rect.bottom() - y

		rec = dlib.rectangle(rects[i].rect.left(),rects[i].rect.top(),rects[i].rect.right(),rects[i].rect.bottom())

		faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
		faceAligned = fa.align(image, gray, rec)
		if not os.path.isfile("Aligned/"+str(source)): 
			cv2.imwrite("Aligned/"+str(source), faceAligned)
		else:
			cv2.imwrite("Aligned/"+str(source)+str(1)+".ppm", faceAligned)