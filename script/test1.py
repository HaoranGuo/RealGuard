import dlib_recognize_face
import cv2
import numpy as np
import dlib

detect_path = "./model/mmod_human_face_detector.dat"
predictor_path = "./model/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./model/dlib_face_recognition_resnet_model_v1.dat"
faces_folder = "./face"
FACES_FEATURES_CSV_FILE = "./data/face_features.csv"

DISTANCE_THRESHOLD = 0.35

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

DRecFace = dlib_recognize_face.Recognize_Face(detect_path, predictor_path, face_rec_model_path, FACES_FEATURES_CSV_FILE, faces_folder)

image = cv2.imread("./face/sb/7d28a2f801b3b4e2bd87f89711327169.png")
image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
image = np.array(image)

t = detector(image, 1)
if len(t) == 0:
    print("No Face Detected")
    exit()
shape = sp(image, t[0])
face_descriptor = facerec.compute_face_descriptor(image, shape)

i1, i2, i3, i4 = DRecFace.recognize_face(face_descriptor, DISTANCE_THRESHOLD)
print(i2, i3, i4)

# isreg, isnew, dist = DRecFace.register_face(image, 'ghr', '002')
# print(dist)