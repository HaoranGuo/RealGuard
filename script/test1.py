import dlib_recognize_face
import cv2
import numpy as np

detect_path = "./model/mmod_human_face_detector.dat"
predictor_path = "./model/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./model/dlib_face_recognition_resnet_model_v1.dat"
faces_folder = "./face"
FACES_FEATURES_CSV_FILE = "./data/face_features.csv"

DRecFace = dlib_recognize_face.Recognize_Face(detect_path, predictor_path, face_rec_model_path, FACES_FEATURES_CSV_FILE, faces_folder)

image = cv2.imread("./face/sb/7d28a2f801b3b4e2bd87f89711327169.png")
image = np.array(image)

isreg, isnew, dist = DRecFace.register_face(image, 'ghr', '002')
print(dist)

# isreg, isnew, dist = DRecFace.register_face(image, 'ghr', '002')
# print(dist)