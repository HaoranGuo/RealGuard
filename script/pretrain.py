import dlib_recognize_face
import time

predictor_path = './model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = './model/dlib_face_recognition_resnet_model_v1.dat'
faces_folder = './face'
FACES_FEATURES_CSV_FILE = './data/face_features.csv'

# Calculate running time
start_time = time.time()

ExtractFace = dlib_recognize_face.Extract_Face_Feature(predictor_path, face_rec_model_path, faces_folder, FACES_FEATURES_CSV_FILE)
# ExtractFace.update_features_csv()
ExtractFace.add_face()

end_time = time.time()
print('Running time: %s Seconds'%(end_time-start_time))