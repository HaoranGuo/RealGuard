import csv
import dlib
import numpy as np
import os
import cv2
import pyrealsense2 as rs

predictor_path = '/home/haoran/GitHub/dlib-anti-spoof/data/shape_predictor_5_face_landmarks.dat'
face_rec_model_path = '/home/haoran/GitHub/dlib-anti-spoof/data/dlib_face_recognition_resnet_model_v1.dat'
faces_folder = '/home/haoran/GitHub/dlib-anti-spoof/face'
FACES_FEATURES_CSV_FILE = '/home/haoran/GitHub/dlib-anti-spoof/data/face_features.csv'

FACES_FATURES_DISTANCE_THRESHOLD = 0.3

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def get_128d_features_of_face(image_path):
    image = cv2.imread(image_path)
    faces = detector(image, 1)

    if len(faces) != 0:
        shape = predictor(image, faces[0])
        face_descriptor = facerec.compute_face_descriptor(image, shape)
    else:
        face_descriptor = 0
    return face_descriptor

def get_mean_features_of_face(path):
    path = os.path.abspath(path)
    subDirs = [os.path.join(path, f) for f in os.listdir(path)]
    subDirs = list(filter(lambda x:os.path.isdir(x), subDirs))
    for index in range(0, len(subDirs)):
        subDir = subDirs[index]
        person_label = os.path.split(subDir)[-1]
        image_paths = [os.path.join(subDir, f) for f in os.listdir(subDir)]
        image_paths = list(filter(lambda x:os.path.isfile(x), image_paths))
        feature_list_of_person_x = []
        for image_path in image_paths:
            # 计算每一个图片的特征
            feature = get_128d_features_of_face(image_path)
            if feature == 0:
                continue
            
            feature_list_of_person_x.append(feature)
        
        # 计算当前人脸的平均特征
        features_mean_person_x = np.zeros(128, dtype=object, order='C')
        if feature_list_of_person_x:
            features_mean_person_x = np.array(feature_list_of_person_x, dtype=object).mean(axis=0)
        
        yield (features_mean_person_x, person_label)

def extract_features_to_csv(faces_dir):
    mean_features_list = list(get_mean_features_of_face(faces_dir))
    with open(FACES_FEATURES_CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for mean_features in mean_features_list:
            person_features = mean_features[0]
            person_label = mean_features[1]
            person_features = np.insert(person_features, 0, person_label, axis=0)
            writer.writerow(person_features)

def get_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    # 将feature_2转换为float64类型
    feature_2 = np.array(feature_2, dtype=np.float64)
    feature_sum = 0
    for i in range(len(feature_1)):
        feature_1[i] = float(feature_1[i])
        feature_2[i] = float(feature_2[i])
        feature_sq = (feature_1[i] - feature_2[i]) ** 2
        feature_sum += feature_sq
    dist = np.sqrt(feature_sum)
    return dist

def update_database(faces_folder_path):
    extract_features_to_csv(faces_folder_path)

def recognize_face(face_descriptor, csv_file, DISTANCE_THRESHOLD):
    database = []
    name = ""
    dist = 0
    with open(csv_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            database.append((row[0], row[1:]))
    
    face_feature_distance_list = []
    for face in database:
        dist = get_euclidean_distance(face_descriptor, face[1])
        dist = round(dist, 4)
        if dist < DISTANCE_THRESHOLD:
            face_feature_distance_list.append((face[0], dist))
    
    if len(face_feature_distance_list) == 0:
        return False, name, dist
    else:
        face_feature_distance_list.sort(key=lambda x:x[1])
        name = face_feature_distance_list[0][0]
        dist = face_feature_distance_list[0][1]
        return True, name, dist



# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# pipeline.start(config)

# # Load FACES_FEATURES_CSV_FILE to database
# database = []
# with open(FACES_FEATURES_CSV_FILE, "r") as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         database.append((row[0], row[1:]))

# win = dlib.image_window()
# while not win.is_closed():
#     frames = pipeline.wait_for_frames()
#     color_frame = frames.get_color_frame()
#     if not color_frame:
#         continue
#     image = np.asanyarray(color_frame.get_data())
#     faces = detector(image, 1)
#     person_dist = 0
#     person_label = ""
#     flag = 0
#     if len(faces) != 0:
#         for i in range(len(faces)):
#             face = faces[i]
#             shape = predictor(image, faces[0])
#             face_descriptor = facerec.compute_face_descriptor(image, shape)
#             face_feature_distance_list = []
#             for face_data in database:

#                 dist = get_euclidean_distance(face_descriptor, face_data[1])
#                 dist = round(dist, 4)

#                 if dist >= FACES_FATURES_DISTANCE_THRESHOLD:
#                     print("Unknow person")
#                     continue

#                 face_feature_distance_list.append((face_data[0], dist))
            

#             sorted(face_feature_distance_list, key=lambda x:x[1])
#             if face_feature_distance_list:
#                 person_dist = face_feature_distance_list[0][1]
#                 person_label = face_feature_distance_list[0][0]
#                 print("Person: {}, dist: {}".format(person_label, person_dist))

#     win.clear_overlay()
#     win.set_image(image)
#     win.add_overlay(faces)
