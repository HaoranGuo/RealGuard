import csv
import dlib
import numpy as np
import os
import cv2
import pyrealsense2 as rs
import validate_face

predictor_path = '/home/haoran/GitHub/RealGuard/model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = '/home/haoran/GitHub/RealGuard/model/dlib_face_recognition_resnet_model_v1.dat'
faces_folder = '/home/haoran/GitHub/RealGuard/face'
FACES_FEATURES_CSV_FILE = '/home/haoran/GitHub/RealGuard/data/face_features.csv'

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

def recognize_from_3_frame_picture(ir_image_batch, depth_image, 
                                   DEPTH_SCALE, DISTANCE_THRESHOLD, 
                                   PREDICTOR_FILE, REC_MODEL_FILE, 
                                   CSV_FILE):
    # 如果ir_image_batch的size不是3，返回False
    if len(ir_image_batch) != 3:
        return False, None, None
    # 如果ir_image_batch第一张和第三张没有人脸，返回False
    detector_ = dlib.get_frontal_face_detector()
    face0 = detector_(ir_image_batch[0], 1)
    face1 = detector_(ir_image_batch[2], 1)
    if len(face0) == 0 and len(face1) == 0:
        return False, None, None
    IS_PEOPLE = False
    IS_VALIDATE = False
    IS_RECOGNIZE = False

    predictor_ = dlib.shape_predictor(PREDICTOR_FILE)
    shape0 = predictor_(ir_image_batch[0], face0[0])
    # shape1 = predictor_(ir_image_batch[2], face1[0])
    # if shape0.num_parts != 68 or shape1.num_parts != 68:
    if shape0.num_parts != 68:
        return False, None, None
    else:
        IS_PEOPLE = True
    
    if not validate_face.validate_face(depth_image, DEPTH_SCALE, shape0):
        return False, None, None
    else:
        IS_VALIDATE = True

    face_rec_model_ = dlib.face_recognition_model_v1(REC_MODEL_FILE)
    image = cv2.cvtColor(ir_image_batch[2], cv2.COLOR_GRAY2BGR)
    # face_descriptor = face_rec_model_(image, shape1)
    face_descriptor = face_rec_model_.compute_face_descriptor(image, shape0)
    IS_RECOGNIZE, name, dist = recognize_face(face_descriptor, CSV_FILE, DISTANCE_THRESHOLD)

    if IS_PEOPLE and IS_VALIDATE and IS_RECOGNIZE:
        return True, name, dist
    else:
        return False, None, None

def recognize_from_2_frame_picture(ir_image, depth_image, 
                                   DEPTH_SCALE, DISTANCE_THRESHOLD, 
                                   PREDICTOR_FILE, REC_MODEL_FILE, 
                                   CSV_FILE):
    detector_ = dlib.get_frontal_face_detector()
    face = detector_(ir_image, 1)
    if len(face) == 0:
        print("len(face) == 0")
        return False, None, None
    IS_PEOPLE = False
    IS_VALIDATE = False
    IS_RECOGNIZE = False

    predictor_ = dlib.shape_predictor(PREDICTOR_FILE)
    shape = predictor_(ir_image, face[0])
    if shape.num_parts != 68:
        print("shape.num_parts != 68")
        return False, None, None
    else:
        IS_PEOPLE = True

    if not validate_face.validate_face(depth_image, DEPTH_SCALE, shape):
        print("not validate_face.validate_face(depth_image, DEPTH_SCALE, shape)")
        return False, None, None
    else:
        IS_VALIDATE = True

    face_rec_model_ = dlib.face_recognition_model_v1(REC_MODEL_FILE)
    image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
    face_descriptor = face_rec_model_.compute_face_descriptor(image, shape)
    IS_RECOGNIZE, name, dist = recognize_face(face_descriptor, CSV_FILE, DISTANCE_THRESHOLD)

    if IS_PEOPLE and IS_VALIDATE and IS_RECOGNIZE:
        return True, name, dist
    else:
        return False, None, None

update_database(faces_folder)

# pipeline = rs.pipeline()
# config = rs.config()
# # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
# profile = pipeline.start(config)
# # get the depth sensor's depth scale
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_sensor.set_option(rs.option.emitter_enabled, 0)
# # pipeline.start(config)

# # Load FACES_FEATURES_CSV_FILE to database
# database = []
# with open(FACES_FEATURES_CSV_FILE, "r") as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         database.append((row[0], row[1:]))

# win = dlib.image_window()
# while not win.is_closed():
#     frames = pipeline.wait_for_frames()
#     # color_frame = frames.get_color_frame()
#     color_frame = frames.get_infrared_frame(1)
#     if not color_frame:
#         continue
#     image = np.asanyarray(color_frame.get_data())
#     win.clear_overlay()
#     win.set_image(image)

#     faces = detector(image, 1)
#     person_dist = 0
#     person_label = ""
#     flag = 0
#     if len(faces) != 0:
#         for i in range(len(faces)):
#             face = faces[i]
#             shape = predictor(image, faces[0])
#             # 将image转换成RGB格式
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

#     win.add_overlay(faces)
