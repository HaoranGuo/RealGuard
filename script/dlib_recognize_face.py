import dlib
import numpy as np
import csv
from PIL import Image
import cv2
import os
import time

class Recognize_Face:
    def __init__(self, CNN_DETECT_MODEL, LANDMARKS_MODEL, ROCOGNITION_RESNET_MODEL, FACES_FEATURES_CSV_FILE):
        self.detector = dlib.cnn_face_detection_model_v1(CNN_DETECT_MODEL)
        self.predictor = dlib.shape_predictor(LANDMARKS_MODEL)
        self.facerec = dlib.face_recognition_model_v1(ROCOGNITION_RESNET_MODEL)
        self.FACES_FEATURES_CSV_FILE = FACES_FEATURES_CSV_FILE
        self.DEPTH_SCALE = 0.001

        # 分割出csv文件的路径和文件名
        csv_path, csv_name = os.path.split(self.FACES_FEATURES_CSV_FILE)
        # 如果csv所在的目录不存在，创建目录
        if not os.path.isdir(csv_path):
            os.makedirs(csv_path)


    def find_depth_from(self, depth_image, depth_scale, face, markup_from, markup_to):
        data1 = np.asanyarray(depth_image)

        average_depth = 0
        n_points = 0
        for i in range(markup_from, markup_to + 1):
            point = face.part(i)

            data1 = data1.reshape(-1)
            indexx = point.x
            indexy = point.y

            # Get the size of the depth image
            # col = depth_image.shape[1]
            # row = depth_image.shape[0]
            # print("col = ", depth_image.shape[1])
            # print("row = ", depth_image.shape[0])

            col = 640
            row = 480

            if indexx < 0:
                indexx = 0
            if indexy < 0:
                indexy = 0
            if indexx > col:
                indexx = col
            if indexy > row:
                indexy = row
            indexk = indexy * col + indexx
            if indexk >= len(data1):
                indexk = len(data1)-1
            depth_in_pixels = data1[indexk]
            if not depth_in_pixels:
                continue
            average_depth += depth_in_pixels * depth_scale
            n_points += 1
        if not n_points:
            return False, 0
        p_average_depth = average_depth / n_points
        return True, p_average_depth
    
    
    def validate_face(self, depth_image, depth_scale, face):
        # Collect all the depth information for the different facial parts
        # For the ears, only one may be visible -- we take the closer one!
        # left_ear_depth = 100
        # right_ear_depth = 100
        # flag, right_ear_depth = self.find_depth_from(depth_image, depth_scale, face, 0, 1)
        # flag0, left_ear_depth = self.find_depth_from(depth_image, depth_scale, face, 15, 16)
        # if not flag and not flag0:
        #     return False
        # ear_depth = left_ear_depth
        # if right_ear_depth != 0 and left_ear_depth != 0:
        #     if right_ear_depth < left_ear_depth:
        #         ear_depth = right_ear_depth
        #     else:
        #         ear_depth = left_ear_depth
        # elif right_ear_depth == 0:
        #     ear_depth = left_ear_depth
        # elif left_ear_depth == 0:
        #     ear_depth = right_ear_depth

        # chin_depth = 0
        # flag, chin_depth = self.find_depth_from(depth_image, depth_scale, face, 7, 9)
        # if not flag:
        #     return False
        eyebrow_depth = 0
        flag, left_eyebrow_depth = self.find_depth_from(depth_image, depth_scale, face, 17, 21)
        flag0, right_eyebrow_depth = self.find_depth_from(depth_image, depth_scale, face, 22, 26)
        if not flag and not flag0:
            return False
        if left_eyebrow_depth != 0 and right_eyebrow_depth != 0:
            if left_eyebrow_depth < right_eyebrow_depth:
                eyebrow_depth = left_eyebrow_depth
            else:
                eyebrow_depth = right_eyebrow_depth
        elif left_eyebrow_depth == 0:
            eyebrow_depth = right_eyebrow_depth
        elif right_eyebrow_depth == 0:
            eyebrow_depth = left_eyebrow_depth
        

        nose_depth = 0
        flag, nose_depth = self.find_depth_from(depth_image, depth_scale, face, 29, 30)
        if not flag:
            return False

        right_eye_depth = 0
        flag, right_eye_depth = self.find_depth_from(depth_image, depth_scale, face, 36, 41)
        if not flag:
            return False
        left_eye_depth = 0
        flag, left_eye_depth = self.find_depth_from(depth_image, depth_scale, face, 42, 47)
        if not flag:
            return False
        eye_depth = 0
        if left_eye_depth != 0 and right_eye_depth != 0:
            if left_eye_depth < right_eye_depth:
                eye_depth = left_eye_depth
            else:
                eye_depth = right_eye_depth
        elif left_eye_depth == 0:
            eye_depth = right_eye_depth
        elif right_eye_depth == 0:
            eye_depth = left_eye_depth

        mouth_depth = 0
        flag, mouth_depth = self.find_depth_from(depth_image, depth_scale, face, 48, 67)
        if not flag:
            return False
        
        # print(right_ear_depth, left_ear_depth, ear_depth, chin_depth, nose_depth, right_eye_depth, left_eye_depth, eye_depth, mouth_depth)

        # We just use simple heuristics to determine whether the depth information agrees with
        # what's expected: that the nose tip, for example, should be closer to the camera than
        # the eyes.
        # These heuristics are fairly basic but nonetheless serve to illustrate the point that
        # depth data can effectively be used to distinguish between a person and a picture of a
        # person...
        if nose_depth >= eyebrow_depth:
            print("nose_depth >= eyebrow_depth")
            return False
        if eye_depth - nose_depth > 0.08:
            print("eye_depth - nose_depth > 0.08")
            return False
        # if eye_depth - nose_depth < 0.02:
        #     print("eye_depth - nose_depth < 0.02")
        #     return False
        # if ear_depth <= eye_depth:
        #     print("ear_depth <= eye_depth")
        #     return False
        # if mouth_depth <= nose_depth:
        #     print("mouth_depth <= nose_depth")
        #     return False
        if mouth_depth > eye_depth:
            print("mouth_depth > eye_depth")
            return False
        # if mouth_depth - nose_depth < 0.02:
        #     print("mouth_depth - nose_depth < 0.02")
        #     return False
        
        # All the distances, collectively, should not span a range that makes no sense. I.e.,
        # if the face accounts for more than 20cm of depth, or less than 2cm, then something's
        # not kosher!

        # x = max([nose_depth, mouth_depth, chin_depth, eye_depth, ear_depth])
        # n = min([nose_depth, mouth_depth, chin_depth, eye_depth, ear_depth])
        x = max([nose_depth, mouth_depth, eye_depth, eyebrow_depth])
        n = min([nose_depth, mouth_depth, eye_depth, eyebrow_depth])
        if x - n > 0.20:
            print("x - n > 0.20")
            return False
        if x - n < 0.01:
            print("x - n < 0.01")
            return False
        
        return True


    def get_euclidean_distance(self, feature_1, feature_2):
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


    def recognize_face_outside(self, face_descriptor, DISTANCE_THRESHOLD, csv_file):
        database = []
        name = ""
        dist = 0
        with open(csv_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                database.append((row[0], row[6:]))
        
        face_feature_distance_list = []
        for face in database:
            dist = self.get_euclidean_distance(face_descriptor, face[1])
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
        
    
    def recognize_face(self, face_descriptor, DISTANCE_THRESHOLD):
        IS_RECOGNIZED, name, dist = self.recognize_face_outside(face_descriptor, DISTANCE_THRESHOLD, self.FACES_FEATURES_CSV_FILE)
        return IS_RECOGNIZED, name, dist
    

    def recognize_from_2_frame(self, frame_2d, frame_3d, DISTANCE_THRESHOLD, is_gray = False):
        IS_PEOPLE = False
        IS_VALIDATE = False
        IS_RECOGNIZE = False
        
        face = self.detector(frame_2d, 1)
        if len(face) == 0:
            print("No face detected!")
            return -1, None, None
        # Convert mmod_rectangle to dlib.rectangle
        d = face[0]
        face = dlib.rectangle(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
        
        shape = self.predictor(frame_2d, face)

        if shape.num_parts != 68:
            print("No face detected! (num_parts != 68)")
            return -1, None, None
        else:
            IS_PEOPLE = True

        IS_VALIDATE = self.validate_face(frame_3d, 0.001, shape)
        if not IS_VALIDATE:
            print("No face detected! (validate_face)")
            return -2, None, None

        # Convert Gray to RGB Using PIL
        if is_gray:
            # image = Image.fromarray(frame_2d)
            # image = image.convert('RGB')
            image = cv2.cvtColor(frame_2d, cv2.COLOR_GRAY2BGR)
            image = np.array(image)
        else:
            image = frame_2d

        face_descriptor = self.facerec.compute_face_descriptor(image, shape)
        IS_RECOGNIZE, name, dist = self.recognize_face(face_descriptor, DISTANCE_THRESHOLD)

        if IS_PEOPLE and IS_VALIDATE and IS_RECOGNIZE:
            return 1, name, dist
        else:
            print("No face detected! (IS_PEOPLE and IS_VALIDATE and IS_RECOGNIZE")
            return -3, None, None
        

    def recognize_and_continue_learning(self, frame_2d, frame_3d, DISTANCE_THRESHOLD, LEARNING_THRESHOLD, is_gray = False):
        IS_PEOPLE = False
        IS_VALIDATE = False
        IS_RECOGNIZE = False
        
        face = self.detector(frame_2d, 1)
        if len(face) == 0:
            print("No face detected!")
            return -1, None, None
        # Convert mmod_rectangle to dlib.rectangle
        d = face[0]
        face = dlib.rectangle(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
        
        shape = self.predictor(frame_2d, face)

        if shape.num_parts != 68:
            print("No face detected! (num_parts != 68)")
            return -1, None, None
        else:
            IS_PEOPLE = True

        IS_VALIDATE = self.validate_face(frame_3d, 0.001, shape)
        if not IS_VALIDATE:
            print("No face detected! (validate_face)")
            return -2, None, None

        # Convert Gray to RGB Using PIL
        if is_gray:
            # image = Image.fromarray(frame_2d)
            # image = image.convert('RGB')
            image = cv2.cvtColor(frame_2d, cv2.COLOR_GRAY2BGR)
            image = np.array(image)
        else:
            image = frame_2d

        face_descriptor = self.facerec.compute_face_descriptor(image, shape)
        IS_RECOGNIZE, name, dist = self.recognize_face(face_descriptor, DISTANCE_THRESHOLD)

        if IS_PEOPLE and IS_VALIDATE and IS_RECOGNIZE:
            if dist <= LEARNING_THRESHOLD:
                with open(self.FACES_FEATURES_CSV_FILE, "r", newline="") as csvfile:
                    reader = csv.reader(csvfile)
                    person_names = []
                    feature_arrays = []
                    for row in reader:
                        person_names.append(row[0])
                        feature_arrays.append(row[1:])
                    weight = (1 - dist) / 15
                    for i in range(len(person_names)):
                        if person_names[i] == name:
                            original_feature_array = feature_arrays[i]
                            original_feature_array = np.array(original_feature_array, dtype=np.float64)
                            original_feature_array = original_feature_array * (1 - weight) + np.array(face_descriptor) * weight
                            # 修改csv文件中姓名为name的特征
                            feature_arrays[i] = original_feature_array.tolist()
                            break
                    # 将修改后的特征写入csv文件
                    with open(self.FACES_FEATURES_CSV_FILE, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        for i in range(len(person_names)):
                            writer.writerow([person_names[i]] + feature_arrays[i])
                    
                    csvfile.close()
            return 1, name, dist    
        else:
            print("No face detected! (IS_PEOPLE and IS_VALIDATE and IS_RECOGNIZE")
            return -3, None, None

        
    def read_image(self, IMAGE_PATH):
        # Use dlib to load the image as a numpy array
        image = dlib.load_rgb_image(IMAGE_PATH)
        image = np.array(image)
        return image


class Extract_Face_Feature:
    def __init__(self, LANDMARKS_MODEL, ROCOGNITION_RESNET_MODEL, FACES_FOLDER, FACES_FEATURES_CSV_FILE):
        self.FACES_FOLDER = FACES_FOLDER
        self.FACES_FEATURES_CSV_FILE = FACES_FEATURES_CSV_FILE
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(LANDMARKS_MODEL)
        self.facerec = dlib.face_recognition_model_v1(ROCOGNITION_RESNET_MODEL)
        if not os.path.isdir(self.FACES_FOLDER):
            os.makedirs(self.FACES_FOLDER)
        csv_path, csv_name = os.path.split(self.FACES_FEATURES_CSV_FILE)
        # 如果csv所在的目录不存在，创建目录
        if not os.path.isdir(csv_path):
            os.makedirs(csv_path)


    def get_128d_features_of_face(self, image_path):
        # image = dlib.load_rgb_image(image_path)
        # image = Image.open(image_path)
        image = cv2.imread(image_path)
        image = np.array(image)
        faces = self.detector(image, 1)

        if len(faces) != 0:
            shape = self.predictor(image, faces[0])
            face_descriptor = self.facerec.compute_face_descriptor(image, shape)
        else:
            face_descriptor = 0
        return face_descriptor


    def get_mean_features_of_face(self, path):
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
                feature = self.get_128d_features_of_face(image_path)
                if feature == 0:
                    continue
                
                feature_list_of_person_x.append(feature)
            
            # 计算当前人脸的平均特征
            features_mean_person_x = np.zeros(128, dtype=object, order='C')
            if feature_list_of_person_x:
                features_mean_person_x = np.array(feature_list_of_person_x, dtype=object).mean(axis=0)
            
            yield (features_mean_person_x, person_label)


    def extract_features_to_csv(self, faces_dir, FACES_FEATURES_CSV_FILE):
        mean_features_list = list(self.get_mean_features_of_face(faces_dir))
        with open(FACES_FEATURES_CSV_FILE, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for mean_features in mean_features_list:
                person_features = mean_features[0]
                person_label = mean_features[1]
                person_features = np.insert(person_features, 0, person_label, axis=0)
                writer.writerow(person_features)

    def extract_features(self):
        self.extract_features_to_csv(self.FACES_FOLDER, self.FACES_FEATURES_CSV_FILE)


    def add_face_features(self, FACES_FEATURES_CSV_FILE, path):
        path = os.path.abspath(path)
        for f in os.listdir(path):
            subDir = os.path.join(path, f)
            if not os.path.isdir(subDir):
                continue
            # 读取FACES_FEATURES_CSV_FILE中的人名，如果已经存在，则不再添加
            with open(FACES_FEATURES_CSV_FILE, "r", newline="") as csvfile:
                reader = csv.reader(csvfile)
                person_names = [row[0] for row in reader]
            if f in person_names:
                continue
            else:
                person_label = os.path.split(subDir)[-1]
                image_paths = [os.path.join(subDir, f) for f in os.listdir(subDir)]
                image_paths = list(filter(lambda x:os.path.isfile(x), image_paths))
                feature_list_of_person_x = []
                for image_path in image_paths:
                    # 计算每一个图片的特征
                    feature = self.get_128d_features_of_face(image_path)
                    if feature == 0:
                        continue
                    
                    feature_list_of_person_x.append(feature)

                # 计算当前人脸的平均特征
                features_mean_person_x = np.zeros(128, dtype=object, order='C')
                if feature_list_of_person_x:
                    features_mean_person_x = np.array(feature_list_of_person_x, dtype=object).mean(axis=0)

                with open(FACES_FEATURES_CSV_FILE, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    person_features = features_mean_person_x
                    person_label = person_label
                    person_features = np.insert(person_features, 0, person_label, axis=0)
                    writer.writerow(person_features)

    def add_face(self):
        self.add_face_features(self.FACES_FEATURES_CSV_FILE, self.FACES_FOLDER)


class Register_Face:
    def __init__(self, CNN_DETECT_MODEL, LANDMARKS_MODEL, ROCOGNITION_RESNET_MODEL, FACES_FEATURES_CSV_FILE, FACES_FOLDER):
        self.detector = dlib.cnn_face_detection_model_v1(CNN_DETECT_MODEL)
        self.predictor = dlib.shape_predictor(LANDMARKS_MODEL)
        self.facerec = dlib.face_recognition_model_v1(ROCOGNITION_RESNET_MODEL)
        self.FACES_FEATURES_CSV_FILE = FACES_FEATURES_CSV_FILE
        self.FACES_FOLDER = FACES_FOLDER
        if not os.path.isdir(self.FACES_FOLDER):
            os.makedirs(self.FACES_FOLDER)
    
    def save_image(self, image, name):
        abs_path = os.path.abspath(self.FACES_FOLDER)
        image_path = os.path.join(abs_path, name)
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        current_time = time.time()
        image_name = str(current_time) + ".jpg"
        image_path = os.path.join(image_path, image_name)
        cv2.imwrite(image_path, image)
    
    def get_128d_features_of_face(self, image):
        image = np.array(image)
        faces = self.detector(image, 1)

        if len(faces) != 0:
            shape = self.predictor(image, faces[0].rect)
            face_descriptor = self.facerec.compute_face_descriptor(image, shape)
        else:
            face_descriptor = 0
        return face_descriptor

    def get_euclidean_distance(self, feature_1, feature_2):
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
    
    def register_face(self, image, name, id):
        is_new = False
        recognized_time = ''
        pic_cnt = 0
        recognized_times = 0
        feature = 0
        rt_arr = []
        rts_arr = []
        add_arr = []
        name_arr = []
        id_arr = []
        pic_cnt_arr = []
        feature_arr = []
        with open(self.FACES_FEATURES_CSV_FILE, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == name:
                    print(name, " is already in the csv file.")
                    row_id = row[1]
                    if row_id != id:
                        print("The id is different from the csv file.")
                        csvfile.close()
                        return False, False, -1
                    added_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    is_new = False
                    break
            else:
                is_new = True
                print(name, " is not in the csv file.")
                row_id = id
                recognized_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                added_time = recognized_time
                pic_cnt = 1
                recognized_times = 0
                csvfile.close()

        if is_new:
            with open(self.FACES_FEATURES_CSV_FILE, "a", newline="") as csvfile:
                feature = self.get_128d_features_of_face(image)
                if feature == 0:
                    print("No face detected!")
                    csvfile.close()
                    return False, False, -1
                writer = csv.writer(csvfile)
                # feature_str = ''
                # for i in range(len(feature) -1):
                #     feature_str += str(feature[i]) + ','
                # feature_str += str(feature[-1])
                # person_info = str(name) + ',' + str(row_id) + ',' + added_time + ',' + recognized_time + ',' + str(pic_cnt) + ',' + str(recognized_times)
                person_info = [name, row_id, added_time, recognized_time, pic_cnt, recognized_times]
                # feature = str(feature)
                feature = np.array(feature, dtype=object)
                person_features = np.insert(feature, 0, person_info, axis=0)
                # person_features = person_info + ',' + feature_str
                writer.writerow(person_features)
                print("Add " + name + " successfully.")
                csvfile.close()
            self.save_image(image, name)
            return True, True, 0
        else:
            feature = self.get_128d_features_of_face(image)
            if feature == 0:
                print("No face detected!")
                return False, False, -1
            with open(self.FACES_FEATURES_CSV_FILE, "r", newline="") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    name_arr.append(row[0])
                    id_arr.append(row[1])
                    add_arr.append(row[2])
                    rt_arr.append(row[3])
                    pic_cnt_arr.append(row[4])
                    rts_arr.append(row[5])
                    feature_arr.append(row[6:])
                    old_feature = np.array(row[6:], dtype=np.float64)
                    if row[0] == name:
                        dist = self.get_euclidean_distance(feature, old_feature)
                        if dist > 0.42:
                            csvfile.close()
                            return False, False, dist
            with open(self.FACES_FEATURES_CSV_FILE, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for i in range(len(name_arr)):
                    if name_arr[i] == name:
                        pic_cnt = int(pic_cnt_arr[i])
                        old_feature = np.array(feature_arr[i], dtype=np.float64)
                        
                        added_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        # 两个feature加权平均
                        new_feature = (feature + old_feature * pic_cnt) / (pic_cnt + 1)
                        pic_cnt += 1
                        person_info = [name_arr[i], id_arr[i], added_time, rt_arr[i], str(pic_cnt), rts_arr[i]]
                        # new_feature = str(new_feature)
                        new_feature = np.array(new_feature, dtype=object)
                        person_features = np.insert(new_feature, 0, person_info, axis=0)
                        writer.writerow(person_features)
                    else:
                        person_info = [name_arr[i], id_arr[i], add_arr[i], rt_arr[i], pic_cnt_arr[i], rts_arr[i]]
                        person_features = np.insert(feature_arr[i], 0, person_info, axis=0)
                        writer.writerow(person_features)
                print("Update " + name + " successfully.")
                csvfile.close()
            self.save_image(image, name)
            return True, False, dist