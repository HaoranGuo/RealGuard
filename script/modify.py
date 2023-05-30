import argparse
import dlib
import csv
import os
import time
import cv2
import numpy as np

detect_path = './model/mmod_human_face_detector.dat'
predictor_path = './model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = './model/dlib_face_recognition_resnet_model_v1.dat'
faces_folder = './face'
FACES_FEATURES_CSV_FILE = './data/face_features.csv'

parser = argparse.ArgumentParser(description='Modify the csv file, which saved the face features and personal information.')
parser.add_argument('-l', '--list', action='store_true', help='List all the people, ID and added time and latest recognized time.')
parser.add_argument('-d', '--delete', type=str, default='Unknown', help='Using name to find and delete the person from the csv file.')
parser.add_argument('-f', '--find', type=str, default='Unknown', help='Using name to figure out whether the person was in the csv file.')
parser.add_argument('-a', '--add', type=str, default='Unknown', help='Add a new person to the csv file.')
parser.add_argument('-ll', '--listlatest', action='store_true', help='List the latest recognized time of the person.')

args = parser.parse_args()

def get_128d_features_of_face(image_path):
    # image = dlib.load_rgb_image(image_path)
    # image = Image.open(image_path)
    detector = dlib.cnn_face_detection_model_v1(detect_path)
    predictor = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    image = cv2.imread(image_path)
    image = np.array(image)
    faces = detector(image, 1)

    if len(faces) != 0:
        shape = predictor(image, faces[0])
        face_descriptor = facerec.compute_face_descriptor(image, shape)
    else:
        face_descriptor = 0
    return face_descriptor

if args.list:
    print("name, id, added time, latest recognized time")
    with open(FACES_FEATURES_CSV_FILE, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row[0] + " " + row[1] + " " + row[2] + " " + row[3])
        csvfile.close()

if args.delete != 'Unknown':
    delete_name = args.delete
    reverification = input("Are you sure to delete " + delete_name + "? (y/[n])")
    if reverification == 'y' or reverification == 'Y':
        with open(FACES_FEATURES_CSV_FILE, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            lines = [row for row in reader]
            for row in lines:
                if row[0] == delete_name:
                    print("Found " + delete_name + " in the csv file.")
                    break
            else:
                print("Not found " + delete_name + " in the csv file.")
                csvfile.close()
                exit(0)
            csvfile.close()
        with open(FACES_FEATURES_CSV_FILE, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for row in lines:
                if row[0] != delete_name:
                    writer.writerow(row)
                else:
                    print("Delete " + delete_name + " successfully.")
            csvfile.close()

if args.find != 'Unknown':
    find_name = args.find
    with open(FACES_FEATURES_CSV_FILE, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == find_name:
                print("Found " + find_name + " in the csv file.")
                print("name, id, added time, latest recognized time, trained times")
                print(row[0] + " " + row[1] + " " + row[2] + " " + row[3] + " " + row[4])
                break
        else:
            print("Not found " + find_name + " in the csv file.")
        csvfile.close()

if args.add != 'Unknown':
    add_name = args.add
    path = os.path.abspath(faces_folder)
    print(path)
    for f in os.listdir(path):
        if f == add_name:
            print("Found " + add_name + " in the face folder.")
            break
    else:
        print("Not found " + add_name + " in the face folder.")
        exit(0)

    with open(FACES_FEATURES_CSV_FILE, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == add_name:
                print(add_name, " is already in the csv file.")
                csvfile.close()
                exit(0)
        else:
            id = input("Please input the id of " + add_name + ": ")
            image_path = os.path.join(path, add_name)
            image_paths = [os.path.join(image_path, f) for f in os.listdir(image_path)]
            image_paths = list(filter(lambda x:os.path.isfile(x), image_paths))
            print(path)
            print(image_path)
            print(image_paths)
            pic_cnt = 0
            feature_list_of_person_x = []
            detector = dlib.cnn_face_detection_model_v1(detect_path)
            predictor = dlib.shape_predictor(predictor_path)
            facerec = dlib.face_recognition_model_v1(face_rec_model_path)
            for image in image_paths:
                image = cv2.imread(image)
                image = np.array(image)
                faces = detector(image, 1)

                if len(faces) != 0:
                    shape = predictor(image, faces[0].rect)
                    if shape == 0:
                        continue
                    else:
                        face_descriptor = facerec.compute_face_descriptor(image, shape)
                        pic_cnt += 1
                else:
                    continue
                feature = face_descriptor
                if feature == 0:
                    continue
                
                feature_list_of_person_x.append(feature)
            features_mean_person_x = np.zeros(128, dtype=object, order='C')
            if feature_list_of_person_x:
                features_mean_person_x = np.array(feature_list_of_person_x, dtype=object).mean(axis=0)
            csvfile.close()
    with open(FACES_FEATURES_CSV_FILE, "a", newline="") as csvfile:
        added_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        writer = csv.writer(csvfile)
        person_info = [add_name, id, added_time, added_time, pic_cnt, 0]
        person_features = np.insert(features_mean_person_x, 0, person_info, axis=0)
        writer.writerow(person_features)
        print("Add " + add_name + " successfully.")
        csvfile.close()

if args.listlatest:
    # 输出整个csv文件中最新识别时间和相对应的人名id
    with open(FACES_FEATURES_CSV_FILE, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        latest_time = 0
        latest_name = ''
        latest_id = ''
        for row in reader:
            time_str = row[3]
            timeArray = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            timeStamp = int(time.mktime(timeArray))
            if timeStamp > latest_time:
                latest_time = timeStamp
                latest_name = row[0]
                latest_id = row[1]
        print("The latest recognized time is " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_time)) + " of " + latest_name + " " + latest_id)
        csvfile.close()
