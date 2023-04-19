import pyrealsense2 as rs
import cv2
import numpy as np
import dlib
import validate_face
import recognize_face

predictor_path = '/home/haoran/GitHub/dlib-anti-spoof/data/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = '/home/haoran/GitHub/dlib-anti-spoof/data/dlib_face_recognition_resnet_model_v1.dat'
faces_folder = '/home/haoran/GitHub/dlib-anti-spoof/face'
FACES_FEATURES_CSV_FILE = '/home/haoran/GitHub/dlib-anti-spoof/data/face_features.csv'

FACES_FATURES_DISTANCE_THRESHOLD = 0.3
IS_DISPLAY = True
IS_RECOGNIZE = False
IS_CONTINUOUS = False
RECOGNIZED_TIMES = 4

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)
# get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
DEPTH_SCALE = depth_sensor.get_depth_scale()

align_to_color = rs.align(rs.stream.color)

if __name__ == '__main__':
    cnt = 0
    past_name = ''
    current_name = ''
    final_name = ''
    if IS_DISPLAY:
        win = dlib.image_window()
        while not win.is_closed():
            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not IS_RECOGNIZE:
                if not depth_frame or not color_frame:
                    IS_CONTINUOUS = False
                    IS_RECOGNIZE = False
                    cnt = 0
                    past_name = ''
                    current_name = ''
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                dets = detector(color_image, 1)
                if len(dets) == 0:
                    IS_RECOGNIZE = False
                    IS_CONTINUOUS = False
                    cnt = 0
                    past_name = ''
                    current_name = ''
                    continue
                else:
                    shape = predictor(color_image, dets[0])
                    if not validate_face.validate_face(depth_frame, DEPTH_SCALE, shape):
                        IS_RECOGNIZE = False
                        IS_CONTINUOUS = False
                        cnt = 0
                        past_name = ''
                        current_name = ''
                        continue
                    else:
                        face_descriptor = facerec.compute_face_descriptor(color_image, shape)
                        recog, temp_name, temp_distance = recognize_face.recognize_face(face_descriptor, FACES_FEATURES_CSV_FILE, FACES_FATURES_DISTANCE_THRESHOLD)
                        if not recog:
                            IS_RECOGNIZE = False
                            IS_CONTINUOUS = False
                            cnt = 0
                            past_name = ''
                            current_name = ''
                            continue
                        else:
                            if not IS_CONTINUOUS:
                                past_name = temp_name
                                current_name = temp_name
                                cnt = 1
                                IS_CONTINUOUS = True
                                IS_RECOGNIZE = False
                            else:
                                current_name = temp_name
                                if past_name == current_name:
                                    cnt += 1
                                    if cnt >= RECOGNIZED_TIMES:
                                        final_name = current_name
                                        IS_RECOGNIZE = True
                                else:
                                    past_name = current_name
                                    cnt = 0
                                    IS_RECOGNIZE = False
            else:
                print("Recognized: " + final_name)
                final_name = ''
                IS_RECOGNIZE = False
                IS_CONTINUOUS = False
                cnt = 0
                past_name = ''
                current_name = ''
            win.clear_overlay()
            win.set_image(color_image)
            win.add_overlay(dets[0])
            win.add_overlay(shape)
     
