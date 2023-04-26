import time
import pyrealsense2 as rs
import cv2
import numpy as np
import dlib
import dlib_recognize_face
import validate_face
import recognize_face

detect_path = './model/mmod_human_face_detector.dat'
predictor_path = './model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = './model/dlib_face_recognition_resnet_model_v1.dat'
faces_folder = './face'
FACES_FEATURES_CSV_FILE = './data/face_features.csv'

FACES_FATURES_DISTANCE_THRESHOLD = 0.35
# 0: Display, 1: Validate, 2: Recognize, 3: ALL
IS_TEST = 3
IS_PEOPLE = False
IS_VALIDATE = False
IS_RECOGNIZE = False
IS_CONTINUOUS = False
RECOGNIZED_TIMES = 4

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

profile = pipeline.start(config)
# get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.emitter_enabled, 0)
DEPTH_SCALE = depth_sensor.get_depth_scale()

align_to_color = rs.align(rs.stream.color)

shape = 0
dets = 0

if __name__ == '__main__':
    # Display Version
    DRecFace = dlib_recognize_face.Recognize_Face(detect_path, predictor_path, face_rec_model_path, FACES_FEATURES_CSV_FILE)
    if IS_TEST == 0:
        win = dlib.image_window()
        while not win.is_closed():
            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame(1)
            if not depth_frame or not ir_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())
            win.clear_overlay()
            win.set_image(ir_image)
            dets = detector(ir_image, 1)
            if len(dets) == 0:
                IS_PEOPLE = False
                IS_VALIDATE = False
                IS_RECOGNIZE = False
                depth_sensor.set_option(rs.option.emitter_enabled, 0)
                continue
            elif len(dets) > 0 or IS_PEOPLE:
                if not IS_VALIDATE:
                    if not IS_PEOPLE:
                        shape = predictor(ir_image, dets[0])
                        if shape.num_parts == 68:
                            IS_PEOPLE = True
                            depth_sensor.set_option(rs.option.emitter_enabled, 1)
                            print("People detected")
                        else:
                            IS_PEOPLE = False
                            shape = 0
                            depth_sensor.set_option(rs.option.emitter_enabled, 0)
                    else:
                        if DRecFace.validate_face(depth_image, 0.001, shape):
                            IS_VALIDATE = True
                            print("People validated")
                        else:
                            IS_VALIDATE = False
                        IS_PEOPLE = False
                        depth_sensor.set_option(rs.option.emitter_enabled, 0)
                else:
                    shape = predictor(ir_image, dets[0])
                    image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                    face_descriptor = facerec.compute_face_descriptor(image, shape)
                    IS_RECOGNIZE, name, dist = recognize_face.recognize_face(face_descriptor, FACES_FEATURES_CSV_FILE, FACES_FATURES_DISTANCE_THRESHOLD)
                    if IS_RECOGNIZE:
                        print("Recognized: " + name + " Distance: " + str(dist))
                    else:
                        print("Not Recognized")
                    IS_VALIDATE = False
                    IS_PEOPLE = False
                    IS_RECOGNIZE = False
                    temp_shape = 0
                    depth_sensor.set_option(rs.option.emitter_enabled, 0)

    elif IS_TEST == 1:
        win = dlib.image_window()
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                depth_sensor.set_option(rs.option.emitter_enabled, 0)
                continue
            ir_image = np.asanyarray(ir_frame.get_data())
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)
            ir_frame = frames.get_infrared_frame(1)
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame:
                depth_sensor.set_option(rs.option.emitter_enabled, 0)
                continue
            depth_image = np.asanyarray(depth_frame.get_data())

            dets = detector(ir_image, 1)
            if len(dets) == 0:
                depth_sensor.set_option(rs.option.emitter_enabled, 0)
                continue
            else:
                shape = predictor(ir_image, dets[0])
                if shape.num_parts == 68:
                    if DRecFace.validate_face(depth_image, 0.001, shape):
                        print("People validated")
                    else:
                        print("People not validated")
                else:
                    print("People not detected")
                depth_sensor.set_option(rs.option.emitter_enabled, 0)
            win.clear_overlay()
            win.set_image(ir_image)
            time.sleep(1)

    elif IS_TEST == 2:
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        win = dlib.image_window()
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                continue
            ir_image = np.asanyarray(ir_frame.get_data())
            dets = detector(ir_image, 1)
            if len(dets) == 0:
                continue
            else:
                shape = predictor(ir_image, dets[0])
                image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                face_descriptor = facerec.compute_face_descriptor(image, shape)
                IS_RECOGNIZE, name, dist = DRecFace.recognize_face(face_descriptor, 0.35)
                if IS_RECOGNIZE:
                    print("Recognized: " + name + " Distance: " + str(dist))
                else:
                    print("Not Recognized")
            win.clear_overlay()
            win.set_image(ir_image)


    # Use API Version
    elif IS_TEST == 3:
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
        win = dlib.image_window()
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                depth_sensor.set_option(rs.option.emitter_enabled, 0)
                continue
            ir_image = np.asanyarray(ir_frame.get_data())
            win.clear_overlay()
            win.set_image(ir_image)
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame:
                depth_sensor.set_option(rs.option.emitter_enabled, 0)
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            image = ir_image
            # image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            is_recognized, name, dist = DRecFace.recognize_from_2_frame(image, depth_image, 0.35, True)
            if is_recognized == 1:
                print("Recognized: " + name + " Distance: " + str(dist))
            # else:
            #     print("Not Recognized")
            
            # Delay 150ms
            
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
            time.sleep(0.2)
            
            # # 获取第3帧数据
            # depth_sensor.set_option(rs.option.emitter_enabled, 0)
            # frames = pipeline.wait_for_frames()
            # aligned_frames = align_to_color.process(frames)
            # ir_frame = frames.get_infrared_frame(1)
            # if not ir_frame:
            #     depth_sensor.set_option(rs.option.emitter_enabled, 0)
            #     continue
            # ir_image = np.asanyarray(ir_frame.get_data())
            # if len(dets) == 0:
            #     depth_sensor.set_option(rs.option.emitter_enabled, 0)
            #     continue
            # ir_image_batch.append(ir_image)

            # # 人脸检测
            # IS_RECOGNIZE, name, dist = recognize_face.recognize_from_3_frame_picture(ir_image_batch, depth_image, 
            #                                                                          DEPTH_SCALE, FACES_FATURES_DISTANCE_THRESHOLD, 
            #                                                                          predictor_path, face_rec_model_path, 
            #                                                                          FACES_FEATURES_CSV_FILE)
            # if IS_RECOGNIZE:
            #     print("Recognized: " + name + " Distance: " + str(dist))

