# Use dlib and realsense to judge whether the face is spoofing or not
import dlib
import numpy as np
import pyrealsense2 as rs

# Calculates the average depth for a range of two-dimentional points in face, such that:
#     point(n) = face.part(n)
# and puts the result in *p_average_depth.

# Points for which no depth is available (is 0) are ignored and not factored into the average.

# Returns true if an average is available (at least one point has depth); false otherwise.
def find_depth_from(depth_image, depth_scale, face, markup_from, markup_to):
    data1 = np.asanyarray(depth_image)

    average_depth = 0
    n_points = 0
    for i in range(markup_from, markup_to + 1):
        point = face.part(i)

        data1 = data1.reshape(-1)
        indexx = point.x
        indexy = point.y

        # Get the size of the depth image
        col = depth_image.shape[1]
        row = depth_image.shape[0]

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

# Returns whether the given 68-point facial landmarks denote the face of a real
# person (and not a picture of one), using the depth data in depth_frame.

# See markup_68 for an explanation of the point topology.
def validate_face(depth_image, depth_scale, face):
    # Collect all the depth information for the different facial parts
    # For the ears, only one may be visible -- we take the closer one!
    left_ear_depth = 100
    right_ear_depth = 100
    flag, right_ear_depth = find_depth_from(depth_image, depth_scale, face, 0, 1)
    flag0, left_ear_depth = find_depth_from(depth_image, depth_scale, face, 15, 16)
    if not flag and not flag0:
        return False
    ear_depth = left_ear_depth
    if right_ear_depth != 0 and left_ear_depth != 0:
        if right_ear_depth < left_ear_depth:
            ear_depth = right_ear_depth
        else:
            ear_depth = left_ear_depth
    elif right_ear_depth == 0:
        ear_depth = left_ear_depth
    elif left_ear_depth == 0:
        ear_depth = right_ear_depth

    chin_depth = 0
    flag, chin_depth = find_depth_from(depth_image, depth_scale, face, 7, 9)
    if not flag:
        return False

    nose_depth = 0
    flag, nose_depth = find_depth_from(depth_image, depth_scale, face, 29, 30)
    if not flag:
        return False

    right_eye_depth = 0
    flag, right_eye_depth = find_depth_from(depth_image, depth_scale, face, 36, 41)
    if not flag:
        return False
    left_eye_depth = 0
    flag, left_eye_depth = find_depth_from(depth_image, depth_scale, face, 42, 47)
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
    flag, mouth_depth = find_depth_from(depth_image, depth_scale, face, 48, 67)
    if not flag:
        return False
    
    # print(right_ear_depth, left_ear_depth, ear_depth, chin_depth, nose_depth, right_eye_depth, left_eye_depth, eye_depth, mouth_depth)

    # We just use simple heuristics to determine whether the depth information agrees with
    # what's expected: that the nose tip, for example, should be closer to the camera than
    # the eyes.
    # These heuristics are fairly basic but nonetheless serve to illustrate the point that
    # depth data can effectively be used to distinguish between a person and a picture of a
    # person...
    if nose_depth >= eye_depth:
        # print("nose_depth >= eye_depth")
        return False
    if eye_depth - nose_depth > 0.07:
        return False
    if ear_depth <= eye_depth:
        # print("ear_depth <= eye_depth")
        return False
    if mouth_depth <= nose_depth:
        # print("mouth_depth <= nose_depth")
        return False
    if mouth_depth > chin_depth:
        # print("mouth_depth > chin_depth")
        return False
    
    # All the distances, collectively, should not span a range that makes no sense. I.e.,
    # if the face accounts for more than 20cm of depth, or less than 2cm, then something's
    # not kosher!

    x = max([nose_depth, mouth_depth, chin_depth, eye_depth, ear_depth])
    n = min([nose_depth, mouth_depth, chin_depth, eye_depth, ear_depth])
    if x - n > 0.20:
        # print("x - n > 0.20")
        return False
    if x - n < 0.01:
        # print("x - n < 0.02")
        return False
    
    return True



# IS_VALIDATED = False
# shape = 0
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

# profile = pipeline.start(config)
# # get the depth sensor's depth scale
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_sensor.set_option(rs.option.emitter_enabled, 0)
# depth_scale = depth_sensor.get_depth_scale()

# detector = dlib.get_frontal_face_detector()
# predictor_path = '/home/haoran/GitHub/RealGuard/data/shape_predictor_68_face_landmarks.dat'
# predictor = dlib.shape_predictor(predictor_path)

# align_to_color = rs.align(rs.stream.color)

# win = dlib.image_window()

# while not win.is_closed():
#     frames = pipeline.wait_for_frames()
#     aligned_frames = align_to_color.process(frames)
#     aligned_depth_frame = aligned_frames.get_depth_frame()
#     # color_frame = aligned_frames.get_color_frame()
#     color_frame = frames.get_infrared_frame(1)
#     if not aligned_depth_frame or not color_frame:
#         continue
#     depth_image = np.asanyarray(aligned_depth_frame.get_data())
#     color_image = np.asanyarray(color_frame.get_data())
#     win.clear_overlay()
#     win.set_image(color_image)
#     dets = detector(color_image)
#     if len(dets) == 0:
#         IS_VALIDATED = False
#         shape = 0
#         depth_sensor.set_option(rs.option.emitter_enabled, 0)
#     else:
#         for k, d in enumerate(dets):
#             if not IS_VALIDATED:
#                 shape = predictor(color_image, d)
#                 if shape.num_parts == 68:
#                     IS_VALIDATED = True
#                     depth_sensor.set_option(rs.option.emitter_enabled, 1)
#                     print("Face validated")
#                 else:
#                     IS_VALIDATED = False
#                     shape = 0
#                     depth_sensor.set_option(rs.option.emitter_enabled, 0)
#             else:
#                 if validate_face(depth_image, depth_scale, shape):
#                     print("Good face")
#                     win.add_overlay(dlib.rectangle(d.left(), d.top(), d.right(), d.bottom()), dlib.rgb_pixel(0, 255, 0))
#                 else:
#                     win.add_overlay(dlib.rectangle(d.left(), d.top(), d.right(), d.bottom()), dlib.rgb_pixel(255, 0, 0))
#                     print("Bad face")
#                 shape = 0
#                 IS_VALIDATED = False
#                 depth_sensor.set_option(rs.option.emitter_enabled, 0)
#         # win.add_overlay(shape)
        