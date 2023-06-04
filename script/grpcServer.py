import grpc
import realGuard_pb2, realGuard_pb2_grpc
from concurrent import futures
import logging
import os,time
import numpy as np
import dlib_recognize_face
import cv2

detect_path = './model/mmod_human_face_detector.dat'
predictor_path = './model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = './model/dlib_face_recognition_resnet_model_v1.dat'
faces_folder = './face'
FACES_FEATURES_CSV_FILE = './data/face_features.csv'

# Status Code List
# 100: Success
# 404: Error
# 201: No Face Detected
# 202: Depth Validation Fail
# 203: Face Recognition Fail
class auth(realGuard_pb2_grpc.authServicer):
    def __init__(self) -> None:
        pass

    def do_auth(self, request, context):
        
        #ir Img
        f = open("./pic/irImg.jpg",'wb')
        f.write(request.ir_img)
        f.close()
        #depth data
        depthData = request.depth_data

        #auth logic
        image_2d = DRecFace.read_image("./pic/irImg.jpg")
        # depth = np.array(depthData)
        # Convert from Bytes to int
        depth = np.frombuffer(depthData, dtype=np.uint16)
        # is_recognized, name, dist = DRecFace.recognize_from_2_frame(image_2d, depth, 0.35)
        # 如果为灰度图则转为RGB
        if len(image_2d.shape) == 2:
            image_2d = cv2.cvtColor(image_2d, cv2.COLOR_GRAY2RGB)
        is_recognized, name, id, dist = DRecFace.recognize_from_2_frame(image_2d, depth, 0.30)

        if is_recognized == 1:
            print("Recognized: " + name + " " + id + " " + str(dist))
            return realGuard_pb2.auth_result(status = 100, result = dist, name = name, id = "Success", instruction = 0)
        elif is_recognized == -1:
            print("No Face Detected")
            return realGuard_pb2.auth_result(status = 201, result = -1, name = "Unknown", id = "No Face Detected", instruction = 0)
        elif is_recognized == -2:
            print("Depth Validation Fail")
            return realGuard_pb2.auth_result(status = 202, result = -1, name = "Unknown", id = "Depth Validation Fail", instruction = 0)
        elif is_recognized == -3:
            print("Face Recognition Fail")
            return realGuard_pb2.auth_result(status = 203, result = -1, name = "Unknown", id = "Face Recognition Fail", instruction = 0)
        else:
            print("Error")
            return realGuard_pb2.auth_result(status = 404, result = -1, name = "Unknown", id = "Error", instruction = 0)

        #fill in this result list
            

class register(realGuard_pb2_grpc.registerServicer):
    def __init__(self) -> None:
        pass

    def pic_register(self, request, context):
        #ir Img
        f = open("./pic/irImg_toRegister.jpg",'wb')
        f.write(request.ir_img)
        f.close()

        candidateName : str= request.name
        cadidateId : str = request.studentId

        #不采纳时仅作验证
        isPicTaken : bool = request.take

        reg_image = cv2.imread("./pic/irImg_toRegister.jpg")

        if isPicTaken:
            
            # 如果为灰度图则转为RGB
            if len(reg_image.shape) == 2:
                reg_image = cv2.cvtColor(reg_image, cv2.COLOR_GRAY2RGB)
            isreg, isnew, dist = DRecFace.register_face(reg_image, candidateName, cadidateId)
            #logic to be compeleted
            if isreg:
                if isnew:
                    status_ = 100
                    dist_ = 0
                else:
                    status_ = 101
                    dist_ = dist
                DRecFace.save_image(reg_image, candidateName)
            else:
                status_ = 400
                dist_ = -1
        else:
            status_ = 200
            isreg, isnew, dist = DRecFace.register_face(reg_image, candidateName, cadidateId, isCheck=True)
            dist_ = dist
            

        #return the result
        #status说明：
        #首次录入：100
        #再次录入：101
        #进行验证：200
        #不合法：400
        #其中result即为dist，当照片检测出不合法、无法操作时result应为-1，当照片为首次录入时，返回0
        return realGuard_pb2.register_result(status = status_, result = dist_, take = isPicTaken)

        



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    realGuard_pb2_grpc.add_authServicer_to_server(auth(), server)
    realGuard_pb2_grpc.add_registerServicer_to_server(register(), server)
    server.add_insecure_port('[::]:5051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    DRecFace = dlib_recognize_face.Recognize_Face(detect_path, predictor_path, face_rec_model_path, FACES_FEATURES_CSV_FILE, faces_folder)
    # DRegFace = dlib_recognize_face.Register_Face(detect_path, predictor_path, face_rec_model_path, FACES_FEATURES_CSV_FILE, faces_folder)
    logging.basicConfig()
    print("Server is running...")
    print("Waiting for client...")
    serve()