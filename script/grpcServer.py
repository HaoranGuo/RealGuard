import grpc
import realGuard_pb2, realGuard_pb2_grpc
from concurrent import futures
import logging
import os,time
import numpy as np
import dlib_recognize_face

predictor_path = './model/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = './model/dlib_face_recognition_resnet_model_v1.dat'
faces_folder = './face'
FACES_FEATURES_CSV_FILE = './data/face_features.csv'

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
        depth = np.array(depthData)
        is_recognized, name, dist = DRecFace.recognize_from_2_frame(image_2d, depth, 0.35)

        if is_recognized:
            print("Recognized: " + name)
        else:
            print("Not Recognized")

        #fill in this result list
        return realGuard_pb2.auth_result(status = 400, result = dist, name = name, id = "23456", instruction = 0)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    realGuard_pb2_grpc.add_authServicer_to_server(auth(), server)
    server.add_insecure_port('[::]:5051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    DRecFace = dlib_recognize_face.Recognize_Face(predictor_path, face_rec_model_path, FACES_FEATURES_CSV_FILE)
    logging.basicConfig()
    print("Server is running...")
    print("Waiting for client...")
    serve()