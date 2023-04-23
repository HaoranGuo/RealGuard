import grpc
import realGuard_pb2, realGuard_pb2_grpc
from concurrent import futures
import logging

class auth(realGuard_pb2_grpc.authServicer):
    def __init__(self) -> None:
        pass

    def do_auth(self, request, context):

        f = open("./irImg.jpg",'wb')
        f.write(request.ir_img)
        f.close()

        return realGuard_pb2.auth_result(status = 400, result = 0.89, name = "Huandong", id = "23456", instruction = 0)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    realGuard_pb2_grpc.add_authServicer_to_server(auth(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()