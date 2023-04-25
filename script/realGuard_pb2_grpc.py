# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import realGuard_pb2 as realGuard__pb2


class authStub(object):
    """前端申请认证
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.do_auth = channel.unary_unary(
                '/realGuardRpc.auth/do_auth',
                request_serializer=realGuard__pb2.auth_request.SerializeToString,
                response_deserializer=realGuard__pb2.auth_result.FromString,
                )


class authServicer(object):
    """前端申请认证
    """

    def do_auth(self, request, context):
        """Sends a greeting
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_authServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'do_auth': grpc.unary_unary_rpc_method_handler(
                    servicer.do_auth,
                    request_deserializer=realGuard__pb2.auth_request.FromString,
                    response_serializer=realGuard__pb2.auth_result.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'realGuardRpc.auth', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
# class auth(object):
#     """前端申请认证
#     """

#     @staticmethod
#     def do_auth(request,
#             target,
#             options=(),
#             channel_credentials=None,
#             call_credentials=None,
#             insecure=False,
#             compression=None,
#             wait_for_ready=None,
#             timeout=None,
#             metadata=None):
#         return grpc.experimental.unary_unary(request, target, '/realGuardRpc.auth/do_auth',
#             realGuard__pb2.auth_request.SerializeToString,
#             realGuard__pb2.auth_result.FromString,
#             options, channel_credentials,
#             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)