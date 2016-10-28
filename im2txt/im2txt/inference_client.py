r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from concurrent import futures
import time


import math
import os


import tensorflow as tf

from im2txt import server_pb2

import grpc

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file", "",
                       "input file"
                       "input")
#

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = server_pb2.ShowAndTellServiceStub(channel)
    with tf.gfile.GFile(FLAGS.input_file, "r") as f:
        image = f.read()
        response = stub.ShowAndTell(
                server_pb2.ShowAndTellRequest(image_data=image))
        for caption in response.captions:
            print("%s (p=%f)" % (caption.caption, caption.score))
    #print("Greeter client received: " + response.message)

if __name__ == '__main__':
    run()
