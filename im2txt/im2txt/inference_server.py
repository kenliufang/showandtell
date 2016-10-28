# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from concurrent import futures
import time


import math
import os


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

from im2txt import server_pb2

import grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")

#tf.flags.DEFINE_int32("port", 5006, "Text file containing the vocabulary.")

class ShowAndTellService(server_pb2.ShowAndTellServiceServicer):

    def Init(self):
        # Build the inference graph.
        g = tf.Graph()
        with g.as_default():
            self.model = inference_wrapper.InferenceWrapper()
            restore_fn = self.model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
        g.finalize()

  # Create the vocabulary.
        self.vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

        self.sess = tf.Session(graph=g)
        # Load the model from checkpoint.
        restore_fn(self.sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        self.generator = caption_generator.CaptionGenerator(self.model, self.vocab)

        #for filename in filenames:
        #  with tf.gfile.GFile(filename, "r") as f:
        #    image = f.read()
        #  captions = generator.beam_search(sess, image)
        #  print("Captions for image %s:" % os.path.basename(filename))
        #  for i, caption in enumerate(captions):
        #    # Ignore begin and end words.
        #    sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        #    sentence = " ".join(sentence)
        #    print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
        print ("loaded\n")


    
    def ShowAndTell(self, request, context):
        captions = self.generator.beam_search(self.sess, request.image_data)
        reply = server_pb2.ShowAndTellReply()
        for i, caption in enumerate(captions):
            capt = reply.captions.add()
            # Ignore begin and end words.
            sentence = [self.vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            capt.caption = " ".join(sentence)
            capt.score = math.exp(caption.logprob)
        return reply

def main(_):
  service = ShowAndTellService()
  service.Init()
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  server_pb2.add_ShowAndTellServiceServicer_to_server(
          service, server)
  server.add_insecure_port('[::]:50051')
  server.start()
  try:
    while True:
        time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
      server.stop(0)


if __name__ == "__main__":
  tf.app.run()
