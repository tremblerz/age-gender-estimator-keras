# import tensorflow as tf
# from tensorflow.python.platform import gfile
#
# model_filename = 'tensorflow_model/constant_graph_weights.pb'
# with gfile.FastGFile(model_filename, 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     g_in = tf.import_graph_def(graph_def)
#
# with tf.Session(graph=g_in) as sess:
#     # print([n.name for n in tf.get_default_graph().as_graph_def().node])
#
#     input_tensors = [sess.graph.get_tensor_by_name('import/the_input:0')]
#     output_tensors = [sess.graph.get_tensor_by_name('import/output_node0:0'),
#                       sess.graph.get_tensor_by_name('import/output_node1:0')
#                       ]
#
#     print(input_tensors[0].get_shape())
#
#     print(input_tensors, output_tensors)
#
#     tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, input_tensors, output_tensors)
#     open("converted_model.tflite", "wb").write(tflite_model)

import tensorflow as tf


graph_def_file = "tensorflow_model/constant_graph_weights.pb"
input_arrays = ["the_input"]
output_arrays = ["output_node0", "output_node1"]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)