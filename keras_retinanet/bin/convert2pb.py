import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
from sys import argv
import sys

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

from .. import models

input_path = argv[-2]
weight_file = argv[-1]
weight_file_path = osp.join(input_path,weight_file)
output_graph_name = weight_file[:-3] + '.pb'


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return tf.Session(config=config)


# convert hdf5 file to protocolbuffer file
def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    if osp.exists(output_dir) is False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)

output_dir = osp.join(os.getcwd(),"trans_model")
K.tensorflow_backend.set_session(get_session())

# optionally load config parameters
anchor_parameters = None

h5_model = models.load_model(weight_file_path)
# check if this is indeed a training model
models.check_training_model(h5_model)

# convert the model
h5_model = models.convert_model(h5_model, nms=True, class_specific_filter=False, anchor_params=anchor_parameters)
h5_to_pb(h5_model,output_dir = output_dir,model_name = output_graph_name)
print('model saved')