from tensorflow import keras
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.keras.applications.vgg16 import VGG16
import flops.node_manager as nm
import sys


class flop_calculator:
    """
    class used to calculate the number of flops of a neural network
    """
    def flop_calc(self, concrete_func):
        """
        method used to calculate flops
        :param concrete_func: concrete function
        :return: final number of flops
        """
        # replaces variables in a graph with constants of the same values
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(
            concrete_func)

        # operate on the graph
        with tf.Graph().as_default() as graph:
            # imports the graph from graph_def
            tf.graph_util.import_graph_def(graph_def, name='')

            # used to extract secondary informations runtime, such as execution time and memory consumption
            run_meta = tf.compat.v1.RunMetadata()

            # build profiler to monitor a given network parameter, in this case 'float operations'
            opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
            ).with_empty_output().float_operation())
            opts['output'] = 'none'

            # get number of flops through the profiler
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts)

            return flops

    def get_flops(self, model):
        """
        method used to obtain the number of flops
        :param model: model whose flops need to be calculated
        :return: number of total flops
        """

        # concrete function to apply to each layer to obtain inputs value
        concrete = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])

        flops = self.flop_calc(concrete_func)
        res_dict = nm.to_dict(flops)
        print(" ---- REPORT CUSTOM ------\n")
        print('TOTAL_FLOPS: {}\n'.format(flops.total_float_ops))
        print(res_dict)
        return flops.total_float_ops

# initial_model = keras.Sequential([
#         keras.Input(shape=(1, 320, 280, 1)),
#     layers.Conv2D(1, (2, 1), strides=(1, 1), padding="valid",
#                   use_bias=False, kernel_initializer=tf.keras.initializers.Ones())
# ])


def analyze_model(initial_model):
    """
    method used to obtain the number of flops
    :param model: model whose flops need to be calculated
    :return: number of total flops
    """
    model = initial_model

    # concrete function to apply to each layer to obtain inputs value
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in initial_model.inputs])

    # total_flops, graph_nodes = get_flops(concrete_func)

    # calculate number of flops using the previous concrete function
    flops = flop_calculator().flop_calc(concrete_func)

    # convert frozen graph to dict, containing flops number for each operations in the neural network
    res_dict = nm.to_dict(flops)

    return flops, res_dict


if __name__ == "__main__":

    im = VGG16()

    f, r_dict = analyze_model(im)

    print(" ---- REPORT CUSTOM ------\n")
    print('TOTAL_FLOPS: {}\n'.format(f.total_float_ops))
    print(r_dict)
