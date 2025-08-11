# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2021/3/22
# Description:
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def frozen_graph_tacotron2(model, logdir, name, print_graph=False):
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda text, text_len, speaker_ids: model.inference(text, text_len, speaker_ids))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.input_ids.shape, model.input_ids.dtype),
        tf.TensorSpec(model.input_lengths.shape, model.input_lengths.dtype),
        tf.TensorSpec(model.speaker_ids.shape, model.speaker_ids.dtype)
    )

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    if print_graph:
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 50)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=logdir,
                      name=name,
                      as_text=False)


def frozen_graph_parallel_wavagan(model, logdir, name, print_graph=False):
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda mels: model.inference(mels))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs.shape, model.inputs.dtype)
    )

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    if print_graph:
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 50)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=logdir,
                      name=name,
                      as_text=False)


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))
