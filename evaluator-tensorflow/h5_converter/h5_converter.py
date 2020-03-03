import json
import os
import sys
import tempfile
from tensorflow.keras import backend as K

import tensorflow as tf

_AUTHORIZED_TYPES = {
    tf.bool: "boolean",
    tf.string: "string",

    tf.float16: "float",
    tf.float32: "float",
    tf.float64: "double",
    tf.bfloat16: "float",

    tf.int8: "integer",
    tf.int16: "integer",
    tf.int32: "integer",
    tf.int64: "long",
    tf.uint8: "integer",
    tf.uint16: "integer",
    tf.uint32: "integer",
    tf.uint64: "long",
    tf.qint8: "integer",
    tf.qint16: "integer",
    tf.qint32: "integer",
    tf.quint8: "integer",
    tf.quint16: "integer",

    tf.complex64: None,  # 64-bit single-precision complex.
    tf.complex128: None,  # 128-bit double-precision complex.
    tf.resource: None,  # Handle to a mutable resource.
    tf.variant: None  # Values of arbitrary types.
}


class _KerasExporter:

    def __init__(self, network):
        self.network = network
        self.savedmodelname = 'savedmodel'
        self.manifest_json_name = 'manifest.json'

    def export(self, path: str = None) -> str:
        if path is None:
            path = tempfile.mkdtemp()

        session = K.get_session()
        prediction_signature = tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
            {"inputs": self.network.input}, {"prediction": self.network.output})
        if not tf.compat.v1.saved_model.signature_def_utils.is_valid_signature(prediction_signature):
            raise ValueError("Error: Prediction signature not valid!")

        # Export tensorflow saved model
        saved_model_path = os.path.join(path, self.savedmodelname)
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(saved_model_path)
        builder.add_meta_graph_and_variables(
            session,
            [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            },
            main_op=tf.group(tf.tables_initializer(), name='legacy_init_op')
        )
        builder.save()

        # Export manifest
        manifest = self._compute_manifest()
        model_path = os.path.join(path, self.manifest_json_name)
        with open(model_path, 'w') as file_obj:
            json.dump(manifest, file_obj)

        return path

    def _input_tensor(self) -> tf.Tensor:
        return self.network.input

    def _output_tensor(self) -> tf.Tensor:
        return self.network.output

    def _input_manifest(self) -> list:
        return self._tensor_to_manifest(self._input_tensor(), 'input')

    def _output_manifest(self) -> list:
        return self._tensor_to_manifest(self._output_tensor(), 'output')

    def _tensor_to_manifest(self, tensor: tf.Tensor, tensor_type: str) -> list:
        # For all types see : https://www.tensorflow.org/api_docs/python/tf/dtypes/DType
        dimensions_shape = [dimension.value for dimension in tensor.shape.dims]

        type_str = _AUTHORIZED_TYPES.get(tensor.dtype)
        if not type_str:
            raise ValueError(f'Type of {tensor_type} tensor {str(tensor.dtype)} is not supported ...')

        shape_str = ', '.join([str(dim or '?') for dim in dimensions_shape])
        return [
            {
                'name': tensor.name,
                'shape': f'({shape_str})',
                'type': type_str,
                'fields': []
            }
        ]

    def _compute_manifest(self) -> dict:
        return {
            "type": "tensorflow",
            "batch_size": 1,
            "inputs": self._input_manifest(),
            "outputs": self._output_manifest()
        }


model = tf.keras.models.load_model(sys.argv[1])
_KerasExporter(network=model).export(sys.argv[2])
