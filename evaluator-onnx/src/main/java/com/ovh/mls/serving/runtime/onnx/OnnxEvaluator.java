package com.ovh.mls.serving.runtime.onnx;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxSequence;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.ovh.mls.serving.runtime.core.AbstractTensorEvaluator;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.core.tensor.TensorShape;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;


public class OnnxEvaluator extends AbstractTensorEvaluator {
    private static final OrtEnvironment ONNX_ENVIRONMENT;

    static {
        ONNX_ENVIRONMENT = OrtEnvironment.getEnvironment();
    }

    private final OrtSession onnxModel;

    private OnnxEvaluator(
        OrtSession onnxModel,
        List<TensorField> inputTensorFields,
        List<TensorField> outputTensorFields,
        int batchSize
    ) {
        super(inputTensorFields, outputTensorFields, batchSize);

        this.onnxModel = onnxModel;

    }

    // Create a evaluator directly from an ONNX File
    static OnnxEvaluator create(InputStream inputStream) throws EvaluatorException {
        try {
            byte[] bytes = IOUtils.toByteArray(inputStream);
            OrtSession model = ONNX_ENVIRONMENT.createSession(bytes);

            Map<String, TensorField> tensorFields = OnnxGenerator.buildInputs(model.getInputInfo());

            // For now, we need to call the model in order to get more information about the output
            // https://github.com/microsoft/onnxruntime/issues/2334
            Map<String, OnnxTensor> tensors = new HashMap<>(tensorFields.size());
            boolean handleBatch = false;
            for (Map.Entry<String, TensorField> entry : tensorFields.entrySet()) {
                TensorField tensorField = entry.getValue();
                OnnxTensor onnxTensor = OnnxTensor.createTensor(
                    ONNX_ENVIRONMENT,
                    createArray(tensorField.getType(),
                        tensorField.getTensorShape(),
                        1)
                );
                tensors.put(entry.getKey(), onnxTensor);
                handleBatch = tensorField.getTensorShape().handleBatch();
            }

            // Score
            OrtSession.Result scores = model.run(tensors);

            List<TensorField> outputs = OnnxGenerator.buildOutput(model.getOutputInfo(), scores, handleBatch);

            return new OnnxEvaluator(model, new LinkedList<>(tensorFields.values()), outputs, 1);
        } catch (OrtException | IOException e) {
            throw new EvaluatorException(e);
        }
    }

    // Create a evaluator from a manifest
    static OnnxEvaluator create(OnnxEvaluatorManifest manifest, String path) throws IOException, EvaluatorException {
        var inputStream = FileUtils.openInputStream(new File(path, manifest.getOnnxModelUri()));

        try {
            byte[] bytes = IOUtils.toByteArray(inputStream);
            OrtSession model = ONNX_ENVIRONMENT.createSession(bytes);

            return new OnnxEvaluator(model, manifest.getInputs(), manifest.getOutputs(), manifest.getBatchSize());

        } catch (OrtException | IOException e) {
            throw new EvaluatorException(e);
        }
    }

    @Override
    protected TensorIO evaluateTensor(TensorIO tensorIO) throws EvaluationException {
        // Build input
        var inputTensors = this
            .getInputTensorField()
            .stream()
            .collect(Collectors.toMap(
                Field::getName,
                tensorField -> buildInput(tensorField.getName(), tensorIO)
            ));

        // Score
        OrtSession.Result scores = getScore(inputTensors);

        TensorIO output = new TensorIO();

        // Transform results
        getOutputTensorField().forEach(tensorField -> {
            Optional<OnnxValue> result = scores.get(tensorField.getName());
            if (result.isEmpty()) {
                throw new EvaluationException(String.format("Missing output: %s", tensorField.getName()));
            }
            TensorIO singleTensorOutput = getOutput(tensorField, result.get());
            output.merge(singleTensorOutput);
        });

        // Free memory
        inputTensors.values().forEach(OnnxTensor::close);
        scores.iterator().forEachRemaining(entry -> entry.getValue().close());

        return output;
    }

    private OrtSession.Result getScore(Map<String, OnnxTensor> tensors) {
        try {
            return this.onnxModel.run(tensors);
        } catch (OrtException e) {
            throw new EvaluationException(e);
        }
    }

    private static Object createArray(DataType type, TensorShape shape, int batchSize) {
        final Object array = Array.newInstance(type.getJavaClass(), shape.getNewShape(batchSize));

        if (type == DataType.STRING) {
            // To avoid onnx c binding error, we fill the array with an empty string
            fillStringArray(array);
        }

        return array;
    }

    private static void fillStringArray(Object object) {
        for (int i = 0; i < Array.getLength(object); i++) {
            final Object subArray = Array.get(object, i);

            if (subArray == null) {
                Arrays.fill((String[]) object, "");
                return;
            } else {
                fillStringArray(subArray);
            }
        }
    }

    private static TensorIO getOutput(
        TensorField tensorField,
        OnnxValue value
    ) {
        HashMap<String, Tensor> tensorMap = new HashMap<>();
        try {
            switch (value.getType()) {
                case ONNX_TYPE_SEQUENCE:
                    // We transform the seq in a array
                    var mapFromSeq = createArrayFromSeq(tensorField, (OnnxSequence) value);
                    Tensor tensorSeq = Tensor.fromData(tensorField.getType(), mapFromSeq);
                    tensorMap.put(tensorField.getName(), tensorSeq);
                    break;
                case ONNX_TYPE_TENSOR:
                    Tensor tensor = Tensor.fromData(tensorField.getType(), value.getValue());
                    tensorMap.put(tensorField.getName(), tensor);
                    break;
                default:
                    throw new OrtException(
                        String.format("Not handle: %s,Info: %s", value.getType(), value.getInfo()));
            }
            return new TensorIO(tensorMap);

        } catch (OrtException e) {
            throw new EvaluationException(e);
        }
    }

    private static Object createArrayFromSeq(
        TensorField tensorField,
        OnnxSequence sequence
    ) throws OrtException {
        var batchSize = sequence.getInfo().length;
        var array = createArray(tensorField.getType(), tensorField.getTensorShape(), batchSize);
        var sequenceResults = sequence.getValue();

        if (!sequence.getInfo().isSequenceOfMaps() || sequence.getInfo().mapInfo.keyType != OnnxJavaType.INT64) {
            throw new EvaluationException("Only sequence of maps with INT64 keys are supported for now...");
        }

        for (int i = 0; i < batchSize; i++) {
            var subArray = Array.get(array, i);
            Map map = (Map) sequenceResults.get(i);
            for (int j = 0; j < sequence.getInfo().mapInfo.size; j++) {
                Object obj = map.get(((Integer) j).longValue());
                Array.set(subArray, j, obj);
            }
        }
        return array;
    }

    private static OnnxTensor buildInput(String name, TensorIO io) throws EvaluationException {
        try {
            Tensor tensor = io.getTensor(name);
            return OnnxTensor.createTensor(ONNX_ENVIRONMENT, tensor.getData());
        } catch (OrtException e) {
            throw new EvaluationException(e);
        }
    }

}
