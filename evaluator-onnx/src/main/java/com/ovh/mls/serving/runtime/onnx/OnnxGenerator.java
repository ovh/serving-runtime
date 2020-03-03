package com.ovh.mls.serving.runtime.onnx;

import ai.onnxruntime.MapInfo;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxSequence;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.EvaluatorGenerator;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorGenerator;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.core.tensor.TensorIndex;
import com.ovh.mls.serving.runtime.core.tensor.TensorShape;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.typesafe.config.Config;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@IncludeAsEvaluatorGenerator(extension = "onnx")
public class OnnxGenerator implements EvaluatorGenerator {

    // Build Output TensorBuilder from ONNX Informations
    static List<TensorField> buildOutput(
        Map<String, NodeInfo> nodeInfos,
        OrtSession.Result scores,
        boolean batch
    ) throws EvaluatorException {
        List<TensorField> outputs = new ArrayList<>();
        for (Map.Entry<String, NodeInfo> entry : nodeInfos.entrySet()) {
            Optional<OnnxValue> optionalOnnxValue = scores.get(entry.getKey());
            if (optionalOnnxValue.isEmpty()) {
                throw new EvaluatorException(String.format("Missing output: %s", entry.getKey()));
            }
            OnnxValue onnxValue = optionalOnnxValue.get();
            switch (onnxValue.getType()) {
                case ONNX_TYPE_TENSOR:
                    outputs.add(buildOutputFromTensor(entry.getValue(), batch));
                    break;
                case ONNX_TYPE_SEQUENCE:
                    // Seq are handle as tensors, we just need to create a specific tensor.
                    outputs.add(buildFromSeq((OnnxSequence) onnxValue, entry.getKey(), batch));
            }
        }
        return outputs;
    }

    private static TensorField buildFromSeq(
        OnnxSequence value,
        String name,
        boolean batch
    ) throws EvaluatorException {

        var info = value.getInfo();

        if (info.mapInfo == null) {
            throw new EvaluatorException("Sequence without map not handle");
        }

        int[] shape;
        if (batch) {
            // Seems like a bug in onnx sklearn, the output does not seems batched like the input
            // https://github.com/onnx/sklearn-onnx/issues/312
            shape = new int[] {-1, info.mapInfo.size};
        } else {
            shape = new int[] {info.mapInfo.size};
        }

        return new TensorField(
            name,
            getType(info.mapInfo.valueType),
            new TensorShape(shape),
            new ArrayList<>()
        );
    }

    private static TensorField buildOutputFromTensor(NodeInfo nodeInfo, boolean batch) throws EvaluatorException {
        var tensorInfo = (TensorInfo) nodeInfo.getInfo();

        var shape = tensorInfo.getShape();
        if (batch) {
            // Seems like a bug in onnx sklearn, the output does not seems batched like the input
            // https://github.com/onnx/sklearn-onnx/issues/312
            shape[0] = -1;
        }

        TensorShape tensorShape = new TensorShape(shape);

        return new TensorField(
            nodeInfo.getName(),
            getType(tensorInfo.type),
            tensorShape,
            new ArrayList<>()
        );
    }

    static Map<String, TensorField> buildInputs(Map<String, NodeInfo> nodeInfos) throws EvaluatorException {

        return nodeInfos.entrySet().stream().collect(Collectors.toMap(
            Map.Entry::getKey,
            entry -> {
                var tensorField = new TensorField();
                tensorField.setName(entry.getKey());

                NodeInfo nodeInfo = entry.getValue();

                if (!(nodeInfo.getInfo() instanceof TensorInfo)) {
                    throw new EvaluatorException(
                        String.format("Entry node not handle: %s", nodeInfo.getInfo().toString())
                    );
                }

                var tensorInfo = (TensorInfo) nodeInfo.getInfo();
                var shape = new TensorShape(tensorInfo.getShape());

                TensorField tField = new TensorField(
                    nodeInfo.getName(),
                    getType(tensorInfo.type),
                    shape,
                    new ArrayList<>()
                );
                return tField;
            }
        ));
    }

    private static DataType getType(OnnxJavaType javaType) throws EvaluatorException {
        switch (javaType) {
            case DOUBLE:
                return DataType.DOUBLE;
            case INT32:
                return DataType.INTEGER;
            case INT64:
                return DataType.LONG;
            case BOOL:
                return DataType.BOOLEAN;
            case FLOAT:
                return DataType.FLOAT;
            case STRING:
                return DataType.STRING;
            case UNKNOWN:
            case INT8:
            case INT16:
            default:
                throw new EvaluatorException("Type not handle: " + javaType);
        }
    }

    @Override
    public Evaluator generate(File file, Config config) throws EvaluatorException, FileNotFoundException {
        return OnnxEvaluator.create(new FileInputStream(file));
    }
}
