package com.ovh.mls.serving.runtime.onnx;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.EvaluatorManifest;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorManifest;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;

import java.io.IOException;
import java.util.List;

// In case of direct deserialization we override parent JsonTypeInfo
@JsonTypeInfo(use = JsonTypeInfo.Id.NONE)
@IncludeAsEvaluatorManifest(type = OnnxEvaluatorManifest.TYPE)
@JsonIgnoreProperties(ignoreUnknown = true)
public class OnnxEvaluatorManifest implements EvaluatorManifest {
    public static final String TYPE = "onnx";

    @JsonProperty("onnx_model_uri")
    private String onnxModelUri;

    private Integer batchSize = 1;

    private List<TensorField> inputs;
    private List<TensorField> outputs;

    @Override
    public Evaluator create(String path) throws IOException, EvaluatorException {
        return OnnxEvaluator.create(this, path);
    }

    @Override
    public String getType() {
        return TYPE;
    }

    public String getOnnxModelUri() {
        return onnxModelUri;
    }

    public void setOnnxModelUri(String onnxModelUri) {
        this.onnxModelUri = onnxModelUri;
    }

    public Integer getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(Integer batchSize) {
        this.batchSize = batchSize;
    }

    public List<TensorField> getInputs() {
        return inputs;
    }

    public void setInputs(List<TensorField> inputs) {
        this.inputs = inputs;
    }

    public List<TensorField> getOutputs() {
        return outputs;
    }

    public void setOutputs(List<TensorField> outputs) {
        this.outputs = outputs;
    }
}
