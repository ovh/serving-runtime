package com.ovh.mls.serving.runtime.tensorflow;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.databind.PropertyNamingStrategy;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.ovh.mls.serving.runtime.core.EvaluatorManifest;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorManifest;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;

import java.io.IOException;
import java.util.List;

@IncludeAsEvaluatorManifest(type = "tensorflow")
@JsonTypeInfo(use = JsonTypeInfo.Id.NONE)
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonNaming(PropertyNamingStrategy.SnakeCaseStrategy.class)
public class TensorflowEvaluatorManifest implements EvaluatorManifest {

    private static final String type = "tensorflow";

    private String savedModelUri;

    private Integer batchSize = 1;

    private List<TensorField> inputs;

    private List<TensorField> outputs;

    public TensorflowEvaluatorManifest() {
    }

    public String getSavedModelUri() {
        return savedModelUri;
    }

    public void setSavedModelUri(String savedModelUri) {
        this.savedModelUri = savedModelUri;
    }

    public Integer getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(Integer batchSize) {
        this.batchSize = batchSize;
    }

    @Override
    public String getType() {
        return type;
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

    @Override
    public TensorflowEvaluator create(String path1) throws EvaluatorException, IOException {
        return TensorflowEvaluator.create(this, path1);
    }
}
