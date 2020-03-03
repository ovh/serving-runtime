package com.ovh.mls.serving.runtime.core;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractEvaluatorManifest implements EvaluatorManifest {

    private List<Field> inputs = new ArrayList<>();
    private List<Field> outputs = new ArrayList<>();

    public List<Field> getInputs() {
        return inputs;
    }

    public AbstractEvaluatorManifest setInputs(List<Field> inputs) {
        this.inputs = inputs;
        return this;
    }

    public List<Field> getOutputs() {
        return outputs;
    }

    public AbstractEvaluatorManifest setOutputs(List<Field> outputs) {
        this.outputs = outputs;
        return this;
    }
}
