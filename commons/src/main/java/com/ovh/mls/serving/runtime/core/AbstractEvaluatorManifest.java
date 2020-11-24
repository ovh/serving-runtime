package com.ovh.mls.serving.runtime.core;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractEvaluatorManifest<F extends Field> implements EvaluatorManifest {

    private List<F> inputs = new ArrayList<>();
    private List<F> outputs = new ArrayList<>();

    public List<F> getInputs() {
        return inputs;
    }

    public AbstractEvaluatorManifest<F> setInputs(List<F> inputs) {
        this.inputs = inputs;
        return this;
    }

    public List<F> getOutputs() {
        return outputs;
    }

    public AbstractEvaluatorManifest<F> setOutputs(List<F> outputs) {
        this.outputs = outputs;
        return this;
    }
}
