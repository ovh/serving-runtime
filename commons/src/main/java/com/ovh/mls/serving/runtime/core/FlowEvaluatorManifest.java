package com.ovh.mls.serving.runtime.core;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@IncludeAsEvaluatorManifest(type = "flow")
public class FlowEvaluatorManifest implements EvaluatorManifest {

    private static final String type = "flow";

    private List<Field> outputs = new ArrayList<>();

    @JsonProperty("evaluator_manifests")
    private List<EvaluatorManifest> evaluatorManifests;

    public List<EvaluatorManifest> getEvaluatorManifests() {
        return evaluatorManifests;
    }

    public FlowEvaluatorManifest setEvaluatorManifests(List<EvaluatorManifest> evaluatorManifests) {
        this.evaluatorManifests = evaluatorManifests;
        return this;
    }

    public List<Field> getOutputs() {
        return outputs;
    }

    public FlowEvaluatorManifest setOutputs(List<Field> outputs) {
        this.outputs = outputs;
        return this;
    }

    @Override
    public FlowEvaluator create(String path) throws EvaluatorException, IOException {
        return FlowEvaluator.create(this, path);
    }

    @Override
    public String getType() {
        return type;
    }
}
