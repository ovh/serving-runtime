package com.ovh.mls.serving.runtime.timeseries;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.ovh.mls.serving.runtime.core.AbstractEvaluatorManifest;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorManifest;

import java.util.Collections;
import java.util.Map;

// In case of direct deserialization we override parent JsonTypeInfo
@JsonTypeInfo(use = JsonTypeInfo.Id.NONE)
@IncludeAsEvaluatorManifest(type = "pi")
public class PredictionIntervalEvaluatorManifest extends AbstractEvaluatorManifest {

    private static final String type = "pi";

    @JsonProperty("residuals_std")
    private Map<String, Double> residualsStd = Collections.emptyMap();

    public Map<String, Double> getResidualsStd() {
        return residualsStd;
    }

    public PredictionIntervalEvaluatorManifest setResidualsStd(Map<String, Double> residualsStd) {
        this.residualsStd = residualsStd;
        return this;
    }

    public PredictionIntervalEvaluator create(String path) {
        return PredictionIntervalEvaluator.create(this, path);
    }

    @Override
    public String getType() {
        return type;
    }
}
