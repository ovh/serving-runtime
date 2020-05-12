package com.ovh.mls.serving.runtime.processors;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.ovh.mls.serving.runtime.core.AbstractEvaluatorManifest;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorManifest;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;

import java.util.Map;

@JsonTypeInfo(use = JsonTypeInfo.Id.NONE)
@IncludeAsEvaluatorManifest(type = "standard_scaler")
public class StandardScalerManifest extends AbstractEvaluatorManifest<Field> {

    private static final String type = "standard_scaler";

    private Map<String, MeanStd> meanStdMap;

    public Map<String, MeanStd> getMeanStdMap() {
        return meanStdMap;
    }

    public StandardScalerManifest setMeanStdMap(Map<String, MeanStd> meanStdMap) {
        this.meanStdMap = meanStdMap;
        return this;
    }

    @Override
    public StandardScaler create(String path) throws EvaluatorException {
        return StandardScaler.create(this, path);
    }

    @Override
    public String getType() {
        return type;
    }
}
