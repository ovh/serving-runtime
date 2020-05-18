package com.ovh.mls.serving.runtime.timeseries;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.ovh.mls.serving.runtime.core.AbstractEvaluatorManifest;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorManifest;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;

// In case of direct deserialization we override parent JsonTypeInfo
@JsonTypeInfo(use = JsonTypeInfo.Id.NONE)
@IncludeAsEvaluatorManifest(type = DatetimeEvaluatorManifest.TYPE)
public class DatetimeEvaluatorManifest extends AbstractEvaluatorManifest<DateTensorField> {

    public static final String TYPE = "datetime_encoder";

    @Override
    public DatetimeEvaluator create(String path) throws EvaluatorException {
        return new DatetimeEvaluator(this.getInputs(), this.getOutputs());
    }

    @Override
    public String getType() {
        return TYPE;
    }

}
