package com.ovh.mls.serving.runtime.core;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.databind.PropertyNamingStrategy;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;

import java.io.IOException;

/**
 * @see Evaluator
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonNaming(PropertyNamingStrategy.SnakeCaseStrategy.class)
public interface EvaluatorManifest {

    Evaluator create(String path) throws EvaluatorException, IOException;

    String getType();

}
