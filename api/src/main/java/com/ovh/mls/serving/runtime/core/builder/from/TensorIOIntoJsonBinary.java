package com.ovh.mls.serving.runtime.core.builder.from;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.builder.Builder;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.img.BinaryContent;
import org.apache.http.entity.ContentType;

/**
 * Convert a TensorIO into a Json Binary content
 */
public class TensorIOIntoJsonBinary implements Builder<TensorIO, BinaryContent> {

    private final ObjectMapper mapper;
    private final boolean shouldSimplify;

    public TensorIOIntoJsonBinary(ObjectMapper mapper, boolean shouldSimplify) {
        this.shouldSimplify = shouldSimplify;
        this.mapper = mapper;
    }

    @Override
    public BinaryContent build(TensorIO input) throws EvaluationException {
        try {
            return new BinaryContent(
                "json",
                ContentType.APPLICATION_JSON,
                this.mapper.writeValueAsBytes(
                    input.intoMap(shouldSimplify)
                )
            );
        } catch (JsonProcessingException e) {
            throw new EvaluationException(e);
        }
    }
}
