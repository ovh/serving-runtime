package com.ovh.mls.serving.runtime.huggingface.tokenizer;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.ovh.mls.serving.runtime.core.AbstractEvaluatorManifest;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorManifest;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;

import java.nio.file.Paths;

// In case of direct deserialization we override parent JsonTypeInfo
@JsonTypeInfo(use = JsonTypeInfo.Id.NONE)
@IncludeAsEvaluatorManifest(type = HuggingFaceTokenizerEvaluatorManifest.TYPE)
public class HuggingFaceTokenizerEvaluatorManifest extends AbstractEvaluatorManifest<TensorField> {

    public static final String TYPE = "huggingface_tokenizer";

    @JsonProperty("saved_model_uri")
    private String savedModelUri;

    @Override
    public HuggingFaceTokenizerEvaluator create(String path) throws EvaluatorException {
        Tokenizer tokenizer = Tokenizer.fromFile(Paths.get(savedModelUri));
        return new HuggingFaceTokenizerEvaluator(tokenizer);
    }

    @Override
    public String getType() {
        return TYPE;
    }
}
