package com.ovh.mls.serving.runtime.huggingface.tokenizer;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.ovh.mls.serving.runtime.core.AbstractEvaluatorManifest;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorManifest;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.nio.file.Paths;

// In case of direct deserialization we override parent JsonTypeInfo
@JsonTypeInfo(use = JsonTypeInfo.Id.NONE)
@IncludeAsEvaluatorManifest(type = HuggingFaceTokenizerEvaluatorManifest.TYPE)
public class HuggingFaceTokenizerEvaluatorManifest extends AbstractEvaluatorManifest<TensorField> {

    public static final String TYPE = "huggingface_tokenizer";

    private static final Logger LOGGER = LoggerFactory.getLogger(HuggingFaceTokenizerEvaluatorManifest.class);

    @JsonProperty
    private String savedModelUri;

    @JsonProperty
    private boolean addSpecialTokens;

    @Override
    public HuggingFaceTokenizerEvaluator create(String path) throws EvaluatorException {
        Tokenizer tokenizer;
        try {
            tokenizer = Tokenizer.fromFile(Paths.get(savedModelUri));
        } catch (EvaluatorException e1) {
            Path localSavedModelUri = Paths.get(path, savedModelUri);
            try {
                tokenizer = Tokenizer.fromFile(localSavedModelUri);
            } catch (EvaluatorException e2) {
                LOGGER.error("Cannot load HuggingFace tokenizer {}", savedModelUri, e1);
                LOGGER.error("Cannot load HuggingFace tokenizer {}", localSavedModelUri, e2);
                throw new EvaluatorException(
                    String.format("Cannot load HuggingFace tokenizer %s or %s", savedModelUri, localSavedModelUri)
                );
            }
        }
        return new HuggingFaceTokenizerEvaluator(tokenizer, addSpecialTokens);
    }

    @Override
    public String getType() {
        return TYPE;
    }
}
