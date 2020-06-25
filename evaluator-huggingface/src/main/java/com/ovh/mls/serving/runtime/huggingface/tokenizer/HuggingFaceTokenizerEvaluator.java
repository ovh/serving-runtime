package com.ovh.mls.serving.runtime.huggingface.tokenizer;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;

import java.util.List;

public class HuggingFaceTokenizerEvaluator implements Evaluator<Field> {

    private static final String INPUT_SEQUENCE = "sequence";
    private static final String OUTPUT_TOKENS = "tokens";
    private static final String OUTPUT_IDS = "ids";
    private static final String OUTPUT_TYPE_IDS = "typeIds";
    private static final String OUTPUT_SPECIAL_TOKENS_MASK = "specialTokensMask";
    private static final String OUTPUT_ATTENTION_MASK = "attentionMask";

    private final Tokenizer tokenizer;

    @Override
    public TensorIO evaluate(TensorIO io, EvaluationContext evaluationContext) throws EvaluationException {
        String input = (String) io.getTensors().get("sequence").getCoord(0);
        Encoding encoding = tokenizer.encode(input);

        TensorIO output = new TensorIO();
        output.getTensors().put(OUTPUT_TOKENS, new Tensor(
            DataType.STRING,
            new int[]{encoding.getTokens().length},
            encoding.getTokens()
        ));
        output.getTensors().put(OUTPUT_IDS, new Tensor(
            DataType.INTEGER,
            new int[]{encoding.getIds().length},
            encoding.getIds()
        ));
        output.getTensors().put(OUTPUT_TYPE_IDS, new Tensor(
            DataType.INTEGER,
            new int[]{encoding.getTypeIds().length},
            encoding.getTypeIds()
        ));
        output.getTensors().put(OUTPUT_SPECIAL_TOKENS_MASK, new Tensor(
            DataType.INTEGER,
            new int[]{encoding.getSpecialTokensMask().length},
            encoding.getSpecialTokensMask()
        ));
        output.getTensors().put(OUTPUT_ATTENTION_MASK, new Tensor(
            DataType.INTEGER,
            new int[]{encoding.getAttentionMask().length},
            encoding.getAttentionMask()
        ));

        return output;
    }

    public HuggingFaceTokenizerEvaluator(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
    }

    @Override
    public List<Field> getInputs() {
        return List.of(
            new Field(INPUT_SEQUENCE, DataType.STRING)
        );
    }

    @Override
    public List<Field> getOutputs() {
        return List.of(
            new Field(OUTPUT_TOKENS, DataType.STRING),
            new Field(OUTPUT_IDS, DataType.INTEGER),
            new Field(OUTPUT_TYPE_IDS, DataType.INTEGER),
            new Field(OUTPUT_SPECIAL_TOKENS_MASK, DataType.INTEGER),
            new Field(OUTPUT_ATTENTION_MASK, DataType.INTEGER)
        );
    }
}
