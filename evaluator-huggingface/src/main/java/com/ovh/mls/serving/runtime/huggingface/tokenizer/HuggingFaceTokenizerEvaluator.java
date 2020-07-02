package com.ovh.mls.serving.runtime.huggingface.tokenizer;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;

import java.util.List;

/**
 * Inputs may of shape 1 (text) or shape N (tokenized text)
 * The second input is optional
 */
public class HuggingFaceTokenizerEvaluator implements Evaluator<Field> {

    private static final String INPUT1 = "input1";
    private static final String INPUT2 = "input2";
    private static final String OUTPUT_TOKENS = "tokens";
    private static final String OUTPUT_IDS = "ids";
    private static final String OUTPUT_TYPE_IDS = "typeIds";
    private static final String OUTPUT_SPECIAL_TOKENS_MASK = "specialTokensMask";
    private static final String OUTPUT_ATTENTION_MASK = "attentionMask";

    private final Tokenizer tokenizer;
    private final boolean addSpecialTokens;

    public HuggingFaceTokenizerEvaluator(Tokenizer tokenizer, boolean addSpecialTokens) {
        this.tokenizer = tokenizer;
        this.addSpecialTokens = addSpecialTokens;
    }

    @Override
    public TensorIO evaluate(TensorIO io, EvaluationContext evaluationContext) throws EvaluationException {
        // Encode input
        Tensor input1 = io.getTensors().get(INPUT1);
        Tensor input2 = io.getTensors().get(INPUT2);
        Encoding encoding;
        if (input1.getShape().getArrayShape()[0] == 1) {
            String inputSequence1 = (String) input1.getCoord(0);
            if (input2 == null) {
                // Encode one text input
                encoding = tokenizer.encode(inputSequence1, addSpecialTokens);
            } else {
                String inputSequence2 = (String) input2.getCoord(0);
                // Encode two text input
                encoding = tokenizer.encode(inputSequence1, inputSequence2, addSpecialTokens);
            }
        } else {
            String[] inputSequence1 = (String[]) input1.getData();
            if (input2 == null) {
                // Encode one tokenized input
                encoding = tokenizer.encode(inputSequence1, addSpecialTokens);
            } else {
                String[] inputSequence2 = (String[]) input2.getData();
                // Encode two tokenized input
                encoding = tokenizer.encode(inputSequence1, inputSequence2, addSpecialTokens);
            }
        }

        // Build output
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

    @Override
    public List<Field> getInputs() {
        return List.of(
            new Field(INPUT1, DataType.STRING),
            new Field(INPUT2, DataType.STRING)
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
