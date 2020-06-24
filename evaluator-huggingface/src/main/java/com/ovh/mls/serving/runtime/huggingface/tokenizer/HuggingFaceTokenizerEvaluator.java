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

    private final Tokenizer tokenizer;

    @Override
    public TensorIO evaluate(TensorIO io, EvaluationContext evaluationContext) throws EvaluationException {
        String input = (String) io.getTensors().get("sequence").getCoord(0);
        Encoding encoding = tokenizer.encode(input);

        TensorIO output = new TensorIO();
        output.getTensors().put("tokens", new Tensor(
            DataType.STRING,
            new int[]{encoding.getTokens().length},
            encoding.getTokens()
        ));
        output.getTensors().put("ids", new Tensor(
            DataType.INTEGER,
            new int[]{encoding.getIds().length},
            encoding.getIds()
        ));
        output.getTensors().put("typeIds", new Tensor(
            DataType.INTEGER,
            new int[]{encoding.getTypeIds().length},
            encoding.getTypeIds()
        ));
        output.getTensors().put("specialTokensMask", new Tensor(
            DataType.INTEGER,
            new int[]{encoding.getSpecialTokensMask().length},
            encoding.getSpecialTokensMask()
        ));
        output.getTensors().put("attentionMask", new Tensor(
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
            new Field("sequence", DataType.STRING)
        );
    }

    @Override
    public List<Field> getOutputs() {
        return List.of(
            new Field("tokens", DataType.STRING),
            new Field("ids", DataType.INTEGER),
            new Field("typeIds", DataType.INTEGER),
            new Field("specialTokensMask", DataType.INTEGER),
            new Field("attentionMask", DataType.INTEGER)
        );
    }
}
