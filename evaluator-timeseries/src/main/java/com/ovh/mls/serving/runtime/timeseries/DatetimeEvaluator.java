package com.ovh.mls.serving.runtime.timeseries;

import com.ovh.mls.serving.runtime.core.AbstractTensorEvaluator;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Evaluator used to encode and decode datetimes, it takes only one input which is the date to decode
 * It can answer with several output, each one representing a configured encoded value for the input date
 */
public class DatetimeEvaluator extends AbstractTensorEvaluator<DateTensorField> {

    private final DateTensorField singleInput;

    public DatetimeEvaluator(
        List<DateTensorField> inputs,
        List<DateTensorField> outputs
    ) {
        super(inputs, outputs, 1);
        if (inputs.size() != 1) {
            throw new EvaluatorException("There should be only 1 input for a datetime evaluator");
        }
        this.singleInput = this.getInputs().get(0);
    }

    @Override
    protected TensorIO evaluateTensor(TensorIO tensorIO) throws EvaluationException {
        String inputTensorName = this.singleInput.getName();
        Tensor inputTensor = tensorIO.getTensor(inputTensorName);
        if (inputTensor == null) {
            throw new EvaluationException(String.format("Impossible to find a tensor with name '%s'", inputTensorName));
        }
        if (this.singleInput.getType() != inputTensor.getType()) {
            throw new EvaluationException(
                String.format("Input type for tensor %s is different from given one", inputTensorName));
        }

        Function<Object, LocalDateTime> decodeFunction = this.singleInput.decodeDatetimeFunction();
        Map<String, Tensor> outputsTensors = new HashMap<>();
        for (DateTensorField outputsField : this.getOutputs()) {
            Function<LocalDateTime, Object> encodeFunction = outputsField.encodeDatetimeFunction();
            Function<Object, Object> fullTransfo = encodeFunction.compose(decodeFunction);
            Tensor outputTensor = inputTensor.apply(fullTransfo, outputsField.getType());
            outputsTensors.put(outputsField.getName(), outputTensor);
        }
        return new TensorIO(outputsTensors);
    }

}
