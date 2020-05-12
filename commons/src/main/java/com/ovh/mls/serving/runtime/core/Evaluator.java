package com.ovh.mls.serving.runtime.core;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;

import java.util.List;

public interface Evaluator<F extends Field> {
    /**
     * Applies the evaluator operations over a Table and returns the initial
     * Table enriched with generated outputs.
     *
     * @param io 'EvaluatorIO' implementation over which the operations are applied
     * @return 'EvaluatorIO' implementation with generated outputs
     */
    TensorIO evaluate(TensorIO io, EvaluationContext evaluationContext) throws EvaluationException;

    /**
     * @return The list of inputs required by the evaluator
     */
    @JsonProperty("inputs")
    List<F> getInputs();

    /**
     * @return The list of outputs added to the Table after applying the evaluator
     */
    @JsonProperty("outputs")
    List<F> getOutputs();

    /**
     * @return The required number of rows in the Table for the evaluator
     */
    @JsonProperty("rolling_windows_size")
    default int getRollingWindowSize() {
        return 1;
    }
}
