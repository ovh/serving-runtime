package com.ovh.mls.serving.runtime.core;

import com.ovh.mls.serving.runtime.core.builder.Builder;
import com.ovh.mls.serving.runtime.core.builder.TensorIOIntoTensorIO;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.core.tensor.TensorIndex;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;

import java.beans.Transient;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public abstract class AbstractTensorEvaluator<F extends TensorField> implements Evaluator<F> {

    private final int rollingWindowsSize;

    private final List<F> inputTensorFields;
    private final List<F> outputTensorFields;
    private final Builder<TensorIO, TensorIO> inputTensorBuilder;
    private final Builder<TensorIO, TensorIO> outputTensorBuilder;

    public AbstractTensorEvaluator(
        List<F> inputTensorFields,
        List<F> outputTensorFields,
        int rollingWindowsSize
    ) {
        this.rollingWindowsSize = rollingWindowsSize;
        this.inputTensorFields = inputTensorFields;

        this.outputTensorFields = outputTensorFields;

        // The input tensor builder should apply the configured rolling window
        this.inputTensorBuilder = new TensorIOIntoTensorIO(
            this.inputTensorFields, false, this.rollingWindowsSize);

        // The output tensor should build indexes if any and simplify the shape so that output will as as simple for
        // user to understand
        this.outputTensorBuilder = new TensorIOIntoTensorIO(this.outputTensorFields, true);
    }

    @Override
    public TensorIO evaluate(TensorIO tensorIO, EvaluationContext evaluationContext)
        throws EvaluationException {
        // Convert the input TensorIO into the wanted format for the serialized model
        TensorIO input = this.inputTensorBuilder.build(tensorIO);
        // If user ask for debug the input step of that evaluator, return this step
        if (evaluationContext.shouldStop(true)) {
            return input;
        }
        // Evaluate the serialized model
        TensorIO output = this.evaluateTensor(input);
        // If user ask for debug the output step of that evaluator, return this step
        if (evaluationContext.shouldStop(false)) {
            return output;
        }
        // Convert the output of the model into the simpliest format for the user
        TensorIO finalOutput = this.outputTensorBuilder.build(output);
        // Increment the evaluation context with the evaluated batch size
        evaluationContext.incEvaluationBy(output.getBatchSize());
        return finalOutput;
    }

    protected abstract TensorIO evaluateTensor(TensorIO tensorIO) throws EvaluationException;

    @Override
    public List<F> getInputs() {
        return new ArrayList<F>(this.inputTensorFields);
    }

    @Override
    public List<F> getOutputs() {
        return new ArrayList<F>(this.outputTensorFields);
    }

    @Override
    public int getRollingWindowSize() {
        return rollingWindowsSize;
    }

    @Transient
    public List<? extends TensorField> getInputTensorField() {
        return this.inputTensorFields;
    }

    @Transient
    public List<? extends TensorField> getOutputTensorField() {
        return this.outputTensorFields;
    }

    @Transient
    public List<TensorIndex> getInputTensorIndexes() {
        return this.getInputTensorField()
            .stream()
            .flatMap(x -> x.getFields().stream())
            .collect(Collectors.toList());
    }

    @Transient
    public List<TensorIndex> getOutputTensorIndexes() {
        return this.getOutputTensorField()
            .stream()
            .flatMap(x -> x.getFields().stream())
            .collect(Collectors.toList());
    }

}
