package com.ovh.mls.serving.runtime.validation;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;

class ValidatorTest {

    @Test
    void validate() throws EvaluatorException {
        List<Field> inputs = Collections.singletonList(
            new Field("testinput", DataType.DOUBLE)
        );

        EvaluatorTest evaluatorTest = new EvaluatorTest(inputs);
        Validator.validate(evaluatorTest);
    }

    @Test
    void failValidate() {
        List<Field> inputs = Collections.singletonList(
            new Field("testinput", DataType.STRING)
        );

        EvaluatorTest evaluatorTest = new EvaluatorTest(inputs);
        Assertions.assertThrows(EvaluatorException.class, () ->
            Validator.validate(evaluatorTest)
        );
    }

    @NumberOnly
    public static class EvaluatorTest implements Evaluator {

        List<Field> inputs;

        public EvaluatorTest(List<Field> inputs) {
            this.inputs = inputs;
        }

        @Override
        public TensorIO evaluate(TensorIO io, EvaluationContext evaluationContext) {
            return null;
        }

        @Override
        public List<Field> getInputs() {
            return this.inputs;
        }

        @Override
        public List<Field> getOutputs() {
            return null;
        }

        @Override
        public int getRollingWindowSize() {
            return 0;
        }
    }
}
