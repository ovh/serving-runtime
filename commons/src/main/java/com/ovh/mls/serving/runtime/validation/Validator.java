package com.ovh.mls.serving.runtime.validation;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import org.apache.commons.lang3.StringUtils;

import java.util.stream.Collectors;

public class Validator {

    /**
     * Applies validation check over an evaluator instance based on its implementation class available
     * annotations.
     *
     * @param evaluator instance to check
     * @throws EvaluatorException if one check fails an exception is thrown
     */
    public static void validate(Evaluator<?> evaluator) throws EvaluatorException {
        if (evaluator.getClass().isAnnotationPresent(NumberOnly.class)) {
            checkNumbers(evaluator);
        }
    }

    /**
     * For evaluator that only support Numbers as input check that all input fields are of Number type.
     * @see DataType
     *
     * @param evaluator evaluator only supporting Number input
     * @throws EvaluatorException throws an exception if one field is not a Number type
     */
    private static void checkNumbers(Evaluator<?> evaluator) throws EvaluatorException {

        String nonNumberFields = evaluator.getInputs().stream()
            .filter(field -> !DataType.isNumberType(field.getType()))
            .map(field -> String.format("(%s,%s)", field.getName(), field.getType()))
            .collect(Collectors.joining(","));

        if (StringUtils.isNotEmpty(nonNumberFields)) {
            throw new EvaluatorException(String.format(
                "Only Number fields are supported for evaluator %s. Unsupported fields: %s",
                evaluator.getClass().getSimpleName(),
                nonNumberFields
            ));
        }
    }
}
