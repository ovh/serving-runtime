package com.ovh.mls.serving.runtime.exceptions.deserialization;

import com.ovh.mls.serving.runtime.core.Field;

public class UnexpectedValueForColumnException extends TableDeserializationException {

    public UnexpectedValueForColumnException(Field expectedField, Object object) {
        super(
            String.format(
                "Expected column '%s' to be of type '%s', found '%s' with value '%s'",
                expectedField.getName(),
                expectedField.getType().toString(),
                object.getClass().getSimpleName(),
                object.toString()
            )
        );
    }
}
