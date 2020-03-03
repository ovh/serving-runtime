package com.ovh.mls.serving.runtime.exceptions.deserialization;

import java.util.List;
import java.util.Map;

public class DifferentColumnLengthException extends TableDeserializationException {

    public DifferentColumnLengthException(Map<Integer, List<String>> errorMap) {
        super(
            String.format(
                "Not all the column have the same size : %s",
                errorMap.toString()
            )
        );
    }

}
