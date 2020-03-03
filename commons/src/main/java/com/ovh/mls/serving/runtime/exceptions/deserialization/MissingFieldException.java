package com.ovh.mls.serving.runtime.exceptions.deserialization;

import java.util.Set;

public class MissingFieldException extends TableDeserializationException {

    public MissingFieldException(Set<String> missingFields) {
        super(String.format(
            "The following fields were expected but not found : %s",
            missingFields.toString()
        ));
    }

}
