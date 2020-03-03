package com.ovh.mls.serving.runtime.exceptions.deserialization;

public class TableDeserializationException extends RuntimeException {

    public TableDeserializationException(String message) {
        super(message);
    }

    public TableDeserializationException(String message, Throwable cause) {
        super(message, cause);
    }

}
