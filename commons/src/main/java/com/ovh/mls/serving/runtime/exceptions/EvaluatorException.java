package com.ovh.mls.serving.runtime.exceptions;

public class EvaluatorException extends RuntimeException {
    public EvaluatorException(String message) {
        super(message);
    }

    public EvaluatorException(Exception e) {
        super(e);
    }

    public EvaluatorException(String message, Exception e) {
        super(e);
    }
}
