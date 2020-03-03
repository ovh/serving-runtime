package com.ovh.mls.serving.runtime.exceptions;

public class EvaluationException extends RuntimeException {
    public EvaluationException(String message) {
        super(message);
    }

    public EvaluationException(Exception e) {
        super(e);
    }

    public EvaluationException(String message, Exception e) {
        super(message, e);
    }
}
