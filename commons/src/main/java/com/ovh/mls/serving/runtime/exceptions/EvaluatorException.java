package com.ovh.mls.serving.runtime.exceptions;

public class EvaluatorException extends RuntimeException {
    public EvaluatorException(String message) {
        super(message);
    }

    public EvaluatorException(Throwable throwable) {
        super(throwable);
    }

    public EvaluatorException(String message, Throwable throwable) {
        super(message, throwable);
    }
}
