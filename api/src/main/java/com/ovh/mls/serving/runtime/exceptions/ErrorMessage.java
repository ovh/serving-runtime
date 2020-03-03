package com.ovh.mls.serving.runtime.exceptions;

import org.eclipse.jetty.http.HttpStatus;

/**
 * Error Message Model
 */
public class ErrorMessage {
    private final int status;
    private final String message;

    public ErrorMessage(String message) {
        this.message = message;
        this.status = HttpStatus.INTERNAL_SERVER_ERROR_500;
    }

    public ErrorMessage(String message, int status) {
        this.message = message;
        this.status = status;
    }

    public int getStatus() {
        return status;
    }

    public String getMessage() {
        return message;
    }
}
