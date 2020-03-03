package com.ovh.mls.serving.runtime.exceptions;

public class SwiftConfigurationException extends Exception {

    public SwiftConfigurationException(String message) {
        super(message);
    }

    public SwiftConfigurationException(Exception e) {
        super(e);
    }

    public static SwiftConfigurationException keyNotFoundException(String key) {
        String message = String.format("Swift key '%s' was not found in configurations collection", key);
        return new SwiftConfigurationException(message);
    }
}
