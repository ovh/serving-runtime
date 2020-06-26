package com.ovh.mls.serving.runtime.huggingface.tokenizer;

import com.ovh.mls.serving.runtime.utils.NativeUtils;

import java.io.IOException;
import java.nio.file.Path;

/**
 * A `Tokenizer` is capable of encoding/decoding any text
 */
public class Tokenizer {

    private static final String NATIVE_LIBRARY_NAME = "huggingface_tokenizer_jni";

    static {
        try {
            // Look for library in classpath
            System.loadLibrary(NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError ignored) {
            try {
                // Look for library in JAR
                NativeUtils.loadLibraryFromJar("/" + System.mapLibraryName(NATIVE_LIBRARY_NAME));
            } catch (IOException e) {
                throw new IllegalStateException("Cannot load native library", e);
            }
        }
    }

    // Pointer to the Rust structure
    private long handle = -1;

    private Tokenizer() {
    }

    /**
     * Instantiate a new Tokenizer from the given file
     *
     * May throw an EvaluatorException
     */
    public static native Tokenizer fromFile(Path file);

    /**
     * Encode the given input
     *
     * May throw an EvaluatorException
     */
    public native Encoding encode(String input);

    /**
     * Encode the given input
     *
     * May throw an EvaluatorException
     */
    public native Encoding encode(String input1, String input2);

    /**
     * Encode the given input
     *
     * May throw an EvaluatorException
     */
    public native Encoding encode(String[] tokens);

    /**
     * Encode the given input
     *
     * May throw an EvaluatorException
     */
    public native Encoding encode(String[] tokens1, String[] tokens2);

    /**
     * Give back the Rust pointer to be freed
     */
    private native void releaseHandle();

    @Override
    protected void finalize() throws Throwable {
        try {
            releaseHandle();
        } finally {
            super.finalize();
        }
    }
}
