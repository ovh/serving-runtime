package com.ovh.mls.serving.runtime.huggingface.tokenizer;

import java.util.Optional;

/**
 * Represents the output of a Tokenizer
 */
public class Encoding {

    // Pointer to the Rust structure
    private long handle = -1;

    private Encoding() {
    }

    public native boolean isEmpty();

    public native long size();

    public native String[] getTokens();

    public native Optional<Integer>[] getWords();

    public native int[] getIds();

    public native int[] getTypeIds();

    public native Offset[] getOffsets();

    public native int[] getSpecialTokensMask();

    public native int[] getAttentionMask();

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
