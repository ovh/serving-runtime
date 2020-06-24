package com.ovh.mls.serving.runtime.huggingface.tokenizer;

import java.util.Objects;

public class Offset {

    private final long start;
    private final long end;

    public Offset(long start, long end) {
        this.start = start;
        this.end = end;
    }

    public long getStart() {
        return start;
    }

    public long getEnd() {
        return end;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Offset offset = (Offset) o;
        return start == offset.start && end == offset.end;
    }

    @Override
    public int hashCode() {
        return Objects.hash(start, end);
    }

    @Override
    public String toString() {
        return "Offset{" +
            "start=" + start +
            ", end=" + end +
            '}';
    }
}
