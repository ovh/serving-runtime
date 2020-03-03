package com.ovh.mls.serving.runtime.core.builder;

import com.ovh.mls.serving.runtime.exceptions.EvaluationException;

/**
 * Builder is a generic interface for converting an Object into another
 * @param <I> The input object class
 * @param <O> The output object class
 */
public interface Builder<I, O> {

    O build(I input) throws EvaluationException;

}
