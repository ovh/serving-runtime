package com.ovh.mls.serving.runtime.core;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/**
 * Class annotated with EvaluatorGenerator need to add this annotation and declare a unique extension name
 *
 * @see EvaluatorGenerator
 */
@Retention(RetentionPolicy.RUNTIME)
public @interface IncludeAsEvaluatorGenerator {
    String extension();
}
