package com.ovh.mls.serving.runtime.core;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/**
 * Class implementing EvaluatorManifest need to add this annotation and declare a unique type property
 * use by the ObjectMapper to deserialize the JSON manifest. All classes with this annotation will be registered
 * at runtime as EvaluatorManifest subtypes.
 *
 * @see EvaluatorUtil
 * @see EvaluatorManifest
 */
@Retention(RetentionPolicy.RUNTIME)
public @interface IncludeAsEvaluatorManifest {
    String type();
}
