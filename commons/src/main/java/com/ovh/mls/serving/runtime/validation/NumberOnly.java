package com.ovh.mls.serving.runtime.validation;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/**
 * Annotation used for validation purposes. Any Evaluator annotated with this annotation can only handle Number inputs.
 *
 * @see Validator
 */
@Retention(RetentionPolicy.RUNTIME)
public @interface NumberOnly {
}
