package com.ovh.mls.serving.runtime.core;

import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.typesafe.config.Config;

import java.io.File;
import java.io.FileNotFoundException;

public interface EvaluatorGenerator {

    Evaluator generate(File filename, Config evaluatorConfig) throws EvaluatorException, FileNotFoundException;

}
