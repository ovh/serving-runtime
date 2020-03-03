package com.ovh.mls.serving.runtime.tensorflow;

import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.EvaluatorGenerator;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorGenerator;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.typesafe.config.Config;
import org.tensorflow.SavedModelBundle;

import java.io.File;

import static com.ovh.mls.serving.runtime.tensorflow.TensorflowEvaluator.DEFAULT_TAG_TENSORFLOW;

@IncludeAsEvaluatorGenerator(extension = "pb")
public class TensorflowPbGenerator implements EvaluatorGenerator {

    @Override
    public Evaluator generate(File filename, Config evaluatorConfig) throws EvaluatorException {

        SavedModelBundle savedModel = SavedModelBundle.load(
            filename.getParent(),
            DEFAULT_TAG_TENSORFLOW
        );
        return TensorflowEvaluator.create(savedModel);
    }
}
