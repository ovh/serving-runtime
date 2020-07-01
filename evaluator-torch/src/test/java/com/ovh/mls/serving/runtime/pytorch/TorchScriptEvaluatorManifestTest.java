package com.ovh.mls.serving.runtime.pytorch;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.EvaluatorManifest;
import com.ovh.mls.serving.runtime.core.EvaluatorUtil;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.typesafe.config.ConfigFactory;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Map;

class TorchScriptEvaluatorManifestTest {

    @Test
    public void testCreate() throws IOException {
        // Load evaluator
        EvaluatorUtil evaluatorUtil = new EvaluatorUtil(ConfigFactory.load());
        TorchScriptEvaluatorManifest torchScriptEvaluatorManifest = (TorchScriptEvaluatorManifest) evaluatorUtil
            .getObjectMapper()
            .readValue(
                getClass().getResourceAsStream("/manifest.json"),
                EvaluatorManifest.class
            );
        TorchScriptEvaluator evaluator = torchScriptEvaluatorManifest.create("src/test/resources");

        // Evaluate
        TensorIO input = new TensorIO(Map.of(
            "input_0", new Tensor(DataType.FLOAT, new int[]{2}, new float[]{0.5f, 0.2f})
        ));
        TensorIO output = evaluator.evaluateTensor(input);
        float[] output0 = (float[]) output.getTensors().get("output_0").getData();

        Assertions.assertEquals(output0[0], 0.120716706f, 1e-6);
        Assertions.assertEquals(output0[1], 0.054772355f, 1e-6);
    }
}