package com.ovh.mls.serving.runtime.onnx;

import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.EvaluatorUtil;
import com.ovh.mls.serving.runtime.core.builder.InputStreamJsonIntoTensorIO;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class OnnxGeneratorIrisTest {
    private static final ClassLoader LOADER = OnnxGeneratorIrisTest.class.getClassLoader();
    private static final Config CONFIG = ConfigFactory.load();
    private static final EvaluatorUtil MAPPER = new EvaluatorUtil(CONFIG);

    private Evaluator onnxEvaluator;

    @BeforeEach
    public void create() throws EvaluatorException, FileNotFoundException {
        onnxEvaluator = new OnnxGenerator().generate(
            new File(LOADER.getResource("onnx/iris/iris.onnx").getFile()),
            CONFIG
        );
    }

    @Test
    public void evaluate() throws EvaluationException {
        final TensorIO input =
            new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper())
                .build(LOADER.getResourceAsStream("onnx/iris/single_gen.json"));

        final EvaluationContext evaluationContext = new EvaluationContext();

        TensorIO output = onnxEvaluator.evaluate(input, evaluationContext);

        assertEquals(1, evaluationContext.totalEvaluation());
        assertEquals(
            Set.of("output_label", "output_probability"),
            output.tensorsNames()
        );

        Tensor outputLabelTensor = output.getTensor("output_label");
        Tensor outputProb0Tensor = output.getTensor("output_probability");

        assertArrayEquals(new int[]{1}, outputLabelTensor.getShapeAsArray());
        assertArrayEquals(new int[]{1, 3}, outputProb0Tensor.getShapeAsArray());

        assertEquals(2, ((long[]) outputLabelTensor.getData())[0]);
        assertEquals(0.0011569946072995663, ((float[][]) outputProb0Tensor.getData())[0][0], 0.000001);
        assertEquals(0.3964444100856781, ((float[][]) outputProb0Tensor.getData())[0][1], 0.000001);
        assertEquals(0.6023985743522644, ((float[][]) outputProb0Tensor.getData())[0][2], 0.000001);

    }

    @Test
    public void evaluateBatch() throws EvaluationException {
        final TensorIO input =
            new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper())
                .build(LOADER.getResourceAsStream("onnx/iris/batch_gen.json"));

        final EvaluationContext evaluationContext = new EvaluationContext();

        TensorIO output = onnxEvaluator.evaluate(input, evaluationContext);

        assertEquals(2, evaluationContext.totalEvaluation());

        Tensor outputLabelTensor = output.getTensor("output_label");
        Tensor outputProb0Tensor = output.getTensor("output_probability");

        assertArrayEquals(new int[]{2}, outputLabelTensor.getShapeAsArray());
        assertArrayEquals(new int[]{2, 3}, outputProb0Tensor.getShapeAsArray());

        assertEquals(2, ((long[]) outputLabelTensor.getData())[0]);
        assertEquals(0.0028399815782904625, ((float[][]) outputProb0Tensor.getData())[0][0], 0.000001);
        assertEquals(0.31503045558929443, ((float[][]) outputProb0Tensor.getData())[0][1], 0.000001);
        assertEquals(0.6821296215057373, ((float[][]) outputProb0Tensor.getData())[0][2], 0.000001);
        assertEquals(2, ((long[]) outputLabelTensor.getData())[1]);
        assertEquals(0.0002457280352246016, ((float[][]) outputProb0Tensor.getData())[1][0], 0.000001);
        assertEquals(0.4622795879840851, ((float[][]) outputProb0Tensor.getData())[1][1], 0.000001);
        assertEquals(0.5374746918678284, ((float[][]) outputProb0Tensor.getData())[1][2], 0.000001);
    }
}
