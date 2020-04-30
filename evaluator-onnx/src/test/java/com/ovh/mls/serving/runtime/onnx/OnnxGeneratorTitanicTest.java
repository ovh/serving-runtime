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

class OnnxGeneratorTitanicTest {
    private static final ClassLoader LOADER = OnnxGeneratorTitanicTest.class.getClassLoader();
    private static final Config CONFIG = ConfigFactory.load();
    private static final EvaluatorUtil MAPPER = new EvaluatorUtil(CONFIG);
    private Evaluator onnxEvaluator;

    @BeforeEach
    public void create() throws EvaluatorException, FileNotFoundException {
        onnxEvaluator = new OnnxGenerator().generate(new File(
                LOADER.getResource("onnx/titanic/pipeline_titanic.onnx").getFile()),
            CONFIG
        );
    }

    @Test
    public void evaluate() throws EvaluationException {

        final TensorIO input =
            new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper())
                .build(LOADER.getResourceAsStream("onnx/titanic/single_gen.json"));

        TensorIO output = onnxEvaluator.evaluate(input, new EvaluationContext());

        assertEquals(
            Set.of("output_label", "output_probability"),
            output.tensorsNames()
        );

        Tensor outputLabel = output.getTensor("output_label");
        Tensor outputProbability0 = output.getTensor("output_probability");

        assertArrayEquals(new int[]{1}, outputLabel.getShapeAsArray());
        assertArrayEquals(new int[]{1, 2}, outputProbability0.getShapeAsArray());

        assertEquals(0, ((long[]) outputLabel.getData())[0]);
        assertEquals(0.8032850623130798, ((float[][]) outputProbability0.getData())[0][0], 0.0000001);
        assertEquals(0.19671493768692017, ((float[][]) outputProbability0.getData())[0][1], 0.0000001);
    }

    @Test
    public void evaluateBatch() throws EvaluationException {

        final TensorIO input =
            new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper())
                .build(LOADER.getResourceAsStream("onnx/titanic/batch_gen.json"));

        TensorIO output = onnxEvaluator.evaluate(input, new EvaluationContext());

        assertEquals(
            Set.of("output_label", "output_probability"),
            output.tensorsNames()
        );

        Tensor outputLabel = output.getTensor("output_label");
        Tensor outputProbability0 = output.getTensor("output_probability");

        assertArrayEquals(new int[]{2}, outputLabel.getShapeAsArray());
        assertArrayEquals(new int[]{2, 2}, outputProbability0.getShapeAsArray());

        assertEquals(0, ((long[]) outputLabel.getData())[0]);
        assertEquals(0.8032850623130798, ((float[][]) outputProbability0.getData())[0][0], 0.0000001);
        assertEquals(0.19671493768692017, ((float[][]) outputProbability0.getData())[0][1], 0.0000001);
        assertEquals(1, ((long[]) outputLabel.getData())[1]);
        assertEquals(0.3988564610481262, ((float[][]) outputProbability0.getData())[1][0], 0.0000001);
        assertEquals(0.6011435389518738, ((float[][]) outputProbability0.getData())[1][1], 0.0000001);
    }
}
