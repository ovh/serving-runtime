package com.ovh.mls.serving.runtime.onnx;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
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

import java.io.IOException;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class OnnxEvaluatorTitanicTest {
    private static final ClassLoader LOADER = OnnxEvaluatorTitanicTest.class.getClassLoader();
    private static final Config CONFIG = ConfigFactory.load();
    private static final EvaluatorUtil MAPPER = new EvaluatorUtil(CONFIG);
    private OnnxEvaluator onnxEvaluator;

    @BeforeEach
    public void create() throws IOException, EvaluatorException {
        ObjectMapper objectMapper = new ObjectMapper();
        OnnxEvaluatorManifest manifest = objectMapper.readValue(
            LOADER.getResourceAsStream("onnx/titanic/manifest.json"),
            OnnxEvaluatorManifest.class
        );
        onnxEvaluator = OnnxEvaluator.create(manifest, "");
    }

    @Test
    public void evaluate() throws EvaluationException {

        final TensorIO input =
            new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper())
                .build(LOADER.getResourceAsStream("onnx/titanic/single.json"));

        TensorIO output = onnxEvaluator.evaluate(input, new EvaluationContext());

        assertEquals(
            Set.of("classification", "probability(0)", "probability(1)"),
            output.tensorsNames()
        );

        Tensor classificationTensor = output.getTensor("classification");
        Tensor probability0 = output.getTensor("probability(0)");
        Tensor probability1 = output.getTensor("probability(1)");

        assertArrayEquals(new int[]{1}, classificationTensor.getShapeAsArray());
        assertArrayEquals(new int[]{1}, probability0.getShapeAsArray());
        assertArrayEquals(new int[]{1}, probability1.getShapeAsArray());

        assertEquals(0, ((long[]) classificationTensor.getData())[0]);
        assertEquals(0.8032850623130798, ((float[]) probability0.getData())[0], 0.00000001);
        assertEquals(0.19671493768692017, ((float[]) probability1.getData())[0], 0.00000001);
    }

    @Test
    public void evaluateBatch() throws EvaluationException {

        final TensorIO input =
            new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper())
                .build(LOADER.getResourceAsStream("onnx/titanic/batch.json"));

        TensorIO output = onnxEvaluator.evaluate(input, new EvaluationContext());

        assertEquals(
            Set.of("classification", "probability(0)", "probability(1)"),
            output.tensorsNames()
        );

        Tensor classificationTensor = output.getTensor("classification");
        Tensor probability0 = output.getTensor("probability(0)");
        Tensor probability1 = output.getTensor("probability(1)");

        assertArrayEquals(new int[]{2}, classificationTensor.getShapeAsArray());
        assertArrayEquals(new int[]{2}, probability0.getShapeAsArray());
        assertArrayEquals(new int[]{2}, probability1.getShapeAsArray());

        assertEquals(0, ((long[]) classificationTensor.getData())[0]);
        assertEquals(0.8032850623130798, ((float[]) probability0.getData())[0]);
        assertEquals(0.19671493768692017, ((float[]) probability1.getData())[0]);

        assertEquals(1, ((long[]) classificationTensor.getData())[1]);
        assertEquals(0.3988564610481262, ((float[]) probability0.getData())[1]);
        assertEquals(0.6011435389518738, ((float[]) probability1.getData())[1]);
    }
}
