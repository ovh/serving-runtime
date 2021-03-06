package com.ovh.mls.serving.runtime.onnx;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.EvaluatorUtil;
import com.ovh.mls.serving.runtime.core.builder.InputStreamJsonIntoTensorIO;
import com.ovh.mls.serving.runtime.core.builder.TensorIOIntoTensorIO;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class OnnxEvaluatorIrisTest {
    private static final ClassLoader LOADER = OnnxEvaluatorIrisTest.class.getClassLoader();
    private static final Config CONFIG = ConfigFactory.load();
    private static final EvaluatorUtil MAPPER = new EvaluatorUtil(CONFIG);

    private OnnxEvaluator onnxEvaluator;

    @BeforeEach
    public void create() throws IOException, EvaluatorException {
        ObjectMapper objectMapper = new ObjectMapper();
        OnnxEvaluatorManifest manifest = objectMapper.readValue(
            LOADER.getResourceAsStream("onnx/iris/manifest.json"),
            OnnxEvaluatorManifest.class
        );
        onnxEvaluator = OnnxEvaluator.create(manifest, "./");
    }

    @Test
    public void evaluate() throws EvaluationException {

        final TensorIO input =
            new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper())
                .build(LOADER.getResourceAsStream("onnx/iris/single.json"));

        final EvaluationContext evaluationContext = new EvaluationContext();

        TensorIO output = onnxEvaluator.evaluate(input, evaluationContext);

        assertEquals(1, evaluationContext.totalEvaluation());
        assertEquals(
            Set.of("classification", "probability(0)", "probability(1)", "probability(2)"),
            output.tensorsNames()
        );

        Tensor classificationTensor = output.getTensor("classification");

        assertArrayEquals(new int[]{1}, classificationTensor.getShapeAsArray());
        assertEquals(2, ((long[]) classificationTensor.getData())[0]);

        Tensor probability0Tensor = output.getTensor("probability(0)");
        assertArrayEquals(new int[]{1}, probability0Tensor.getShapeAsArray());

        Tensor probability1Tensor = output.getTensor("probability(1)");
        assertArrayEquals(new int[]{1}, probability1Tensor.getShapeAsArray());

        Tensor probability2Tensor = output.getTensor("probability(2)");
        assertArrayEquals(new int[]{1}, probability2Tensor.getShapeAsArray());

        assertEquals(0.0011569946072995663, ((float[]) probability0Tensor.getData())[0], 0.000001);
        assertEquals(0.3964444100856781, ((float[]) probability1Tensor.getData())[0], 0.000001);
        assertEquals(0.6023985743522644, ((float[]) probability2Tensor.getData())[0], 0.000001);
    }

    @Test
    public void evaluateBatch() throws EvaluationException {

        final TensorIO input =
            new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper())
                .build(LOADER.getResourceAsStream("onnx/iris/batch.json"));


        final EvaluationContext evaluationContext = new EvaluationContext();

        TensorIO output = onnxEvaluator.evaluate(input, evaluationContext);

        assertEquals(2, evaluationContext.totalEvaluation());
        assertEquals(
            Set.of("classification", "probability(0)", "probability(1)", "probability(2)"), output.tensorsNames());

        Tensor classificationTensor = output.getTensor("classification");
        Tensor probability0 = output.getTensor("probability(0)");
        Tensor probability1 = output.getTensor("probability(1)");
        Tensor probability2 = output.getTensor("probability(2)");

        assertArrayEquals(new int[]{2}, classificationTensor.getShape().getArrayShape());
        long[] classificationTensorData = (long[]) classificationTensor.getData();

        assertArrayEquals(new int[]{2}, probability0.getShape().getArrayShape());
        float[] probability0Data = (float[]) probability0.getData();

        assertArrayEquals(new int[]{2}, probability1.getShape().getArrayShape());
        float[] probability1Data = (float[]) probability1.getData();

        assertArrayEquals(new int[]{2}, probability2.getShape().getArrayShape());
        float[] probability2Data = (float[]) probability2.getData();

        assertArrayEquals(new long[]{2, 2}, classificationTensorData);

        assertEquals(0.0028399815782904625, probability0Data[0], 0.000001);
        assertEquals(0.31503045558929443, probability1Data[0], 0.000001);
        assertEquals(0.6821296215057373, probability2Data[0], 0.000001);

        assertEquals(0.0002457280352246016, probability0Data[1], 0.000001);
        assertEquals(0.4622795879840851, probability1Data[1], 0.000001);
        assertEquals(0.5374746918678284, probability2Data[1], 0.000001);
    }

    @Test
    public void testMultiThread() throws InterruptedException {
        int numThreads = 10;
        int loop = 10;
        ExecutorService executor = Executors.newFixedThreadPool(10);

        String request =
            "{\"sepal_length\": 4.9,\"sepal_width\": 2.5, " +
                "\"petal_length\": 4.5, \"petal_width\": 1.7 }";

        for (int i = 0; i < numThreads; i++) {
            executor.submit(() -> {
                for (int j = 0; j < loop; j++) {
                    try {
                        InputStream is = new ByteArrayInputStream(request.getBytes());
                        final TensorIO input = new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper()).build(is);

                        TensorIO output = onnxEvaluator.evaluate(input, new EvaluationContext());
                        output = new TensorIOIntoTensorIO(
                            onnxEvaluator.getOutputTensorField(),
                            true
                        ).build(output);
                        assertEquals(2, (int) output.getTensor("classification").getData());

                    } catch (EvaluationException e) {
                        e.printStackTrace();
                    }
                }
            });
        }
        executor.shutdown();
        executor.awaitTermination(1, TimeUnit.MINUTES);
    }

}
