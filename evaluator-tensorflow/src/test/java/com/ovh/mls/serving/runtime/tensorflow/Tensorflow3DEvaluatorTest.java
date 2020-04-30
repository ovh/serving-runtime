package com.ovh.mls.serving.runtime.tensorflow;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.EvaluatorUtil;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.builder.InputStreamJsonIntoTensorIO;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorShape;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

@SuppressWarnings("ALL")
public class Tensorflow3DEvaluatorTest {
    private static final ClassLoader LOADER = Tensorflow3DEvaluatorTest.class.getClassLoader();
    private static final Config CONFIG = ConfigFactory.load().getConfig("evaluator");
    private static final EvaluatorUtil MAPPER = new EvaluatorUtil(CONFIG);

    private TensorflowEvaluator tensorFlowEvaluator;

    @BeforeEach
    public void create() throws IOException, EvaluatorException {
        ObjectMapper objectMapper = new ObjectMapper();
        TensorflowEvaluatorManifest manifest = objectMapper.readValue(
            LOADER.getResourceAsStream("tensorflow/3d_savedmodel/3d-tensorflow-model-manifest.json"),
            TensorflowEvaluatorManifest.class
        );
        tensorFlowEvaluator = TensorflowEvaluator.create(manifest, "./");
    }

    @Test
    public void evaluate_tensor() throws EvaluationException {

        final TensorIO tensorIO =
            new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper())
                .build(LOADER.getResourceAsStream("tensorflow/3d_savedmodel/batch_tensor.json"));

        TensorIO output = (TensorIO) tensorFlowEvaluator.evaluate(tensorIO, new EvaluationContext());
        float[] outputTensor = (float[]) output.getTensor("scaled_imputed_label_predicted").getData();

        assertEquals(-0.1746857464313507, outputTensor[0], 0.0000001);
        assertEquals(-0.8396763205528259, outputTensor[1], 0.0000001);
    }

    @Test
    public void evaluate_table() throws EvaluationException {
        InputStreamJsonIntoTensorIO builder = new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper());
        TensorIO tensorIO = builder.build(LOADER.getResourceAsStream("tensorflow/3d_savedmodel/batch.json"));

        final EvaluationContext evaluationContext = new EvaluationContext();

        TensorIO output = tensorFlowEvaluator.evaluate(tensorIO, evaluationContext);

        assertEquals(2, evaluationContext.totalEvaluation());
        assertEquals(Set.of("scaled_imputed_label_predicted"), output.tensorsNames());

        Tensor tensor = output.getTensor("scaled_imputed_label_predicted");
        TensorShape shape = tensor.getShape();
        assertArrayEquals(new int[]{2}, shape.getArrayShape());
        assertEquals(DataType.FLOAT, tensor.getType());
        assertEquals(-0.1746857464313507, ((float[]) tensor.getData())[0], 0.0000001);
        assertEquals(-0.8396763205528259, ((float[]) tensor.getData())[1], 0.0000001);
    }

    @Test
    public void getInputs() {
        List<? extends Field> inputs = tensorFlowEvaluator.getInputTensorIndexes();
        assertEquals(4, inputs.size());
        assertEquals("scaled_imputed_t", inputs.get(0).getName());
        assertEquals(DataType.FLOAT, inputs.get(0).getType());
        assertEquals("scaled_imputed_label", inputs.get(1).getName());
        assertEquals(DataType.FLOAT, inputs.get(1).getType());
        assertEquals("scaled_imputed_feature1", inputs.get(2).getName());
        assertEquals(DataType.FLOAT, inputs.get(2).getType());
        assertEquals("scaled_imputed_feature2", inputs.get(3).getName());
        assertEquals(DataType.FLOAT, inputs.get(3).getType());
    }

    @Test
    public void getOutputs() {
        List<? extends Field> outputs = tensorFlowEvaluator.getOutputTensorIndexes();

        assertEquals(1, outputs.size());

        assertEquals("scaled_imputed_label_predicted", outputs.get(0).getName());
        assertEquals(DataType.FLOAT, outputs.get(0).getType());
    }

    @Test
    public void getBatchSize() {
        assertEquals(2, tensorFlowEvaluator.getRollingWindowSize());
    }
}
