package com.ovh.mls.serving.runtime.tensorflow;

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
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class H5Test {
    private static final ClassLoader LOADER = H5Test.class.getClassLoader();
    private static final Config config = ConfigFactory.load().getConfig("evaluator");
    private static final EvaluatorUtil MAPPER = new EvaluatorUtil(config);

    private static TensorflowEvaluator tensorFlowEvaluator;

    @BeforeAll
    public static void create() throws EvaluatorException {
        tensorFlowEvaluator = (TensorflowEvaluator) new TensorflowH5Generator()
            .generate(new File(LOADER.getResource("tensorflow/test_h5/test.h5").getFile()), config);
    }

    @Test
    public void evaluate() throws EvaluationException {
        InputStreamJsonIntoTensorIO builder = new InputStreamJsonIntoTensorIO(MAPPER.getObjectMapper());
        TensorIO input = builder.build(LOADER.getResourceAsStream("tensorflow/test_h5/single_gen.json"));

        final EvaluationContext evaluationContext = new EvaluationContext();
        TensorIO output = tensorFlowEvaluator.evaluate(input, evaluationContext);

        assertEquals(1, evaluationContext.totalEvaluation());
        assertEquals(Set.of("prediction"), output.tensorsNames());

        Tensor tensor = output.getTensor("prediction");
        TensorShape shape = tensor.getShape();
        assertArrayEquals(new int[]{1, 3}, shape.getArrayShape());
        assertEquals(DataType.FLOAT, tensor.getType());
        assertEquals(5.613032953988295E-7, ((float[][]) tensor.getData())[0][0], 0.00000001);
        assertEquals(7.821146864444017E-4, ((float[][]) tensor.getData())[0][1], 0.00000001);
        assertEquals(0.9992172718048096, ((float[][]) tensor.getData())[0][2], 0.00000001);
    }

    @Test
    public void getInputs() {
        List<Field> inputs = tensorFlowEvaluator.getInputs();
        assertEquals(1, inputs.size());
        assertEquals("inputs", inputs.get(0).getName());
        assertEquals(DataType.FLOAT, inputs.get(0).getType());
    }

    @Test
    public void getOutputs() {
        List<Field> outputs = tensorFlowEvaluator.getOutputs();

        assertEquals(1, outputs.size());

        assertEquals("prediction", outputs.get(0).getName());
        assertEquals(DataType.FLOAT, outputs.get(0).getType());
    }

    @Test
    public void getBatchSize() {
        assertEquals(1, tensorFlowEvaluator.getRollingWindowSize());
    }
}
