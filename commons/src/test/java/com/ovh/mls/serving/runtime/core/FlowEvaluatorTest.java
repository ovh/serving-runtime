package com.ovh.mls.serving.runtime.core;

import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

@SuppressWarnings("ALL")
public class FlowEvaluatorTest {

    private FlowEvaluatorManifest flowEvaluatorManifest;
    private FlowEvaluator flowEvaluator;

    @BeforeEach
    public void setup() throws EvaluatorException, IOException {
        TestEvaluatorManifest manifest1 = new TestEvaluatorManifest();
        List<Field> inputs1 = Collections.singletonList(new Field("input_1_1", DataType.DOUBLE));
        manifest1.setInputs(inputs1);
        List<Field> outputs1 = Collections.singletonList(new Field("output_1_1", DataType.BOOLEAN));
        manifest1.setOutputs(outputs1);
        manifest1.setBatchSize(3);

        TestEvaluatorManifest manifest2 = new TestEvaluatorManifest();
        List<Field> inputs2 = Arrays.asList(
            new Field("output_1_1", DataType.BOOLEAN),
            new Field("input_1_1", DataType.FLOAT),
            new Field("input_2_1", DataType.DOUBLE)
        );
        manifest2.setInputs(inputs2);
        List<Field> outputs2 = Arrays.asList(
            new Field("output_2_1", DataType.DOUBLE),
            new Field("output_2_2", DataType.DOUBLE)
        );
        manifest2.setOutputs(outputs2);
        manifest2.setBatchSize(2);

        flowEvaluatorManifest = new FlowEvaluatorManifest();
        flowEvaluatorManifest.setEvaluatorManifests(Arrays.asList(manifest1, manifest2));
        flowEvaluatorManifest.setOutputs(Arrays.asList(
            new Field("output_1_1", DataType.BOOLEAN),
            new Field("output_2_1", DataType.DOUBLE)
        ));

        flowEvaluator = flowEvaluatorManifest.create("");
    }

    @Test
    public void illegalOutput() {
        flowEvaluatorManifest.setOutputs(Collections.singletonList(new Field("unknown", DataType.DOUBLE)));
        Assertions.assertThrows(EvaluatorException.class, () ->
            flowEvaluatorManifest.create("")
        );
    }

    @Test
    public void getInputs() {
        List<Field> inputs = flowEvaluator.getInputs();

        Assertions.assertEquals(3, inputs.size());
        Assertions.assertEquals(
            new HashSet<>(Arrays.asList(
                new Field("input_1_1", DataType.DOUBLE),
                new Field("input_1_1", DataType.FLOAT),
                new Field("input_2_1", DataType.DOUBLE)
            )),
            new HashSet<>(inputs)
        );
    }

    @Test
    public void getOutputs() {
        List<Field> outputs = flowEvaluator.getOutputs();

        Assertions.assertEquals(2, outputs.size());
        Assertions.assertEquals(
            new HashSet<>(Arrays.asList(
                new Field("output_1_1", DataType.BOOLEAN),
                new Field("output_2_1", DataType.DOUBLE)
            )),
            new HashSet<>(outputs)
        );
    }

    @Test
    public void getDefaultOutputs() throws EvaluatorException, IOException {
        flowEvaluatorManifest.setOutputs(Collections.emptyList());
        FlowEvaluator flowEvaluator2 = flowEvaluatorManifest.create("");

        List<Field> outputs = flowEvaluator2.getOutputs();

        Assertions.assertEquals(2, outputs.size());
        Assertions.assertEquals(
            new HashSet<>(Arrays.asList(
                new Field("output_2_1", DataType.DOUBLE),
                new Field("output_2_2", DataType.DOUBLE)
            )),
            new HashSet<>(outputs)
        );
    }

    @Test
    public void getBatchSize() {
        Assertions.assertEquals(4, flowEvaluator.getRollingWindowSize());
    }

    @Test
    public void evaluate() throws EvaluationException {
        TensorIO tensorIO = new TensorIO(Map.of(
            "input_1_1",
            Tensor.fromDoubleData(new double[] {3.14, 42.0, 0, 37})));

        TensorIO output = flowEvaluator.evaluate(tensorIO, new EvaluationContext());

        Assertions.assertEquals(2, output.getTensors().size());
        Assertions.assertEquals(Set.of("output_1_1", "output_2_1"), output.getTensors().keySet());
    }

    @Test
    public void evaluateNotEnoughData() {
        TensorIO tensorIO = new TensorIO(Map.of(
            "input_1_1",
            Tensor.fromDoubleData(new double[] {3.14})));

        Assertions.assertThrows(
            EvaluationException.class,
            () -> flowEvaluator.evaluate(tensorIO, new EvaluationContext()));
    }

    public static class TestEvaluatorManifest extends AbstractEvaluatorManifest {

        private int batchSize;

        @Override
        public Evaluator create(String path) {
            return new TestEvaluator(getInputs(), getOutputs(), getBatchSize());
        }

        @Override
        public String getType() {
            return "test";
        }

        public int getBatchSize() {
            return batchSize;
        }

        public TestEvaluatorManifest setBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
    }

    public static class TestEvaluator implements Evaluator {

        private final List<Field> inputs;
        private final List<Field> outputs;
        private final int batchSize;

        public TestEvaluator(List<Field> inputs, List<Field> outputs, int batchSize) {
            this.inputs = inputs;
            this.outputs = outputs;
            this.batchSize = batchSize;
        }

        @Override
        public TensorIO evaluate(TensorIO io, EvaluationContext evaluationContext) {
            Map<String, Tensor> tensors = new HashMap<>();
            for (Field output : getOutputs()) {
                tensors.put(output.getName(), Tensor.fromIntData(new int[]{1, 2}));
            }
            return new TensorIO(tensors);
        }

        @Override
        public List<Field> getInputs() {
            return inputs;
        }

        @Override
        public List<Field> getOutputs() {
            return outputs;
        }

        @Override
        public int getRollingWindowSize() {
            return batchSize;
        }
    }
}
