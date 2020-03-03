package com.ovh.mls.serving.runtime.timeseries;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PredictionIntervalEvaluatorTest {

    private static final ClassLoader LOADER = PredictionIntervalEvaluatorTest.class.getClassLoader();

    PredictionIntervalEvaluator createPI() throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        PredictionIntervalEvaluatorManifest predictionIntervalEvaluatorManifest = objectMapper.readValue(
            LOADER.getResourceAsStream("timeseries/prediction-interval-manifest.json"),
            PredictionIntervalEvaluatorManifest.class
        );
        return PredictionIntervalEvaluator.create(predictionIntervalEvaluatorManifest, "");
    }

    PredictionIntervalEvaluator createPartialPI() throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        PredictionIntervalEvaluatorManifest predictionIntervalEvaluatorManifest = objectMapper.readValue(
            LOADER.getResourceAsStream("timeseries/prediction-interval-manifest-partial.json"),
            PredictionIntervalEvaluatorManifest.class
        );
        return PredictionIntervalEvaluator.create(predictionIntervalEvaluatorManifest, "");
    }

    @Test
    void evaluate() throws EvaluationException, IOException {
        PredictionIntervalEvaluator predictionIntervalEvaluator = createPI();
        TensorIO input = new TensorIO(Map.of("label", Tensor.fromDoubleData(36.2)));
        TensorIO output = predictionIntervalEvaluator.evaluate(input, new EvaluationContext());

        assertEquals(1, output.getBatchSize());
        assertEquals(
            Set.of("label_quantile_inf", "label_quantile_sup"),
            output.tensorsNames()
        );
        assertEquals(
            36.0,
            (double) output.getTensor("label_quantile_inf").getData(),
            0.1
        );
        assertEquals(
            36.4,
            (double) output.getTensor("label_quantile_sup").getData(),
            0.1
        );
    }

    @Test
    void evaluateMissingColumn() throws IOException, EvaluationException {
        PredictionIntervalEvaluator predictionIntervalEvaluator = createPI();

        TensorIO input = new TensorIO(Map.of("label", Tensor.fromData(DataType.DOUBLE, new Double[]{null, 36.2})));
        TensorIO output = predictionIntervalEvaluator.evaluate(input, new EvaluationContext());

        assertEquals(2, output.getBatchSize());
        assertEquals(
            Set.of("label_quantile_inf", "label_quantile_sup"),
            output.tensorsNames()
        );

        Double[] outputInfData = (Double[]) output.getTensor("label_quantile_inf").getData();
        Double[] outputSupData = (Double[]) output.getTensor("label_quantile_sup").getData();


        assertNull(outputInfData[0]);
        assertNull(outputSupData[0]);

        assertEquals(36.0, outputInfData[1], 0.1);
        assertEquals(36.4, outputSupData[1], 0.1);
    }

    @Test
    void evaluateMissingValue() throws IOException {
        PredictionIntervalEvaluator predictionIntervalEvaluator = createPI();
        TensorIO input = new TensorIO(Map.of("label1", Tensor.fromDoubleData(36.2)));
        assertThrows(EvaluationException.class,
            () -> predictionIntervalEvaluator.evaluate(input, new EvaluationContext()));
    }

    @Test
    void evaluatePartial() throws EvaluationException, IOException {
        PredictionIntervalEvaluator predictionIntervalEvaluator = createPartialPI();
        TensorIO input = new TensorIO(Map.of(
            "label", Tensor.fromDoubleData(36.2),
            "label_t+1", Tensor.fromDoubleData(42.1)
        ));

        TensorIO output = predictionIntervalEvaluator.evaluate(input, new EvaluationContext());

        assertEquals(1, output.getBatchSize());
        assertEquals(
            Set.of(
                "label_quantile_inf",
                "label_quantile_sup",
                "label_t+1_quantile_sup"
            ),
            output.tensorsNames()
        );
        assertEquals(
            36.0,
            (double) output.getTensor("label_quantile_inf").getData(),
            0.1
        );
        assertEquals(
            36.4,
            (double) output.getTensor("label_quantile_sup").getData(),
            0.1
        );
        assertEquals(
            42.6,
            (double) output.getTensor("label_t+1_quantile_sup").getData(),
            0.1
        );
    }

    @Test
    public void getInputs() throws IOException {
        PredictionIntervalEvaluator predictionIntervalEvaluator = createPI();
        List<Field> inputs = predictionIntervalEvaluator.getInputs();
        assertEquals(1, inputs.size());
        assertEquals("label", inputs.get(0).getName());
        assertEquals(DataType.DOUBLE, inputs.get(0).getType());
    }

    @Test
    public void getInputsPartial() throws IOException {
        PredictionIntervalEvaluator predictionIntervalEvaluator = createPartialPI();
        List<Field> inputs = predictionIntervalEvaluator.getInputs();
        assertEquals(2, inputs.size());
        assertEquals("label", inputs.get(0).getName());
        assertEquals(DataType.DOUBLE, inputs.get(0).getType());
        assertEquals("label_t+1", inputs.get(1).getName());
        assertEquals(DataType.DOUBLE, inputs.get(1).getType());
    }

    @Test
    public void getOutputs() throws IOException {
        PredictionIntervalEvaluator predictionIntervalEvaluator = createPI();
        List<Field> outputs = predictionIntervalEvaluator.getOutputs();

        assertEquals(2, outputs.size());

        assertEquals("label_quantile_inf", outputs.get(0).getName());
        assertEquals(DataType.DOUBLE, outputs.get(0).getType());

        assertEquals("label_quantile_sup", outputs.get(1).getName());
        assertEquals(DataType.DOUBLE, outputs.get(1).getType());
    }

    @Test
    public void getOutputsPartial() throws IOException {
        PredictionIntervalEvaluator predictionIntervalEvaluator = createPartialPI();
        List<Field> outputs = predictionIntervalEvaluator.getOutputs();

        assertEquals(3, outputs.size());

        assertEquals("label_quantile_inf", outputs.get(0).getName());
        assertEquals(DataType.DOUBLE, outputs.get(0).getType());

        assertEquals("label_quantile_sup", outputs.get(1).getName());
        assertEquals(DataType.DOUBLE, outputs.get(1).getType());

        assertEquals("label_t+1_quantile_sup", outputs.get(2).getName());
        assertEquals(DataType.DOUBLE, outputs.get(1).getType());
    }

    @Test
    void getBatchSize() throws IOException {
        PredictionIntervalEvaluator predictionIntervalEvaluator = createPI();
        assertEquals(1, predictionIntervalEvaluator.getRollingWindowSize());
    }
}
