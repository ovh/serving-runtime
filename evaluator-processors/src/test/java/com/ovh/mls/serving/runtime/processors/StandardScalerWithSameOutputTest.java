package com.ovh.mls.serving.runtime.processors;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class StandardScalerWithSameOutputTest {

    private static final ClassLoader LOADER = StandardScalerWithSameOutputTest.class.getClassLoader();

    private StandardScaler standardScaler;

    @BeforeEach
    void setUp() throws IOException, EvaluatorException {
        ObjectMapper objectMapper = new ObjectMapper();
        StandardScalerManifest standardScalerManifest = objectMapper.readValue(
            LOADER.getResourceAsStream("processors/standard-scaler-same-output-manifest.json"),
            StandardScalerManifest.class);
        standardScaler = standardScalerManifest.create("");
    }

    @Test
    void evaluate() throws EvaluationException {

        Tensor tensorValue = Tensor.fromDoubleData(new Double[]{null, 25.8, 47.0});
        Tensor tensorLabel = Tensor.fromDoubleData(new Double[]{null, 50.0, 90.0});

        TensorIO input = new TensorIO(Map.of("value", tensorValue, "label", tensorLabel));

        TensorIO output = standardScaler.evaluate(input, new EvaluationContext());

        assertEquals(3, output.getBatchSize());
        assertEquals(Set.of("label", "value"), output.tensorsNames());
        assertArrayEquals(new Double[]{null, 10.4, 21.0}, (Double[]) output.getTensor("value").getData());
        assertArrayEquals(new Double[]{null, 10.0, 20.0}, (Double[]) output.getTensor("label").getData());
    }

    @Test
    void getInputs() {
        List<Field> inputs = standardScaler.getInputs();
        assertEquals(2, inputs.size());
        assertEquals("value", inputs.get(0).getName());
        Assertions.assertEquals(DataType.DOUBLE, inputs.get(0).getType());
        assertEquals("label", inputs.get(1).getName());
        Assertions.assertEquals(DataType.DOUBLE, inputs.get(1).getType());
    }

    @Test
    void getOutputs() {
        List<Field> outputs = standardScaler.getOutputs();
        assertEquals(2, outputs.size());
        assertEquals("value", outputs.get(0).getName());
        Assertions.assertEquals(DataType.DOUBLE, outputs.get(0).getType());
        assertEquals("label", outputs.get(1).getName());
        Assertions.assertEquals(DataType.DOUBLE, outputs.get(1).getType());
    }
}
