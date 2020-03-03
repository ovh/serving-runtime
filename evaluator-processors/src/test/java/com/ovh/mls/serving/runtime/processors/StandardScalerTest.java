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

class StandardScalerTest {

    private static final ClassLoader LOADER = StandardScalerTest.class.getClassLoader();

    private StandardScaler standardScaler;

    @BeforeEach
    void setUp() throws IOException, EvaluatorException {
        ObjectMapper objectMapper = new ObjectMapper();
        StandardScalerManifest standardScalerManifest = objectMapper.readValue(
            LOADER.getResourceAsStream("processors/standard-scaler-manifest.json"),
            StandardScalerManifest.class);
        standardScaler = standardScalerManifest.create("");
    }

    @Test
    void evaluate() throws EvaluationException {
        Tensor tensor = Tensor.fromData(DataType.FLOAT, new Float[]{null, 17.0F, 27.5F});
        TensorIO input = new TensorIO(Map.of("value", tensor));

        TensorIO io = standardScaler.evaluate(input, new EvaluationContext());

        assertEquals(3, io.getBatchSize());
        assertEquals(Set.of("scaled_value"), io.tensorsNames());
        assertArrayEquals(new Double[]{null, 5.0, 10.25},
            (Double[]) io.getTensor("scaled_value").getData());

        tensor = Tensor.fromData(DataType.INTEGER, new Integer[]{null, 17, 27});
        input = new TensorIO(Map.of("value", tensor));
        io = standardScaler.evaluate(input, new EvaluationContext());

        assertEquals(3, io.getBatchSize());
        assertEquals(Set.of("scaled_value"), io.tensorsNames());
        assertArrayEquals(new Double[]{null, 5.0, 10.0},
            (Double[]) io.getTensor("scaled_value").getData());
    }

    @Test
    void getInputs() {
        List<Field> inputs = standardScaler.getInputs();
        assertEquals(1, inputs.size());
        assertEquals("value", inputs.get(0).getName());
        Assertions.assertEquals(DataType.DOUBLE, inputs.get(0).getType());
    }

    @Test
    void getOutputs() {
        List<Field> outputs = standardScaler.getOutputs();
        assertEquals(1, outputs.size());
        assertEquals("scaled_value", outputs.get(0).getName());
        assertEquals(DataType.DOUBLE, outputs.get(0).getType());
    }
}
