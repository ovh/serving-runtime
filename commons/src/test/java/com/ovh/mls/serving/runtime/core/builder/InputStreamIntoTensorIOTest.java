package com.ovh.mls.serving.runtime.core.builder;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.io.InputStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class InputStreamIntoTensorIOTest {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Test
    void testDeserializeVectorMultiple() {
        TensorIO tensorIO = from("core/tensor-input/vector-multiple.json");
        Tensor tensor1 = tensorIO.getTensor("key1");

        assertEquals(DataType.INTEGER, tensor1.getType());
        assertArrayEquals(new int[]{1, 2, 3}, (int[]) tensor1.getData());
        assertTrue(tensor1.isVector());

        Tensor tensor2 = tensorIO.getTensor("key2");
        assertEquals(DataType.INTEGER, tensor2.getType());
        assertArrayEquals(new int[]{4, 5, 6}, (int[]) tensor2.getData());
        assertTrue(tensor2.isVector());
    }

    @Test
    void testDeserializeSimpleTensorsWithUnknownDimension() {
        TensorIO tensorIO = from("core/tensor-input/tensor-single.json");

        assertArrayEquals(
            (int[][][]) tensorIO.getTensors().get("tensor_input_1").getData(),
            new int[][][]{
                new int[][]{
                    new int[]{1, 2},
                    new int[]{3, 4}
                },
                new int[][]{
                    new int[]{5, 6},
                    new int[]{7, 8}
                },
                new int[][]{
                    new int[]{9, 10},
                    new int[]{11, 12}
                }
            }
        );
    }

    @Test
    void testDeserializeSimpleTensors() {
        TensorIO tensorIO = from("core/tensor-input/tensor-single.json");

        assertArrayEquals(
            (int[][][]) tensorIO.getTensors().get("tensor_input_1").getData(),
            new int[][][]{
                new int[][]{
                    new int[]{1, 2},
                    new int[]{3, 4}
                },
                new int[][]{
                    new int[]{5, 6},
                    new int[]{7, 8}
                },
                new int[][]{
                    new int[]{9, 10},
                    new int[]{11, 12}
                }
            }
        );
    }

    @Test
    void testDeserializeMultipleTensors() {
        TensorIO tensorIO = from("core/tensor-input/tensor-multiple.json");

        assertArrayEquals(
            (int[][]) tensorIO.getTensors().get("tensor_input_1").getData(),
            new int[][]{
                new int[]{1, 2, 3},
                new int[]{4, 5, 6},
            }
        );
        assertArrayEquals(
            (int[][]) tensorIO.getTensors().get("tensor_input_2").getData(),
            new int[][]{
                new int[]{7, 8},
                new int[]{9, 10},
                new int[]{11, 12},
            }
        );

    }

    private TensorIO from(String fileName) {
        InputStream inputStream = getResourceFileAsIs(fileName);

        return new InputStreamJsonIntoTensorIO(OBJECT_MAPPER).build(inputStream);
    }

    private InputStream getResourceFileAsIs(String fileName) {
        return getClass().getClassLoader().getResourceAsStream(fileName);
    }

}
