package com.ovh.mls.serving.runtime.core.builder;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.core.tensor.TensorIndex;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorIOIntoTensorIOTest {

    @Test
    void testBuildSymplify() {
        List<TensorField> schema = List.of(
            new TensorField(
                "tensor_input",
                DataType.INTEGER,
                new int[]{-1, 2, 1},
                null
            )
        );

        TensorIO input = new TensorIO(
            Map.of(
                "tensor_input",
                new Tensor(
                    DataType.INTEGER,
                    new int[]{3, 2},
                    new int[][]{
                        new int[]{1, 2},
                        new int[]{3, 4},
                        new int[]{5, 6}
                    })
            )
        );

        // All following case should produce the same TensorIO
        TensorIO outputMainSimp = new TensorIOIntoTensorIO(schema, false).build(input);
        TensorIO outputIndexSimp = new TensorIOIntoTensorIO(schema, true).build(input);

        for (TensorIO output : List.of(outputMainSimp, outputIndexSimp)) {
            assertEquals(1, output.getTensors().size());
            assertEquals(DataType.INTEGER, output.getTensor("tensor_input").getType());
            assertArrayEquals(new int[]{3, 2, 1}, output.getTensor("tensor_input").getShape().getArrayShape());
            assertArrayEquals(new int[][][]{
                new int[][]{new int[]{1}, new int[]{2}},
                new int[][]{new int[]{3}, new int[]{4}},
                new int[][]{new int[]{5}, new int[]{6}},
            }, (int[][][]) output.getTensor("tensor_input").getData());
        }

        // All following case should produce the same TensorIO
        TensorIO outputMainNoSimp = new TensorIOIntoTensorIO(schema, false).build(input);
        TensorIO outputIndexNoSimp = new TensorIOIntoTensorIO(schema, true).build(input);

        for (TensorIO output : List.of(outputMainNoSimp, outputIndexNoSimp)) {
            assertEquals(1, output.getTensors().size());
            assertEquals(DataType.INTEGER, output.getTensor("tensor_input").getType());
            assertArrayEquals(new int[]{3, 2, 1}, output.getTensor("tensor_input").getShape().getArrayShape());
            assertArrayEquals(new int[][][]{
                new int[][]{new int[]{1}, new int[]{2}},
                new int[][]{new int[]{3}, new int[]{4}},
                new int[][]{new int[]{5}, new int[]{6}},
            }, (int[][][]) output.getTensor("tensor_input").getData());
        }

    }

    @Test
    void testBuildWithoutIndexes() {
        List<TensorField> schema1 = List.of(
            new TensorField(
                "tensor_input",
                DataType.INTEGER,
                new int[]{-1, 2},
                null
            )
        );
        List<TensorField> schema2 = List.of(
            new TensorField(
                "tensor_input",
                DataType.INTEGER,
                new int[]{-1, 2},
                new ArrayList<>()
            )
        );
        TensorIO input = new TensorIO(
            Map.of(
                "tensor_input",
                new Tensor(
                    DataType.INTEGER,
                    new int[]{3, 2},
                    new int[][]{
                        new int[]{1, 2},
                        new int[]{3, 4},
                        new int[]{5, 6}
                    })
            )
        );

        for (List<TensorField> schema : List.of(schema1, schema2)) {
            // All following case should produce the same TensorIO
            TensorIO outputMainNoSimp = new TensorIOIntoTensorIO(schema, false).build(input);
            TensorIO outputIndexNoSimp = new TensorIOIntoTensorIO(schema, true).build(input);

            for (TensorIO output : List.of(outputMainNoSimp, outputIndexNoSimp)) {
                assertEquals(1, output.getTensors().size());
                assertEquals(DataType.INTEGER, output.getTensor("tensor_input").getType());
                assertArrayEquals(new int[]{3, 2}, output.getTensor("tensor_input").getShape().getArrayShape());
                assertArrayEquals(new int[][]{
                    new int[]{1, 2},
                    new int[]{3, 4},
                    new int[]{5, 6},
                }, (int[][]) output.getTensor("tensor_input").getData());
            }
        }
    }

    @Test
    void testBuildWithRollingWindows() {
        List<TensorField> schema = List.of(
            new TensorField(
                "tensor_input",
                DataType.INTEGER,
                new int[]{-1, 4},
                List.of(
                    new TensorIndex("column1", 0),
                    new TensorIndex("column2", 1)
                )
            )
        );
        TensorIO input1 = new TensorIO(
            Map.of(
                "tensor_input",
                new Tensor(
                    DataType.INTEGER,
                    new int[]{3, 2},
                    new int[][]{
                        new int[]{1, 2},
                        new int[]{3, 4},
                        new int[]{5, 6}
                    })
            )
        );
        TensorIO input2 = new TensorIO(
            Map.of(
                "column1", new Tensor(DataType.INTEGER, new int[]{3}, new int[]{1, 3, 5}),
                "column2", new Tensor(DataType.INTEGER, new int[]{3}, new int[]{2, 4, 6})
            )
        );

        // Trying to build main

        // All following case should produce the same TensorIO
        TensorIO output1NoSimp = new TensorIOIntoTensorIO(schema, false, 2).build(input1);
        TensorIO output2NoSimp = new TensorIOIntoTensorIO(schema, false, 2).build(input2);

        for (TensorIO output : List.of(output1NoSimp, output2NoSimp)) {
            assertEquals(1, output.getTensors().size());
            assertEquals(DataType.INTEGER, output.getTensor("tensor_input").getType());
            assertArrayEquals(new int[]{2, 4}, output.getTensor("tensor_input").getShape().getArrayShape());
            assertArrayEquals(new int[][]{
                new int[]{1, 2, 3, 4},
                new int[]{3, 4, 5, 6},
            }, (int[][]) output.getTensor("tensor_input").getData());
        }
    }

    @Test
    void testBuildWithSpecifiedIndexes() {
        List<TensorField> schema = List.of(
            new TensorField(
                "tensor_input",
                DataType.INTEGER,
                new int[]{-1, 2},
                List.of(
                    new TensorIndex("column1", 0),
                    new TensorIndex("column2", 1)
                )
            )
        );
        TensorIO input1 = new TensorIO(
            Map.of(
                "tensor_input",
                new Tensor(
                    DataType.INTEGER,
                    new int[]{3, 2},
                    new int[][]{
                        new int[]{1, 2},
                        new int[]{3, 4},
                        new int[]{5, 6}
                })
            )
        );
        TensorIO input2 = new TensorIO(
            Map.of(
                "column1", new Tensor(DataType.INTEGER, new int[]{3}, new int[]{1, 3, 5}),
                "column2", new Tensor(DataType.INTEGER, new int[]{3}, new int[]{2, 4, 6})
            )
        );

        // Trying to build indexes

        // All following case should produce the same TensorIO
        TensorIO output1NoSimp = new TensorIOIntoTensorIO(schema, true).build(input1);
        TensorIO output2NoSimp = new TensorIOIntoTensorIO(schema, true).build(input2);

        for (TensorIO output : List.of(output1NoSimp, output2NoSimp)) {
            assertEquals(2, output.getTensors().size());
            assertEquals(DataType.INTEGER, output.getTensor("column1").getType());
            assertEquals(DataType.INTEGER, output.getTensor("column2").getType());

            assertArrayEquals(new int[]{3}, output.getTensor("column1").getShape().getArrayShape());
            assertArrayEquals(new int[]{3}, output.getTensor("column2").getShape().getArrayShape());

            assertArrayEquals(new int[]{1, 3, 5}, (int[]) output.getTensor("column1").getData());
            assertArrayEquals(new int[]{2, 4, 6}, (int[]) output.getTensor("column2").getData());
        }

        // Trying to build main

        // All following case should produce the same TensorIO
        output1NoSimp = new TensorIOIntoTensorIO(schema, false).build(input1);
        output2NoSimp = new TensorIOIntoTensorIO(schema, false).build(input2);

        for (TensorIO output : List.of(output1NoSimp, output2NoSimp)) {
            assertEquals(1, output.getTensors().size());
            assertEquals(DataType.INTEGER, output.getTensor("tensor_input").getType());
            assertArrayEquals(new int[]{3, 2}, output.getTensor("tensor_input").getShape().getArrayShape());
            assertArrayEquals(new int[][]{
                new int[]{1, 2},
                new int[]{3, 4},
                new int[]{5, 6},
            }, (int[][]) output.getTensor("tensor_input").getData());
        }
    }

}
