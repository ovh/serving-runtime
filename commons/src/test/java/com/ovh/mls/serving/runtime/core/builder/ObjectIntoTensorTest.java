package com.ovh.mls.serving.runtime.core.builder;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ObjectIntoTensorTest {

    @Test
    void testBuildWrongTensorInteger() {
        Object tensorObj = List.of(
            List.of(
                List.of(1, 2),
                List.of(3)
            ),
            List.of(
                List.of(1, 2)
            )
        );

        EvaluationException exception = assertThrows(
            EvaluationException.class,
            () ->  new ObjectIntoTensor().build(tensorObj)
        );
        assertEquals(
            "Impossible to convert object into tensor :" +
            " Dimension number 2 is expected to be of size 2 but found 1 instead",
            exception.getMessage()
        );
    }

    @Test
    void testBuildTensorIntegerWithNull() {
        Object tensorObj = List.of(
            List.of(
                Arrays.asList(null, 2),
                Arrays.asList(3, null)
            ),
            List.of(
                Arrays.asList(null, 2),
                Arrays.asList(3, 4)
            )
        );

        Tensor tensor = new ObjectIntoTensor().build(tensorObj);
        assertEquals(DataType.INTEGER, tensor.getType());
        assertArrayEquals(new int[]{2, 2, 2}, tensor.getShapeAsArray());
        assertTrue(tensor.isNullable());
        assertArrayEquals(new Integer[][][]{
            new Integer[][]{
                new Integer[]{null, 2},
                new Integer[]{3, null}
            },
            new Integer[][]{
                new Integer[]{null, 2},
                new Integer[]{3, 4}
            }
        }, (Integer[][][]) tensor.getData());
    }

    @Test
    void testBuildTensorWithDifferentNumbers1() {
        Object tensorObj = List.of(
            List.of(
                Arrays.asList(null, 1),
                Arrays.asList(2F, null)
            ),
            List.of(
                Arrays.asList(null, 2F),
                Arrays.asList(3L, 3D)
            )
        );

        Tensor tensor = new ObjectIntoTensor().build(tensorObj);
        assertEquals(DataType.DOUBLE, tensor.getType());
        assertArrayEquals(new int[]{2, 2, 2}, tensor.getShapeAsArray());
        assertTrue(tensor.isNullable());
        assertArrayEquals(new Double[][][]{
            new Double[][]{
                new Double[]{null, 1.0},
                new Double[]{2.0, null}
            },
            new Double[][]{
                new Double[]{null, 2.0},
                new Double[]{3.0, 3.0}
            }
        }, (Double[][][]) tensor.getData());
    }

    @Test
    void testBuildWTFTensor() {
        Object tensorObj = List.of(
            List.of(
                Arrays.asList(null, 1),
                Arrays.asList(1, "toto")
            ),
            List.of(
                Arrays.asList(true, 2),
                Arrays.asList(3, 3)
            )
        );

        EvaluationException exception = assertThrows(
            EvaluationException.class,
            () -> new ObjectIntoTensor().build(tensorObj)
        );
        assertEquals(
            "Impossible to find type of tensor : several where found [boolean, integer, string]",
            exception.getMessage()
        );
    }

    @Test
    void testBuildWTFTensor2() {
        Object tensorObj = List.of(
            List.of(
                new ArrayList<>(),
                new ArrayList<>()
            ),
            List.of(
                new ArrayList<>(),
                new ArrayList<>()
            )
        );

        EvaluationException exception = assertThrows(
            EvaluationException.class,
            () -> new ObjectIntoTensor().build(tensorObj)
        );
        assertEquals(
            "Impossible to convert object into tensor : Impossible to find type of tensor : no one where found",
            exception.getMessage()
        );
    }

    @Test
    void testBuildTensorInteger() {
        Object tensorObj = List.of(
            List.of(
                List.of(1, 2),
                List.of(3, 4)
            ),
            List.of(
                List.of(1, 2),
                List.of(3, 4)
            )
        );

        Tensor tensor = new ObjectIntoTensor().build(tensorObj);
        assertEquals(DataType.INTEGER, tensor.getType());
        assertArrayEquals(new int[]{2, 2, 2}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertArrayEquals(new int[][][]{
            new int[][]{
                new int[]{1, 2},
                new int[]{3, 4}
            },
            new int[][]{
                new int[]{1, 2},
                new int[]{3, 4}
            }
        }, (int[][][]) tensor.getData());
    }

    @Test
    void testBuildTensorLong() {
        Object tensorObj = List.of(
            List.of(
                List.of(1L, 2L),
                List.of(3L, 4L)
            ),
            List.of(
                List.of(1L, 2L),
                List.of(3L, 4L)
            )
        );

        Tensor tensor = new ObjectIntoTensor().build(tensorObj);
        assertEquals(DataType.LONG, tensor.getType());
        assertArrayEquals(new int[]{2, 2, 2}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertArrayEquals(new long[][][]{
            new long[][]{
                new long[]{1, 2},
                new long[]{3, 4}
            },
            new long[][]{
                new long[]{1, 2},
                new long[]{3, 4}
            }
        }, (long[][][]) tensor.getData());
    }

    @Test
    void testBuildTensorFloat() {
        Object tensorObj = List.of(
            List.of(
                List.of(1F, 2F),
                List.of(3F, 4F)
            ),
            List.of(
                List.of(1F, 2F),
                List.of(3F, 4F)
            )
        );

        Tensor tensor = new ObjectIntoTensor().build(tensorObj);
        assertEquals(DataType.FLOAT, tensor.getType());
        assertArrayEquals(new int[]{2, 2, 2}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertArrayEquals(new float[][][]{
            new float[][]{
                new float[]{1, 2},
                new float[]{3, 4}
            },
            new float[][]{
                new float[]{1, 2},
                new float[]{3, 4}
            }
        }, (float[][][]) tensor.getData());
    }

    @Test
    void testBuildTensorDouble() {
        Object tensorObj = List.of(
            List.of(
                List.of(1D, 2D),
                List.of(3D, 4D)
            ),
            List.of(
                List.of(1D, 2D),
                List.of(3D, 4D)
            )
        );

        Tensor tensor = new ObjectIntoTensor().build(tensorObj);
        assertEquals(DataType.DOUBLE, tensor.getType());
        assertArrayEquals(new int[]{2, 2, 2}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertArrayEquals(new double[][][]{
            new double[][]{
                new double[]{1, 2},
                new double[]{3, 4}
            },
            new double[][]{
                new double[]{1, 2},
                new double[]{3, 4}
            }
        }, (double[][][]) tensor.getData());
    }

    @Test
    void testBuildTensorBoolean() {
        Object tensorObj = List.of(
            List.of(
                List.of(true, false),
                List.of(false, true)
            ),
            List.of(
                List.of(true, false),
                List.of(false, true)
            )
        );

        Tensor tensor = new ObjectIntoTensor().build(tensorObj);
        assertEquals(DataType.BOOLEAN, tensor.getType());
        assertArrayEquals(new int[]{2, 2, 2}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertArrayEquals(new boolean[][][]{
            new boolean[][]{
                new boolean[]{true, false},
                new boolean[]{false, true}
            },
            new boolean[][]{
                new boolean[]{true, false},
                new boolean[]{false, true}
            }
        }, (boolean[][][]) tensor.getData());
    }

    @Test
    void testBuildTensorString() {
        Object tensorObj = List.of(
            List.of(
                List.of("toto", "perdu"),
                List.of("perdu", "toto")
            ),
            List.of(
                List.of("toto", "perdu"),
                List.of("perdu", "toto")
            )
        );

        Tensor tensor = new ObjectIntoTensor().build(tensorObj);
        assertEquals(DataType.STRING, tensor.getType());
        assertArrayEquals(new int[]{2, 2, 2}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertArrayEquals(new String[][][]{
            new String[][]{
                new String[]{"toto", "perdu"},
                new String[]{"perdu", "toto"}
            },
            new String[][]{
                new String[]{"toto", "perdu"},
                new String[]{"perdu", "toto"}
            }
        }, (String[][][]) tensor.getData());
    }

    @Test
    void testBuildScalarInteger() {
        Tensor tensor = new ObjectIntoTensor().build(1);
        assertEquals(DataType.INTEGER, tensor.getType());
        assertArrayEquals(new int[]{}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertEquals(1, (int) tensor.getData());
    }

    @Test
    void testBuildScalarLong() {
        Tensor tensor = new ObjectIntoTensor().build(1L);
        assertEquals(DataType.LONG, tensor.getType());
        assertArrayEquals(new int[]{}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertEquals(1, (long) tensor.getData());
    }

    @Test
    void testBuildScalarFloat() {
        Tensor tensor = new ObjectIntoTensor().build(1.0F);
        assertEquals(DataType.FLOAT, tensor.getType());
        assertArrayEquals(new int[]{}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertEquals(1, (float) tensor.getData());
    }

    @Test
    void testBuildScalarDouble() {
        Tensor tensor = new ObjectIntoTensor().build(1.0D);
        assertEquals(DataType.DOUBLE, tensor.getType());
        assertArrayEquals(new int[]{}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertEquals(1, (double) tensor.getData());
    }

    @Test
    void testBuildScalarBoolean() {
        Tensor tensor = new ObjectIntoTensor().build(true);
        assertEquals(DataType.BOOLEAN, tensor.getType());
        assertArrayEquals(new int[]{}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertEquals(true, tensor.getData());
    }

    @Test
    void testBuildScalarString() {
        Tensor tensor = new ObjectIntoTensor().build("toto");
        assertEquals(DataType.STRING, tensor.getType());
        assertArrayEquals(new int[]{}, tensor.getShapeAsArray());
        assertFalse(tensor.isNullable());
        assertEquals("toto", tensor.getData());
    }

    @Test
    void testBuildScalarNull() {
        Tensor tensor = new ObjectIntoTensor().build(null);
        assertNull(tensor.getType());
        assertArrayEquals(new int[]{}, tensor.getShapeAsArray());
        assertTrue(tensor.isNullable());
        assertNull(tensor.getData());
    }
}
