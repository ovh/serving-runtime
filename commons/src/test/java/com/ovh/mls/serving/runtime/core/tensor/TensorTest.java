package com.ovh.mls.serving.runtime.core.tensor;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorTest {

    @Test
    public void testSetTensorCoord() {
        TensorShape shape = new TensorShape(new int[] {3, 2});
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, shape);

        assertArrayEquals(
            new int[][] {
                new int[] {0, 0},
                new int[] {0, 0},
                new int[] {0, 0},
            },
            (int[][]) tensor.getData()
        );

        tensor.setOnCoord(1, 2, 0);

        assertArrayEquals(
            new int[][] {
                new int[] {0, 0},
                new int[] {0, 0},
                new int[] {1, 0},
            },
            (int[][]) tensor.getData()
        );
    }

    @Test
    void testSetTensorCoordScalar() {
        TensorShape shape = new TensorShape(new int[] {});
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, shape);
        assertTrue(tensor.isScalar());
        assertEquals(0, tensor.getData());
        tensor.setOnCoord(1);
        assertEquals(1, tensor.getData());
    }

    @Test
    void testReshapeIntoScalar() {
        TensorShape shape = new TensorShape(new int[] {1});
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, shape, new int[] {1});
        assertFalse(tensor.isScalar());
        assertTrue(tensor.isVector());
        assertArrayEquals(new int[] {1}, (int[]) tensor.getData());

        Tensor tensorReshaped = tensor.reshape(new int[] {});
        assertTrue(tensorReshaped.isScalar());
        assertFalse(tensorReshaped.isVector());
        assertEquals(1, tensorReshaped.getData());

        Tensor tensorReReshaped = tensorReshaped.reshape(new int[] {1});
        assertFalse(tensorReReshaped.isScalar());
        assertTrue(tensorReReshaped.isVector());
        assertArrayEquals(new int[] {1}, (int[]) tensorReReshaped.getData());
    }

    @Test
    void testErrorSetTensor1() {
        Tensor tensor = new Tensor(DataType.INTEGER, new int[] {3, 2, 4});
        EvaluationException exception = assertThrows(
            EvaluationException.class,
            () -> tensor.setOnCoord(4, 0, 0, -1)
        );
        assertEquals("Coordinates contained at least an undefined value : [0, 0, -1]", exception.getMessage());
    }

    @Test
    void testErrorSetTensor2() {
        Tensor tensor = new Tensor(DataType.INTEGER, new int[] {3, 2, 4});
        EvaluationException exception = assertThrows(
            EvaluationException.class,
            () -> tensor.setOnCoord(new int[] {4}, 0, 0, -1)
        );
        assertEquals("Expected an array of length 4 to be compatible with undefined coordinate, found 1",
            exception.getMessage());
    }

    @Test
    void testErrorSetTensor3() {
        Tensor tensor = new Tensor(DataType.INTEGER, new int[] {3, 2, 4});
        EvaluationException exception = assertThrows(
            EvaluationException.class,
            () -> tensor.setOnCoord(new int[] {4, 4, 4, 4}, 0, -1, -1)
        );
        assertEquals("Only one undefined coordinate is supported, found 2", exception.getMessage());
    }

    @Test
    public void testSetTensorCoordWithArray() {
        Tensor tensor = new Tensor(DataType.INTEGER, new int[] {3, 2});
        tensor.setOnCoord(new int[] {1, 2, 3}, -1, 0);
        assertArrayEquals(
            new int[][] {
                new int[] {1, 0},
                new int[] {2, 0},
                new int[] {3, 0},
            },
            (int[][]) tensor.getData()
        );
        tensor.setOnCoord(new int[] {0, 0, 0}, -1, 0);
        assertArrayEquals(
            new int[][] {
                new int[] {0, 0},
                new int[] {0, 0},
                new int[] {0, 0},
            },
            (int[][]) tensor.getData()
        );
        tensor.setOnCoord(new int[] {1, 2}, 1, -1);
        assertArrayEquals(
            new int[][] {
                new int[] {0, 0},
                new int[] {1, 2},
                new int[] {0, 0},
            },
            (int[][]) tensor.getData()
        );
    }

    @Test
    public void testReshape1() {
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, new int[] {3, 2}, new int[][] {
            new int[] {1, 2},
            new int[] {3, 4},
            new int[] {5, 6},
        });

        Tensor reshaped = tensor.reshape(new int[] {2, 3});

        assertArrayEquals(
            new int[][] {
                new int[] {1, 2, 3},
                new int[] {4, 5, 6},
            },
            (int[][]) reshaped.getData()
        );

        assertEquals(type, reshaped.getType());
        assertArrayEquals(new int[] {2, 3}, reshaped.getShape().getArrayShape());
    }

    @Test
    public void testReshape2() {
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, new int[] {3, 1}, new int[][] {
            new int[] {1},
            new int[] {2},
            new int[] {3},
        });

        Tensor reshaped = tensor.reshape(new int[] {3});

        assertArrayEquals(
            new int[] {1, 2, 3},
            (int[]) reshaped.getData()
        );
    }

    @Test
    public void testReshape3() {
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, new int[] {3}, new int[] {1, 2, 3});
        Tensor reshaped = tensor.reshape(new int[] {3, 1});

        assertArrayEquals(
            new int[][] {
                new int[] {1},
                new int[] {2},
                new int[] {3},
            },
            (int[][]) reshaped.getData()
        );
    }

    @Test
    public void testReshape4() {
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, new int[] {3}, new int[] {1, 2, 3});
        Tensor reshaped = tensor.reshape(new int[] {-1, 1});

        assertArrayEquals(
            new int[][] {
                new int[] {1},
                new int[] {2},
                new int[] {3},
            },
            (int[][]) reshaped.getData()
        );
    }

    @Test
    public void testGetCoord() {
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, new int[] {3, 2}, new int[][] {
            new int[] {1, 2},
            new int[] {3, 4},
            new int[] {5, 6},
        });

        assertEquals(1, tensor.getCoord(0, 0));
        assertEquals(2, tensor.getCoord(0, 1));
        assertEquals(3, tensor.getCoord(1, 0));
        assertEquals(4, tensor.getCoord(1, 1));
        assertEquals(5, tensor.getCoord(2, 0));
        assertEquals(6, tensor.getCoord(2, 1));
    }

    @Test
    public void testGetCoord2() {
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, new int[] {3, 2}, new int[][] {
            new int[] {1, 2},
            new int[] {3, 4},
            new int[] {5, 6},
        });

        assertArrayEquals(new int[] {1, 3, 5}, (int[]) tensor.getCoord(-1, 0));
        assertArrayEquals(new int[] {2, 4, 6}, (int[]) tensor.getCoord(-1, 1));
        assertArrayEquals(new int[] {1, 2}, (int[]) tensor.getCoord(0, -1));
        assertArrayEquals(new int[] {3, 4}, (int[]) tensor.getCoord(1, -1));
        assertArrayEquals(new int[] {5, 6}, (int[]) tensor.getCoord(2, -1));
    }

    @Test
    public void testGetCoord3() {
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, new int[] {2, 3, 4}, new int[][][] {
            new int[][] {
                new int[] {1, 2, 3, 4},
                new int[] {5, 6, 7, 8},
                new int[] {9, 10, 11, 12}
            },
            new int[][] {
                new int[] {13, 14, 15, 16},
                new int[] {17, 18, 19, 20},
                new int[] {21, 22, 23, 24}
            },
        });

        assertArrayEquals(new int[] {1, 13}, (int[]) tensor.getCoord(-1, 0, 0));
        assertArrayEquals(new int[] {1, 5, 9}, (int[]) tensor.getCoord(0, -1, 0));
        assertArrayEquals(new int[] {1, 2, 3, 4}, (int[]) tensor.getCoord(0, 0, -1));
    }

    @Test
    public void testRoll1() {
        Tensor input = Tensor.fromIntData(new int[][] {
            new int[] {1, 2},
            new int[] {3, 4},
            new int[] {5, 6},
            new int[] {7, 8},
        });

        Tensor output = input.roll(0, 2);
        assertArrayEquals(new int[] {3, 2, 2}, output.getShape().getArrayShape());
        assertArrayEquals(new int[][][] {
            new int[][] {
                new int[] {1, 2},
                new int[] {3, 4}
            },
            new int[][] {
                new int[] {3, 4},
                new int[] {5, 6}
            },
            new int[][] {
                new int[] {5, 6},
                new int[] {7, 8}
            },
        }, (int[][][]) output.getData());
    }

    @Test
    public void testRoll2() {
        Tensor input = Tensor.fromIntData(new int[][] {
            new int[] {1, 2, 3, 4},
            new int[] {5, 6, 7, 8},
        });

        Tensor output = input.roll(1, 2);
        assertArrayEquals(new int[] {2, 3, 2}, output.getShape().getArrayShape());
        assertArrayEquals(new int[][][] {
            new int[][] {
                new int[] {1, 2},
                new int[] {2, 3},
                new int[] {3, 4},
            },
            new int[][] {
                new int[] {5, 6},
                new int[] {6, 7},
                new int[] {7, 8},
            },
        }, (int[][][]) output.getData());
    }

    @Test
    public void slice() {
        DataType type = DataType.INTEGER;
        Tensor tensor = new Tensor(type, new int[] {3, 2, 3}, new int[][][] {
            new int[][] {new int[] {1, 1, 1}, new int[] {2, 2, 2}},
            new int[][] {new int[] {3, 3, 3}, new int[] {4, 4, 4}},
            new int[][] {new int[] {5, 5, 5}, new int[] {6, 6, 6}}
        });

        Tensor slice = tensor.slice(new TensorShape(new int[] {1, 0, 0}), new TensorShape(new int[] {1, 1, 3}));
        Tensor expected = new Tensor(type, new int[] {1, 1, 1}, new int[][][] {
            new int[][] {new int[] {3, 3, 3}}
        });
        assertArrayEquals((int[][][]) expected.getData(), (int[][][]) slice.getData());

        slice = tensor.slice(new TensorShape(new int[] {1, 0, 0}), new TensorShape(new int[] {1, 2, 3}));
        expected = new Tensor(type, new int[] {1, 1, 1}, new int[][][] {
            new int[][] {new int[] {3, 3, 3}, new int[] {4, 4, 4}}
        });
        assertArrayEquals((int[][][]) expected.getData(), (int[][][]) slice.getData());

        slice = tensor.slice(new TensorShape(new int[] {1, 0, 0}), new TensorShape(new int[] {2, 1, 3}));
        expected = new Tensor(type, new int[] {1, 1, 1}, new int[][][] {
            new int[][] {new int[] {3, 3, 3}},
            new int[][] {new int[] {5, 5, 5}}
        });
        assertArrayEquals((int[][][]) expected.getData(), (int[][][]) slice.getData());

        tensor = new Tensor(type, new int[] {6}, new int[] {1, 2, 3, 4, 5, 6});
        slice = tensor.slice(new TensorShape(new int[] {2}), new TensorShape(new int[] {1}));
        expected = new Tensor(type, new int[] {1}, new int[] {3});
        assertArrayEquals((int[]) expected.getData(), (int[]) slice.getData());
    }

    @Test
    public void concat() {
        DataType type = DataType.INTEGER;
        Tensor tensor1 = new Tensor(type, new int[] {2, 3}, new int[][] {
            new int[] {1, 2, 3},
            new int[] {4, 5, 6}
        });
        Tensor tensor2 = new Tensor(type, new int[] {2, 3}, new int[][] {
            new int[] {7, 8, 9},
            new int[] {10, 11, 12}
        });

        Tensor expected = new Tensor(type, new int[] {2, 3}, new int[][] {
            new int[] {1, 2, 3},
            new int[] {4, 5, 6},
            new int[] {7, 8, 9},
            new int[] {10, 11, 12}
        });

        Tensor result = tensor1.concat(tensor2, 0);
        assertArrayEquals((int[][]) expected.getData(), (int[][]) result.getData());

        expected = new Tensor(type, new int[] {2, 3}, new int[][] {
            new int[] {1, 2, 3, 7, 8, 9},
            new int[] {4, 5, 6, 10, 11, 12}
        });

        result = tensor1.concat(tensor2, 1);
        assertArrayEquals((int[][]) expected.getData(), (int[][]) result.getData());

        tensor1 = new Tensor(type, new int[] {}, 1);
        tensor2 = new Tensor(type, new int[] {}, 2);
        expected = new Tensor(type, new int[] {2}, new int[] {1, 2});
        result = tensor1.concat(tensor2, 0);
        assertArrayEquals((int[]) expected.getData(), (int[]) result.getData());

        tensor1 = new Tensor(type, new int[] {2}, new int[] {1, 2});
        tensor2 = new Tensor(type, new int[] {}, 3);
        expected = new Tensor(type, new int[] {2}, new int[] {1, 2, 3});
        result = tensor1.concat(tensor2, 0);
        assertArrayEquals((int[]) expected.getData(), (int[]) result.getData());

        tensor1 = new Tensor(type, new int[] {}, 3);
        tensor2 = new Tensor(type, new int[] {2}, new int[] {1, 2});
        expected = new Tensor(type, new int[] {2}, new int[] {3, 1, 2});
        result = tensor1.concat(tensor2, 0);
        assertArrayEquals((int[]) expected.getData(), (int[]) result.getData());
    }

}
