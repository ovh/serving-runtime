package com.ovh.mls.serving.runtime.core.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class TensorShapeTest {

    @Test
    public void testResolveNewShape4() {
        TensorShape shape = new TensorShape(new int[]{4, 28, 28});

        assertArrayEquals(
            new int[]{4, 28, 28, 1},
            shape.resolveNewShape(new int[]{-1, -1, -1, 1})
        );
        assertArrayEquals(
            new int[]{4, 28, 28},
            shape.resolveNewShape(new int[]{-1, -1, -1})
        );
    }

    @Test
    public void testResolveNewShape3() {
        TensorShape shape = new TensorShape(new int[]{4, 28, 28, 1});

        assertArrayEquals(
            new int[]{4, 28, 28, 1},
            shape.resolveNewShape(new int[]{-1, -1, -1, 1})
        );
        assertArrayEquals(
            new int[]{4, 28, 28},
            shape.resolveNewShape(new int[]{-1, -1, -1})
        );
    }

    @Test
    public void testResolveNewShape2() {
        TensorShape shape = new TensorShape(new int[]{4, 28, 28, 3});

        assertArrayEquals(
            new int[]{4, 28, 28, 3},
            shape.resolveNewShape(new int[]{-1, -1, -1, 3})
        );
    }

    @Test
    public void testResolveNewShape1() {
        TensorShape shape = new TensorShape(new int[]{3, 2});

        assertArrayEquals(
            new int[]{3, 2},
            shape.resolveNewShape(new int[]{-1, 2})
        );
        assertArrayEquals(
            new int[]{3, 2},
            shape.resolveNewShape(new int[]{3, -1})
        );
        assertArrayEquals(
            new int[]{3, 2, 1},
            shape.resolveNewShape(new int[]{3, 2, -1})
        );
        assertArrayEquals(
            new int[]{6, 1},
            shape.resolveNewShape(new int[]{-1, 1})
        );
        assertArrayEquals(
            new int[]{6},
            shape.resolveNewShape(new int[]{-1})
        );
    }

}
