package com.ovh.mls.serving.runtime.core.tensor;

import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TensorIndexIteratorTest {

    @Test
    public void testIterator() throws IOException {
        TensorShape shape = new TensorShape(new int[]{2, 3, 4});
        TensorIndexIterator iterator = new TensorIndexIterator(shape);
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 0, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 0, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 0, 2}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 0, 3}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 1, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 1, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 1, 2}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 1, 3}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 2, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 2, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 2, 2}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 2, 3}, iterator.next());

        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 0, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 0, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 0, 2}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 0, 3}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 1, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 1, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 1, 2}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 1, 3}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 2, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 2, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 2, 2}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 2, 3}, iterator.next());

        assertFalse(iterator.hasNext());
    }

    @Test
    public void testIteratorWithRollingWindow1() throws IOException {
        TensorShape shape = new TensorShape(new int[]{4, 2});
        int indexRollingWindows = 0;
        int rollingWindowSize = 2;

        TensorIndexIterator iterator = new TensorIndexIterator(shape, indexRollingWindows, rollingWindowSize);
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 1}, iterator.next());

        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{2, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{2, 1}, iterator.next());

        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{2, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{2, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{3, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{3, 1}, iterator.next());

        assertFalse(iterator.hasNext());
    }

    @Test
    public void testIteratorWithRollingWindow2() throws IOException {
        TensorShape shape = new TensorShape(new int[]{2, 4});
        int indexRollingWindows = 1;
        int rollingWindowSize = 2;

        TensorIndexIterator iterator = new TensorIndexIterator(shape, indexRollingWindows, rollingWindowSize);
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 2}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 2}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{0, 3}, iterator.next());

        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 0}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 1}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 2}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 2}, iterator.next());
        assertTrue(iterator.hasNext());
        assertArrayEquals(new int[]{1, 3}, iterator.next());

        assertFalse(iterator.hasNext());
    }

}
