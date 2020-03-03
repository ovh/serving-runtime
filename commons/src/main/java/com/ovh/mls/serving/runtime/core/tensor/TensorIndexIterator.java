package com.ovh.mls.serving.runtime.core.tensor;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Iterator;

/**
 * Implementation of Iterator<int[]> used to iterate over each index inside a tensor of a given shape
 * Can also manage a rolling window
 */
public class TensorIndexIterator implements Iterator<int[]> {

    private final TensorShape shape;
    private final int indexRollingWindows;
    private final int rollingWindowSize;
    private final int[] nextTensorIndex;

    private boolean hasNext;
    private int startWindowIndex;
    private int offsetWindowIndex;

    /**
     * Constructor of a simple iterator over indexes of a tensor shape
     * @param shape Shape over which iterating
     */
    public TensorIndexIterator(TensorShape shape) {
        this(shape, 0, 1);
    }

    /**
     * Constructor of an iterator over indexes of a tensor shape with a rolling window
     * @param shape Shape over which iterating
     * @param indexRollingWindows Index of the shape on with we want to roll
     * @param rollingWindowSize Size of the rolling window
     */
    public TensorIndexIterator(TensorShape shape, int indexRollingWindows, int rollingWindowSize) {
        this.shape = shape;
        this.indexRollingWindows = indexRollingWindows;
        this.rollingWindowSize = rollingWindowSize;

        nextTensorIndex = new int[shape.getArrayShape().length];
        hasNext = true;
        startWindowIndex = 0;
        offsetWindowIndex = 0;
    }

    @Override
    public boolean hasNext() {
        return this.hasNext;
    }

    @Override
    public int[] next() {
        int[] result = Arrays.copyOf(nextTensorIndex, nextTensorIndex.length);
        int rank = shape.getArrayShape().length - 1;
        if (rank >= 0) {
            // In case it is not a scalar
            this.incRank(rank);
        } else {
            // In case it is a scalar
            this.hasNext = false;
        }
        return result;
    }

    private void incRank(int rank) {

        int nextRankValue;
        // If we try to increase the rank defined as the 'rolling windows' rank
        if (rank == indexRollingWindows) {
            // We increase the offset window index if the rolling window size is not reach
            if (offsetWindowIndex < rollingWindowSize - 1) {
                offsetWindowIndex++;
                nextRankValue = startWindowIndex + offsetWindowIndex;
            // Otherwise we increase the start index of the window and restart the offset window index
            } else {
                startWindowIndex++;
                if (startWindowIndex + offsetWindowIndex < shape.getArrayShape()[rank]) {
                    offsetWindowIndex = 0;
                    nextRankValue = startWindowIndex + offsetWindowIndex;
                } else {
                    startWindowIndex = 0;
                    offsetWindowIndex = 0;
                    // will inc rank
                    nextRankValue = shape.getArrayShape()[rank];
                }
            }

        // Otherwise the next rank value is simply the current rank value + 1
        } else {
            nextRankValue = nextTensorIndex[rank] + 1;
        }

        // If the next rank value is sill the authorized range, set this value
        if (nextRankValue < shape.getArrayShape()[rank]) {
            Array.setInt(nextTensorIndex, rank, nextRankValue);
        // Otherwise set the rank value at 0 and increase previous ranks
        } else {
            Array.setInt(nextTensorIndex, rank, 0);
            if (rank == 0) {
                this.hasNext = false;

            } else {
                incRank(rank - 1);
            }
        }
    }

}
