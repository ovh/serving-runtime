package com.ovh.mls.serving.runtime.core.tensor;

import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

public class TensorShape {
    protected final int[] shape;

    public TensorShape(int[] shape) {
        this.shape = shape;
    }

//    public TensorShape(String shape) {
//        this.shape = getShapeFromString(shape);
//    }

    public TensorShape(long[] shape) {
        this.shape = Arrays.stream(shape)
            .mapToInt(x -> (int) x)
            .toArray();
    }

//    /**
//     * Extract a shape from a string like (?, 2, 2)
//     */
//    private static int[] getShapeFromString(String stringShape) {
//        String removeParenthesis = stringShape.substring(1, stringShape.length() - 1);
//        String[] split = removeParenthesis.split(",");
//
//        if (removeParenthesis.isEmpty()) {
//            return new int[0];
//        }
//
//        Integer[] result = Arrays
//            .stream(split)
//            .map(String::trim)
//            .map(x -> {
//                if ("?".equals(x)) {
//                    return -1;
//                } else {
//                    return Integer.parseInt(x);
//                }
//            })
//            .toArray(Integer[]::new);
//
//        return ArrayUtils.toPrimitive(result);
//    }

    public boolean handleBatch() {
        return shape.length != 0 && shape[0] == -1;
    }

    public int[] getNewShape(int batchSize, Integer replaceMissingBy) {
        final int[] newShape = Arrays.copyOf(shape, shape.length);

        if (handleBatch()) {
            newShape[0] = batchSize;
        }

        for (int i = 0; i < newShape.length; i++) {
            if (newShape[i] < 0 && replaceMissingBy != null) {
                newShape[i] = replaceMissingBy;
            }
        }

        return newShape;
    }

    public int[] getArrayShape() {
        return this.shape;
    }

    /**
     * Return the current shape but remove all value of '1'
     */
    public TensorShape simplifyShape() {
        return new TensorShape(Arrays
            .stream(this.getArrayShape())
            .filter(x -> x != 1)
            .toArray());
    }

    /**
     * If there is a '-1' value inside the wanted shape, we need to resolve it
     * @return The given shape with resolved '-1' value
     */
    public int[] resolveNewShape(int[] newShape) throws EvaluationException {
        long negativeValues = Arrays.stream(newShape).filter(x -> x < 0).count();
        int[] result = newShape;

        // If only one negative value, infer the missing one
        if (negativeValues == 1) {
            int thisProduct = this.shapeProduct();

            int newShapeProduct = Arrays
                .stream(newShape).filter(x -> x > 0)
                .reduce(1, (x, y) -> x * y);

            int replacedValue = thisProduct / newShapeProduct;

            result = Arrays
                .stream(newShape)
                .map(x -> {
                    if (x < 0) {
                        return replacedValue;
                    } else {
                        return x;
                    }
                })
                .toArray();
        } else if (negativeValues > 1) {
            result = new int[newShape.length];

            for (int i = 0; i < newShape.length; i++) {
                int newShapeDimensionSize = newShape[i];
                int currentShapeDimensionSize = 1;
                if (i < this.shape.length) {
                    currentShapeDimensionSize = this.shape[i];
                }

                if (newShapeDimensionSize < 0) {
                    result[i] = currentShapeDimensionSize;
                } else {
                    if (newShapeDimensionSize != currentShapeDimensionSize) {
                        throw new EvaluationException(
                            String.format(
                                "Unable to resolve shape %s into %s",
                                Arrays.toString(this.getArrayShape()),
                                Arrays.toString(newShape)
                            )
                        );
                    }
                    result[i] = currentShapeDimensionSize;
                }
            }
        }



        if (!isCompatibleWith(result)) {
            throw new EvaluationException(
                String.format(
                    "Unable to resolve shape %s into %s",
                    Arrays.toString(this.getArrayShape()),
                    Arrays.toString(newShape)
                )
            );
        }


        return result;
    }

    /**
     * returning a boolean indicating if the given shape is compatible with the current one (through a reshape or not)
     */
    public boolean isCompatibleWith(int[] otherShape) {
        int thisProduct = this.shapeProduct();
        int otherProduct = new TensorShape(otherShape).shapeProduct();
        return thisProduct == otherProduct;
    }

    /**
     * Return the product of all shape ranks
     */
    public int shapeProduct() {
        return Arrays
            .stream(this.getArrayShape())
            .reduce(1, (x, y) -> x * y);
    }

    public int getRank() {
        return this.getArrayShape().length;
    }

    public boolean isOfRank(int rank) {
        return this.getRank() == rank;
    }

    public boolean isScalarShape() {
        return this.isOfRank(0);
    }

    public boolean isVectorShape() {
        return this.isOfRank(1);
    }

    public boolean isMatrixShape() {
        return this.isOfRank(2);
    }

    public long numberOfUnknownDimensions() {
        return Arrays
            .stream(this.getArrayShape())
            .filter(x -> x < 0)
            .count();
    }


}
