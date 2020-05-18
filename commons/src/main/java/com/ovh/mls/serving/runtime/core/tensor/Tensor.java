package com.ovh.mls.serving.runtime.core.tensor;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;
import java.util.stream.IntStream;

public class Tensor {

    /**
     * The datatype of the tensor
     */
    private DataType type;

    /**
     * Data of the tensor
     * Underlying structure is suppose to be an Array of the current shape with the equivalent java type for datatype
     */
    private Object data;

    /**
     * Shape of the tensor
     */
    private TensorShape shape;

    /**
     * Boolean indicating if the tensor can contains null values
     */
    private final boolean nullableValues;

    public Tensor(DataType type, int[] shape) {
        this(type, new TensorShape(shape), initialData(new TensorShape(shape), type));
    }

    public Tensor(DataType type, TensorShape shape) {
        this(type, shape, initialData(shape, type));
    }

    public Tensor(DataType type, int[] shape, Object data) {
        this(type, new TensorShape(shape), data);
    }

    public Tensor(DataType type, TensorShape shape, Object data) {
        this(type, shape, data, false);
    }

    public Tensor(DataType type, TensorShape shape, Object data, boolean nullableValues) {
        this.type = type;
        this.shape = shape;
        this.data = data;
        this.nullableValues = nullableValues;
        checkNoUnknownDimension();
    }

    /**
     * Accessor of the tensor datatype
     */
    public DataType getType() {
        return type;
    }

    /**
     * Setter of the datatype
     */
    public void setType(DataType type) {
        this.type = type;
    }

    /**
     * Accessor of data array
     */
    public Object getData() {
        return data;
    }

    /**
     * Setter for data array
     */
    public void setData(Object data) {
        this.data = data;
    }

    /**
     * Accessor of the shape
     */
    public TensorShape getShape() {
        return shape;
    }

    /**
     * Accessor of the shape as a java array
     */
    public int[] getShapeAsArray() {
        return this.shape.getArrayShape();
    }

    /**
     * Setter of the shape
     */
    public void setShape(TensorShape shape) {
        this.shape = shape;
        checkNoUnknownDimension();
    }

    /**
     * Accessor of the nullable boolean
     */
    public boolean isNullable() {
        return this.nullableValues;
    }

    /**
     * Extract a new tensor of specified size from the original tensor
     */
    public Tensor slice(TensorShape begin, TensorShape size) {
        if (this.shape.getRank() != begin.getRank() || this.shape.getRank() != size.getRank()) {
            throw new IllegalArgumentException("Provided shapes are not compatible with the tensor shape");
        }

        // check valid sizes
        for (int i = 0; i < this.shape.getRank(); i++) {
            int baseDim = this.shape.shape[i];
            int beginDim = begin.shape[i];
            if (beginDim >= baseDim) {
                throw new IllegalArgumentException(
                    String.format("Begin dimension at rank %d is out of bound (%d >= %d).", i, beginDim, baseDim)
                );
            }
            int sizeDim = size.shape[i];
            if (beginDim + sizeDim > baseDim) {
                throw new IllegalArgumentException(
                    String.format(
                        "Desired size at rank %d is out of bounds (%d + %d >= %d)", i, beginDim, sizeDim, baseDim
                    )
                );
            }

        }
        Object result = getSlice(this.data, begin.shape, size.shape);
        return Tensor.fromData(this.getType(), result);
    }

    private Object getSlice(Object data, int[] begin, int[] size) {
        if (size.length == 1) {
            Object result = Array.newInstance(this.getType().getJavaClass(), size);
            for (int i = begin[0]; i < begin[0] + size[0]; i++) {
                Array.set(result, i - begin[0], Array.get(data, i));
            }
            return result;
        } else if (size.length > 1) {
            Object result = Array.newInstance(this.getType().getJavaClass(), size);
            for (int i = begin[0]; i < begin[0] + size[0]; i++) {
                Array.set(
                    result,
                    i - begin[0],
                    getSlice(
                        Array.get(data, i),
                        Arrays.copyOfRange(begin, 1, begin.length),
                        Arrays.copyOfRange(size, 1, size.length)
                    )
                );
            }
            return result;
        } else {
            throw new IllegalArgumentException("Cannot slice if size is empty");
        }
    }

    /**
     *  Concatenate the new tensor with the current one into a new Tensor
     */
    public Tensor concat(Tensor value, int axis) {
        // handle scalars
        if (this.isScalar()) {
            return this.reshape(new int[] {1}).concat(value, axis);
        }
        if (value.isScalar()) {
            value = value.reshape(new int[] {1});
        }

        // all dimensions should be equal but on the axis
        for (int i = 0; i < this.getRank(); i++) {
            if (i == axis) {
                continue;
            }
            if (this.getShapeAsArray()[i] != value.getShapeAsArray()[i]) {
                throw new IllegalArgumentException("Except for the axis dimension, all other should be equal");
            }
        }

        Object concat;
        if (axis >= this.shape.getRank()) {
            throw new IllegalArgumentException("Axis out of bounds");
        } else {
            concat = concat(
                this.getData(),
                value.getData(),
                this.getShapeAsArray(),
                value.getShapeAsArray()[axis],
                axis
            );
        }
        return fromData(this.getType(), concat);
    }

    private Object concat(Object data, Object data1, int[] baseShape, int concatDim, int axis) {
        int[] newShape = baseShape.clone();
        newShape[axis] += concatDim;
        Object result = Array.newInstance(this.getType().getJavaClass(), newShape);

        if (axis == 0) {
            for (int i = 0; i < baseShape[0]; i++) {
                Array.set(result, i, Array.get(data, i));
            }
            for (int i = 0; i < concatDim; i++) {
                Array.set(result, baseShape[0] + i, Array.get(data1, i));
            }
        } else {
            for (int i = 0; i < baseShape[0]; i++) {
                Array.set(
                    result,
                    i,
                    concat(
                        Array.get(data, i),
                        Array.get(data1, i),
                        Arrays.copyOfRange(baseShape, 1, baseShape.length), concatDim, axis - 1)
                );
            }
        }
        return result;
    }
    /**
     * Access data on this tensor from coordinates
     */
    public Object getCoord(int... coords) {
        int[] undefinedLength = checkNoMoreThanOneUndefinedCoords(coords);
        if (undefinedLength.length == 0) {
            Object last = this.data;
            for (int coord : coords) {
                last = Array.get(last, coord);
            }
            return convertIfString(last);
        } else {
            Object result = Array.newInstance(this.getType().getJavaClass(), undefinedLength[0]);
            for (int i = 0; i < undefinedLength[0]; i++) {
                Object last = this.data;
                for (int coord : coords) {
                    if (coord < 0) {
                        coord = i;
                    }
                    last = Array.get(last, coord);
                }
                last = convertIfString(last);
                Array.set(result, i, last);
            }
            return result;
        }
    }

    /**
     * If this tensor is of type STRING, data can be stored as byte[], this method do the converstion if needed
     */
    private Object convertIfString(Object obj) {
        if (this.getType() == DataType.STRING && obj instanceof byte[]) {
            return new String((byte[]) obj);
        } else {
            return obj;
        }
    }

    /**
     * Set the given value of the wanted coordinates of this tensor
     *
     * @param value  The wanted value to set, this can be an array if the coordinates contain a single '-1' value
     * @param coords The coordinates
     */
    public void setOnCoord(Object value, int... coords) {
        checkCoordsLength(coords);
        if (value != null) {
            if (this.isScalar()) {
                this.data = value;
            } else if (value.getClass().isArray()) {
                this.setArrayOnCoord(value, coords);
            } else {
                this.setSingleOnCoord(value, coords);
            }
        }
    }

    /**
     * Reshape this tensor data into a simple vector
     */
    public Tensor toVector() {
        return this.reshape(new int[] {-1});
    }

    /**
     * Iterator all the indexes of the current tensor
     *
     * @return the iterator
     */
    public Iterator<int[]> coordIterator() {
        return new TensorIndexIterator(this.shape);
    }

    /**
     * Iterator all the indexes of the current tensor with a rolling window
     *
     * @return the iterator
     */
    public Iterator<int[]> coordIterator(int indexRollingWindows, int rollingWindowSize) {
        return new TensorIndexIterator(this.shape, indexRollingWindows, rollingWindowSize);
    }

    /**
     * Reshape the current tensor to the given new shape and current datatype
     *
     * @param newShape The wanted new shape as TensorShape
     * @return The reshaped tensor
     * @throws EvaluationException: if the new shape is not compatible with the previous one
     */
    public Tensor reshape(TensorShape newShape) {
        return this.reshape(newShape.getArrayShape());
    }

    /**
     * Reshape the current tensor to the given new shape and current datatype
     *
     * @param newShape The wanted new shape as int array
     * @return The reshaped tensor
     * @throws EvaluationException: if the new shape is not compatible with the previous one
     */
    public Tensor reshape(int[] newShape) {
        return reshapeWithType(newShape, this.type);
    }

    /**
     * Reshape the current tensor to the given new shape and given datatype
     *
     * @param newShape The wanted new shape as TensorShape
     * @param dataType The wanted datatype
     * @return The reshaped tensor
     * @throws EvaluationException: if the new shape is not compatible with the previous one
     */
    public Tensor reshapeWithType(TensorShape newShape, DataType dataType) {
        return this.reshapeWithType(newShape.getArrayShape(), dataType);
    }

    /**
     * Reshape the current tensor to the given new shape and given datatype
     *
     * @param newShape The wanted new shape as int array
     * @param dataType The wanted datatype
     * @return The reshaped tensor
     * @throws EvaluationException: if the new shape is not compatible with the previous one
     */
    public Tensor reshapeWithType(int[] newShape, DataType dataType) {
        // Resolve the final wanted shape (removing the -1 values in the wanted shape)
        int[] resolvedNewShape = this.getShape().resolveNewShape(newShape);

        // If the resolved shape and the current shape are the same => doing nothing
        if (Arrays.equals(resolvedNewShape, this.getShape().getArrayShape()) && dataType == this.getType()) {
            return this;
        }

        Tensor output = new Tensor(dataType, resolvedNewShape);

        Iterator<int[]> thisIter = this.coordIterator();
        Iterator<int[]> newIter = output.coordIterator();

        while (thisIter.hasNext() && newIter.hasNext()) {
            int[] thisCoord = thisIter.next();
            int[] newCoord = newIter.next();

            Object thisValue = getCoord(thisCoord);
            Object convertedvalue = dataType.convert(thisValue);
            output.setOnCoord(convertedvalue, newCoord);
        }
        return output;
    }

    /**
     * Get the rank os the tensor
     */
    public int getRank() {
        return this.getShape().getRank();
    }

    /**
     * Return true if this tensor is a scalar (rank = 0)
     */
    public boolean isScalar() {
        return this.getShape().isScalarShape();
    }

    /**
     * Return true if this tensor is a vector (rank = 1)
     */
    public boolean isVector() {
        return this.getShape().isVectorShape();
    }

    /**
     * Return true if this tensor is a vector (rank = 2)
     */
    public boolean isMatrix() {
        return this.getShape().isMatrixShape();
    }

    /**
     * Test if the tensor is of the given rank
     */
    public boolean isOfRank(int rank) {
        return this.getShape().isOfRank(rank);
    }

    /**
     * Set an object (suppose to not be an array value) on the given coordinates of the current tensor
     *
     * @param value  The value we want to set
     * @param coords The coordinates of that value
     */
    private void setSingleOnCoord(Object value, int... coords) {
        checkCoordsDefined(coords);

        Object last = this.data;
        for (int i = 0; i < coords.length; i++) {
            int coord = coords[i];
            boolean isLast = i == coords.length - 1;
            if (isLast) {
                Array.set(last, coord, value);
            } else {
                last = Array.get(last, coord);
            }
        }
    }

    /**
     * Set the given value inside current tensor at the given coordinates
     *
     * @param value  the value that we want to set
     * @param coords the wanted coordinates
     */
    private void setArrayOnCoord(Object value, int... coords) {
        int arrayLength = checkUndefinedCoordsCompatible(value, coords);
        for (int i = 0; i < arrayLength; i++) {

            Object last = this.data;
            for (int j = 0; j < coords.length; j++) {
                int coord = coords[j];
                if (coord < 0) {
                    coord = i;
                }

                boolean isLast = j == coords.length - 1;
                if (isLast) {
                    Object valueToSet = Array.get(value, i);
                    Array.set(last, coord, valueToSet);
                } else {
                    last = Array.get(last, coord);
                }
            }

        }
    }

    /**
     * Check that the given datatype is compatible with the current one, throw an exception otherwise
     *
     * @param wantedType the wanted datatype
     */
    public void checkTypeCompatible(DataType wantedType) {

        if (this.getType() != wantedType &&
            (!DataType.isNumberType(this.getType()) || !DataType.isNumberType(wantedType))) {

            throw new EvaluationException(
                String.format(
                    "Incompatibilies in types for tensor : required (%s) but found (%s)",
                    this.getType().toString(),
                    wantedType
                )
            );
        }
    }

    /**
     * Check that given coords doesn't contains any negative value (i.e. they are all defined)
     *
     * @param coords given coords
     */
    private static void checkCoordsDefined(int... coords) {
        if (Arrays.stream(coords).filter(x -> x < 0).count() > 0) {
            throw new EvaluationException(
                String.format(
                    "Coordinates contained at least an undefined value : %s",
                    Arrays.toString(coords)
                )
            );
        }
    }

    /**
     * Check that given coords are of the correct size (i.e same as the shape size)
     *
     * @param coords
     */
    private void checkCoordsLength(int... coords) {
        if (coords.length != this.getShape().getArrayShape().length) {
            throw new EvaluationException(
                String.format(
                    "The given coordinates should be of size %s, given %s with %s",
                    this.getShape().getArrayShape().length,
                    coords.length,
                    Arrays.toString(coords)
                )
            );
        }
    }

    /**
     * Check that there is at max 1 undefined value (i.e negative value) in given coords
     */
    private int[] checkNoMoreThanOneUndefinedCoords(int... coords) {
        int[] undefinedLength = findUndefinedLength(coords);

        if (undefinedLength.length > 1) {
            throw new EvaluationException(
                String.format(
                    "Only one undefined coordinate is supported, found %s",
                    undefinedLength.length
                )
            );
        }

        return undefinedLength;
    }

    /**
     * Find shape length of indexes on which there are undefined values for the given coordinates
     */
    private int[] findUndefinedLength(int... coords) {
        return IntStream
            .range(0, coords.length)
            .map(i -> {
                if (coords[i] >= 0) {
                    return 0;
                } else {
                    return this.getShape().getArrayShape()[i];
                }
            })
            .filter(x -> x != 0)
            .toArray();
    }

    private int checkUndefinedCoordsCompatible(Object value, int... coords) {
        int[] undefinedLength = findUndefinedLength(coords);

        if (undefinedLength.length != 1) {
            throw new EvaluationException(
                String.format(
                    "Only one undefined coordinate is supported, found %s",
                    undefinedLength.length
                )
            );
        }

        int givenLength = Array.getLength(value);
        if (givenLength != undefinedLength[0]) {
            throw new EvaluationException(
                String.format(
                    "Expected an array of length %s to be compatible with undefined coordinate, found %s",
                    undefinedLength[0],
                    givenLength
                )
            );
        }

        return givenLength;
    }

    /**
     * Access the tensor data converted to list
     */
    public Object getDataAsList() {
        return layerToList(this.getData());
    }

    /**
     * Convert the given layer of data from array into list
     */
    private static Object layerToList(Object layer) {
        if (layer != null && layer.getClass().isArray()) {
            int arrayLength = Array.getLength(layer);
            List<Object> list = new ArrayList<>(arrayLength);
            for (int i = 0; i < arrayLength; i++) {
                Object arrayElt = Array.get(layer, i);
                list.add(layerToList(arrayElt));
            }
            return list;
        } else {
            return layer;
        }
    }

    /**
     * Simplify the shape of the tensor by removing dimensions of size 1
     */
    public Tensor simplifyShape() {
        TensorShape symplifiedShape = this.getShape().simplifyShape();
        return this.reshape(symplifiedShape);
    }

    /**
     * Create initial data for the tensor (array or scalar value)
     */
    private static Object initialData(TensorShape shape, DataType type) {
        if (shape.isScalarShape()) {
            return Array.get(Array.newInstance(type.getJavaClass(), 1), 0);
        } else {
            return Array.newInstance(type.getJavaClass(), shape.getArrayShape());
        }
    }

    /**
     * Check that there is no unknown dimension
     */
    private void checkNoUnknownDimension() {
        long unknownDim = this.getShape().numberOfUnknownDimensions();
        if (unknownDim > 0) {
            throw new EvaluationException(
                String.format(
                    "Impossible to create a tensor with unknown dimensions, found %s",
                    unknownDim
                )
            );
        }
    }

    /**
     * Find the shape of the given data
     */
    private static int[] findShapeFromData(Object data) {
        List<Integer> shapeAsList = new ArrayList<>();
        Object last = data;
        while (last != null && last.getClass() != String.class && last.getClass().isArray()) {
            int length = Array.getLength(last);
            shapeAsList.add(length);
            last = Array.get(last, 0);
        }
        return shapeAsList.stream().mapToInt(x -> x).toArray();
    }

    /**
     * Create a Tensor from its data and datatype
     */
    public static Tensor fromData(DataType type, Object data) {
        int[] guessedShape = findShapeFromData(data);
        return new Tensor(type, new TensorShape(guessedShape), data);
    }

    /**
     * Apply a rolling windows on the wanted shape index
     */
    public Tensor roll(int shapeIndex, int windowsSize) {
        int[] currentShape = this.getShape().getArrayShape();
        int currentShapeLength = currentShape.length;

        // Compute the shape of the rolled tensor
        int[] resolvedNewShape = new int[currentShapeLength + 1];
        for (int i = 0; i < currentShapeLength; i++) {
            if (i < shapeIndex) {
                resolvedNewShape[i] = currentShape[i];
            } else if (i > shapeIndex) {
                resolvedNewShape[i + 1] = currentShape[i];
            } else {
                resolvedNewShape[i] = (currentShape[i] - windowsSize) + 1;
                resolvedNewShape[i + 1] = windowsSize;
            }
        }

        // Create the rolled tensor while iterating over window
        Tensor output = new Tensor(this.type, resolvedNewShape);

        Iterator<int[]> thisIter = this.coordIterator(shapeIndex, windowsSize);
        Iterator<int[]> newIter = output.coordIterator();

        while (thisIter.hasNext() && newIter.hasNext()) {
            int[] thisCoord = thisIter.next();
            int[] newCoord = newIter.next();

            Object thisValue = getCoord(thisCoord);
            output.setOnCoord(thisValue, newCoord);
        }
        return output;
    }

    /**
     * Create a Tensor from INTEGER data
     */
    public static Tensor fromIntData(Object data) {
        return fromData(DataType.INTEGER, data);
    }

    /**
     * Create a Tensor from STRING data
     */
    public static Tensor fromStringData(Object data) {
        return fromData(DataType.STRING, data);
    }

    /**
     * Create a Tensor from LONG data
     */
    public static Tensor fromLongData(Object data) {
        return fromData(DataType.LONG, data);
    }

    /**
     * Create a Tensor from FLOAT data
     */
    public static Tensor fromFloatData(Object data) {
        return fromData(DataType.FLOAT, data);
    }

    /**
     * Create a Tensor from DOUBLE data
     */
    public static Tensor fromDoubleData(Object data) {
        return fromData(DataType.DOUBLE, data);
    }

    /**
     * Create a Tensor from BOOLEAN data
     */
    public static Tensor fromBooleanData(Object data) {
        return fromData(DataType.BOOLEAN, data);
    }

    public Object jsonData(boolean simplify) {
        if (simplify) {
            return this.simplifyShape().getData();
        } else {
            return this.getData();
        }
    }

    public <T> Tensor apply(Function<Object, T> func, DataType outputDataType) {
        Tensor outputTensor = new Tensor(outputDataType, this.shape);
        this.coordIterator().forEachRemaining(coords -> {
            Object input = this.getCoord(coords);
            T output = func.apply(input);
            outputTensor.setOnCoord(output, coords);
        });
        return outputTensor;
    }
}
