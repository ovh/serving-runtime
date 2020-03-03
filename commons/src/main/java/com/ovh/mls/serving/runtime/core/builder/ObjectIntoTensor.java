package com.ovh.mls.serving.runtime.core.builder;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorShape;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import org.apache.commons.math3.util.Pair;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Builder for converting an Object
 * (suppose to be a scalar or List of [scalar or List of [scalar or List of etc...]]) into Tensor
 */
public class ObjectIntoTensor implements Builder<Object, Tensor> {

    @Override
    public Tensor build(Object input) throws EvaluationException {
        // Find the shape of the given tensor
        final int[] shapeArray = findShape(input);
        final TensorShape shape = new TensorShape(shapeArray);

        // Find the datatype + nullable of the given tensor
        final Pair<DataType, Boolean> typeAndNullable = findDataType(input);
        final DataType type = typeAndNullable.getKey();
        final Boolean nullable = typeAndNullable.getValue();

        try {
            Object tensorData;
            // If it is a scalar, just use the given object as tensor data
            if (shape.isScalarShape()) {
                tensorData = input;
            // If the data type is null, can't go further..
            } else if (type == null) {
                throw new EvaluationException("Impossible to find type of tensor : no one where found");
            // Otherwise compute the data for the tensor
            } else {
                tensorData = computeTensorData(input, 0, shapeArray, type, nullable);
            }
            return new Tensor(type, shape, tensorData, nullable);

        } catch (EvaluationException e) {
            throw new EvaluationException(
                String.format("Impossible to convert object into tensor : %s", e.getMessage()),
                e
            );
        }
    }

    /**
     * Find the datatype of a tensor object
     * Return the tuple2 of (data type, nullable)
     */
    private static Pair<DataType, Boolean> findDataType(Object tensor) {
        // Start by flattening the input tensor and get all contained data types
        Set<DataType> set = flattenObject(tensor)
            .stream()
            .map(x -> {
                if (x == null) {
                    return null;
                } else {
                    return DataType.fromClass(x.getClass());
                }
            })
            .collect(Collectors.toSet());

        // Remember if the object contains null value
        Boolean nullable = set.contains(null);
        if (nullable) {
            set.remove(null);
        }

        // If we didn't find any datatype, return null as datatype
        if (set.isEmpty()) {
            return new Pair<>(null, nullable);

        // If we find only 1 datatype, return this datatype
        } else if (set.size() == 1) {
            return new Pair<>(set.iterator().next(), nullable);

        // If all found datatype are numbers, return the more generic
        } else if (set.stream().allMatch(DataType::isNumberType)) {
            // If there is only numbers, find the correct data type to use
            DataType type = List.of(DataType.DOUBLE, DataType.FLOAT, DataType.LONG)
                .stream()
                .filter(set::contains)
                .findFirst()
                .orElse(DataType.INTEGER);
            return new Pair<>(type, nullable);

        // If we found several datatype, throw an exception
        } else {
            List<DataType> sortedList = new ArrayList<>(set);
            sortedList.sort(Comparator.comparing(DataType::toString));

            throw new EvaluationException(
                String.format(
                    "Impossible to find type of tensor : several where found %s",
                    Arrays.toString(sortedList.toArray())
                )
            );
        }
    }

    /**
     * Flatten the given tensor object
     */
    private static List<Object> flattenObject(Object tensor) {
        List<Object> result = new ArrayList<>();
        if (tensor instanceof List) {
            for (Object obj : (List) tensor) {
                List<Object> flatLayer = flattenObject(obj);
                result.addAll(flatLayer);
            }
        } else {
            result.add(tensor);
        }
        return result;
    }

    /**
     * Find the shape of a tensor from its simple Object representation
     */
    private static int[] findShape(Object tensor) {
        if (tensor instanceof List && !((List) tensor).isEmpty()) {
            // If tensor is a List => It is a tensor layer, we need to recursively find shapes
            List<Object> tensorLayer = (List) tensor;
            int currentLayerLength = tensorLayer.size();
            int[] nextShape = findShape(tensorLayer.get(0));

            // Construct the final shape
            int[] result = new int[nextShape.length + 1];
            result[0] = currentLayerLength;
            System.arraycopy(nextShape, 0, result, 1, result.length - 1);
            return result;
        } else {
            // If tensor is not a List => it is a scalar, we return the shape for a scalar : []
            return new int[]{};
        }
    }

    /**
     * Convert the given Object into List<Object> or throw exception
     */
    private static List<Object> toListOrFail(Object tensor) throws EvaluationException {
        if (!(tensor instanceof List)) {
            throw new EvaluationException(
                String.format(
                    "Unable to deserialize tensor : all tensor layers should be of type list. Found %s",
                    tensor.getClass().toString()
                )
            );
        }

        return (List<Object>) tensor;
    }

    /**
     * Compute a tensor object representation into tensor data from its shape, index, type an nullable
     * @param tensor The input tensor object representation
     * @param layerIndex the wanted shape's index that we want
     * @param shape the shape of the tensor
     * @param tensorType the data type of the tensor
     * @param nullable can the tensor contains nullable value ?
     * @return The tensor data (scalar or Array of [scalar or Array of [scalar or Array of etc...]])
     * @throws EvaluationException
     */
    private Object computeTensorData(
        Object tensor,
        int layerIndex,
        int[] shape,
        DataType tensorType,
        boolean nullable
    ) throws EvaluationException {

        // Convert the input tensor object representation into List
        List<Object> layer = toListOrFail(tensor);

        // Check if the expected size is the same than the found size
        int expectedDimensionSize = shape[layerIndex];
        if (expectedDimensionSize > 0 && layer.size() != expectedDimensionSize) {
            throw new EvaluationException(
                String.format(
                    "Dimension number %s is expected to be of size %s but found %s instead",
                    layerIndex,
                    expectedDimensionSize,
                    layer.size()
                )
            );
        }

        // Find the shape of the layer
        int[] layerShape = getShapeForLayer(layerIndex, shape);

        // Find the java class needed for creating an array of the tensor
        Class<?> javaClass;
        if (nullable) {
            javaClass = tensorType.getNullableJavaClass();
        } else {
            javaClass = tensorType.getJavaClass();
        }

        // Create output array from java class and shape
        Object tensorArray = Array.newInstance(javaClass, layerShape);

        // Fill the output array with data
        for (int i = 0; i < layer.size(); i++) {
            Object nextLayer = layer.get(i);
            int nextLayerIndex = layerIndex + 1;
            // In case next layer is the last one
            if (nextLayerIndex == shape.length) {
                Object conversion = tensorType.convert(nextLayer);
                Array.set(tensorArray, i, conversion);
            } else {
                Object nextLayerArray = computeTensorData(nextLayer, nextLayerIndex, shape, tensorType, nullable);
                Array.set(tensorArray, i, nextLayerArray);
            }
        }

        return tensorArray;
    }

    /**
     * Get the underlying shape at the given index for an input shape
     *
     * Example with an input shape of [2, 4, 5]:
     * - Underlying shape at index 0 is [2, 4, 5]
     * - Underlying shape at index 1 is [4, 5]
     * - Underlying shape at index 2 is [5]
     */
    private static int[] getShapeForLayer(int index, int[] shape) {
        int layerShapeLength = shape.length - index;
        int[] layerShape = new int[layerShapeLength];
        System.arraycopy(shape, index, layerShape, 0, layerShapeLength);
        return layerShape;
    }
}
