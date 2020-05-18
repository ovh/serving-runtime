package com.ovh.mls.serving.runtime.core.builder;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.core.tensor.TensorIndex;
import com.ovh.mls.serving.runtime.core.tensor.TensorShape;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Builder that convert a TensorIO into another TensorIO with a different format
 */
public class TensorIOIntoTensorIO implements Builder<TensorIO, TensorIO> {

    /**
     * List of tensor fields to use for transformation
     */
    private final List<? extends TensorField> tensorFields;

    /**
     * Indicates if the output tensors should be the main tensor or the computed index for that tensor
     */
    private final boolean buildIndexes;

    /**
     * Indicates the rolling windows size to apply on tensor if any
     */
    private final int rollingWindowsSize;

    public TensorIOIntoTensorIO(List<? extends TensorField> tensorFields) {
        this(tensorFields, false, 1);
    }

    public TensorIOIntoTensorIO(List<? extends TensorField> tensorFields, boolean buildIndexes) {
        this(tensorFields, buildIndexes, 1);
    }

    public TensorIOIntoTensorIO(
        List<? extends TensorField> tensorFields,
        boolean buildIndexes,
        int rollingWindowsSize) {

        this.tensorFields = tensorFields;
        this.buildIndexes = buildIndexes;
        this.rollingWindowsSize = rollingWindowsSize;
    }

    @Override
    public TensorIO build(TensorIO input) throws EvaluationException {
        Map<String, Tensor> outputMap;
        // If indexes are asked => we build indexes describe by manifest fields
        if (this.buildIndexes) {
            outputMap = this.buildIndexes(input);
        // Otherwise we build the main tensor (i.e. not indexes) described by manifest fields
        } else {
            outputMap = this.buildMain(input);
        }

        return new TensorIO(outputMap);
    }

    /**
     * Build indexes of tensor fields described in manifest (if no indexes are describe then return main tensors)
     * @param input The input TensorIO
     * @return The Map<String, Tensor> of tensor indexes
     * @throws EvaluationException
     */
    private Map<String, Tensor> buildIndexes(TensorIO input) throws EvaluationException {
        Map<String, Tensor> outputMap = new HashMap<>();

        for (TensorField tensorField : this.tensorFields) {
            String tensorName = tensorField.getName();
            DataType wantedTensorType = tensorField.getType();
            Tensor mainTensor = input.getTensor(tensorName);

            // Check if indexes already present in input
            List<TensorIndex> indexesFields = Optional.ofNullable(tensorField.getFields()).orElse(new ArrayList<>());
            long foundIndexes = indexesFields
                .stream()
                .map(Field::getName)
                .flatMap(x -> Optional.ofNullable(input.getTensor(x)).stream())
                .count();

            // If we were unable to get the main tensor or all of its indexes => fail
            if (mainTensor == null && !indexesFields.isEmpty() && foundIndexes != indexesFields.size()) {
                throw new EvaluationException(String.format(
                    "Unable to find either a tensor with name %s or all its indexes",
                    tensorName
                ));
            }

            // If we were unable to get the main tensor => build indexes from indexes
            if (mainTensor == null) {
                for (TensorIndex indexField : indexesFields) {
                    String indexName = indexField.getName();
                    outputMap.put(indexName, input.getTensor(indexName));
                }

            // Otherwise build indexes from main tensor
            } else {
                // If there is no indexes describe in tensor field, we just return the main tensor
                if (indexesFields.isEmpty()) {
                    Tensor outputTensor = transformTensor(mainTensor, tensorField);
                    outputMap.put(tensorField.getName(), outputTensor);

                // Otherwise convert main tensor into list of indexes tensor
                } else {
                    // For now only tensor of rank < 2 can be converted into indexes
                    if (mainTensor.getRank() > 2) {
                        throw new EvaluationException(String.format(
                                "Impossible to convert a tensor of rank more than 2 into indexes. Found rank %s",
                                mainTensor.getRank()
                        ));
                    }
                    for (TensorIndex tensorIndex : tensorField.getFields()) {
                        Integer index = tensorIndex.getIndex();
                        Object indexTensorData;
                        if (index == null) {
                            indexTensorData = mainTensor.getCoord(-1);
                        } else {
                            indexTensorData = mainTensor.getCoord(-1, index);
                        }

                        Tensor indexTensor = new Tensor(
                            wantedTensorType,
                            new int[]{Array.getLength(indexTensorData)},
                            indexTensorData
                        );
                        outputMap.put(tensorIndex.getName(), indexTensor);
                    }
                }
            }
        }

        return outputMap;
    }

    /**
     * Build main tensors (i.e not indexes) described as tensor fields
     * If shapes or types of main tensors differ from given shapes and types, transformation will try a reshaping
     *
     * @param input The input TensorIO
     * @return The Map<String, Tensor> of tensor indexes
     * @throws EvaluationException
     */
    private Map<String, Tensor> buildMain(TensorIO input) throws EvaluationException {
        Map<String, Tensor> outputMap = new HashMap<>();

        // Iterate over tensor fields
        for (TensorField tensorField : this.tensorFields) {
            String tensorName = tensorField.getName();
            Tensor outputTensor = input.getTensor(tensorName);

            // If main tensor name doesn't exist in input => check if we can build it from indexes
            if (outputTensor == null && !tensorField.getFields().isEmpty()) {
                List<TensorIndex> tensorIndexList = tensorField.getFields();

                // Iterate over indexes
                for (TensorIndex tensorIndexField : tensorIndexList) {
                    String indexName = tensorIndexField.getName();
                    Tensor indexTensor = input.getTensor(indexName);

                    // If indexes can't be found, we can build the main tensor
                    if (indexTensor == null) {
                        throw new EvaluationException(String.format(
                            "Unable to find either a tensor with name %s or with name %s", tensorName, indexName
                        ));
                    }

                    // If we have a scalar, reshape it into vector of size 1
                    if (indexTensor.isScalar()) {
                        indexTensor = indexTensor.reshape(new int[]{1});
                    }

                    // For now only tensor indexes of rank 1 can be converted into main tensor
                    if (!indexTensor.isVector()) {
                        throw new EvaluationException(String.format(
                            "Only vectors (i.e tensor of rank 1) are supported as indexes. " +
                                "Found a tensor of rank %s for index %s",
                            indexTensor.getRank(), indexName
                        ));
                    }

                    Object indexTensorData = indexTensor.getData();
                    int vectorLength = Array.getLength(indexTensorData);

                    // If main tensor has not been created yet, build his base from indexes info
                    if (outputTensor == null) {
                        outputTensor = new Tensor(indexTensor.getType(),
                            new int[]{vectorLength, tensorIndexList.size()}
                        );
                    }

                    // Checking that all indexes are of the same expected size, fail otherwise
                    if (outputTensor.getShape().getArrayShape()[0] != vectorLength) {
                        throw new EvaluationException(
                            "When using tensor indexes all vectors should be of the same size; but found different ones"
                        );
                    }

                    // Set data of this indexes on the main tensor
                    outputTensor.setOnCoord(indexTensorData, -1, tensorIndexField.getIndex());
                }
            }

            // If main tensor still null => impossible to build it
            if (outputTensor == null) {
                throw new EvaluationException(String.format("Unable to find a tensor with name %s", tensorName));
            }

            outputTensor = transformTensor(outputTensor, tensorField);

            // Add the output tensor in the result map
            outputMap.put(tensorName, outputTensor);
        }

        return outputMap;
    }

    /**
     * Execute several transformation on given tensor :
     * - Roll over a window if asked for it
     * - Reshape the tensor accordingly to the tensor field
     *
     * @param input The given tensor
     * @param wantedTensorField Reference tensor field indicating the wanted shape and type
     */
    private Tensor transformTensor(Tensor input, TensorField wantedTensorField) {
        DataType wantedTensorType = wantedTensorField.getType();
        TensorShape wantedShape = wantedTensorField.getTensorShape();
        Tensor outputTensor = input;

        // Execute a rolling windows on main tensor if asked for it
        if (this.rollingWindowsSize > 1) {
            outputTensor = outputTensor.roll(0, this.rollingWindowsSize);
        }

        // Check that the wanted shape for main tensor is compatible with the one that we have
        outputTensor.checkTypeCompatible(wantedTensorType);

        // Reshape it to the correct shape and type if needed
        outputTensor = outputTensor.reshapeWithType(wantedShape, wantedTensorType);

        return outputTensor;
    }
}
