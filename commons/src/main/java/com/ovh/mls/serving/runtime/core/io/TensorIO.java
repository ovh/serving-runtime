package com.ovh.mls.serving.runtime.core.io;

import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.core.tensor.TensorIndex;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.TableUtils;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

import java.lang.reflect.Array;
import java.util.AbstractMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Object describing a collection of named tensors
 */
public class TensorIO {

    /**
     * The map of (tensor name -> tensor)
     */
    private final Map<String, Tensor> tensors;

    /**
     * Main constructor
     */
    public TensorIO() {
        this(new HashMap<>());
    }

    /**
     * Constructor from map of (tensor name -> tensor)
     */
    public TensorIO(Map<String, Tensor> tensors) {
        this.tensors = tensors;
    }

    /**
     * Accessor for map of (tensor name -> tensor)
     */
    public Map<String, Tensor> getTensors() {
        return this.tensors;
    }

    /**
     * Access a single tensor in the collection by its name
     *
     * @param tensorName The name of the tensor that we want to get
     * @return The wanted tensor (or null if none found)
     */
    public Tensor getTensor(String tensorName) {
        return this.tensors.get(tensorName);
    }

    /**
     * Merge the given collection of tensors with the current one
     *
     * @param other other collection of tensors (TensorIO)
     */
    public void merge(TensorIO other) {
        this.tensors.putAll(other.tensors);
    }

    /**
     * Concatenate underlying tensors tensors along specified axis
     */
    public TensorIO concat(TensorIO tensor, int axis) {
        if (!tensor.tensorsNames().equals(this.tensorsNames())) {
            throw new IllegalArgumentException("TensorIO should have the same underlying tensors");
        }
        Map<String, Tensor> concat = tensor.getTensors().entrySet().stream().collect(Collectors.toMap(
            Map.Entry::getKey,
            entry -> this.getTensor(entry.getKey()).concat(entry.getValue(), axis)
        ));
        return new TensorIO(concat);
    }

    /**
     * Look at the first dimension size of each tensors in collection and return the maximum
     */
    // TODO(mlg) I don't understand yet why we need it, would be good if we could avoid it in future
    public int getBatchSize() {
        return this
            .tensors
            .values()
            .stream()
            .map(Tensor::getData)
            .map(x -> {
                if (x.getClass().isArray() && !(x instanceof String) && !(x instanceof byte[])) {
                    return Array.getLength(x);
                } else {
                    return 1;
                }
            })
            .max(Integer::compare)
            .orElse(0);
    }

    /**
     * Retain only given tensor fields and drop the other ones
     *
     * @param fields Tensor fields that we want to keep
     */
    public void retainFields(List<? extends Field> fields) {
        Set<String> retainingNames = new HashSet<>();

        for (Field field : fields) {
            if (field instanceof TensorField) {
                TensorField tensorField = (TensorField) field;
                if (tensorField.getFields() != null) {
                    for (TensorIndex index : tensorField.getFields()) {
                        retainingNames.add(index.getName());
                    }
                }
            }
            retainingNames.add(field.getName());
        }

        this.tensors.keySet().retainAll(retainingNames);
    }

    /**
     * Return the collection of tensor names as a Set
     */
    public Set<String> tensorsNames() {
        return this.tensors.keySet();
    }

    /**
     * Convert the current TensorIO into a tablesaw Table
     */
    //TODO(mlg) will be good to remove all dependencies to TableSaw by deleting this method and adapt dependencies
    public Table intoTable() {
        Map<String, Tensor> allColumnReshaped = this
            .tensors
            .entrySet()
            .stream()
            // Reshape into vector (if any scala or matrix)
            .map(x -> new AbstractMap.SimpleEntry<>(x.getKey(), x.getValue().toVector()))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        // Compare the size all all vector
        Set<Integer> set = allColumnReshaped
            .values()
            .stream()
            .map(x -> x.getShapeAsArray()[0])
            .collect(Collectors.toSet());

        // All size should be the same
        if (set.size() != 1) {
            throw new EvaluationException("Impossible to convert the given input as a 2 dimensional array.");
        }

        List<Column<?>> columns = allColumnReshaped
            .entrySet()
            .stream()
            .map(x -> {
                Field field = new Field(x.getKey(), x.getValue().getType());
                List<Object> elt = (List<Object>) x.getValue().getDataAsList();
                return TableUtils.listToColumn(elt, field);
            })
            .collect(Collectors.toList());

        Column[] columnsArray = new Column[columns.size()];
        columns.toArray(columnsArray);
        return Table.create(columnsArray);
    }

    public Map<String, Object> intoMap() {
        return intoMap(false);
    }

    /**
     * Convert this TensorIO into Map<String, Object> that can be used for JSON serialization
     */
    public Map<String, Object> intoMap(boolean simplify) {
        return this
            .getTensors()
            .entrySet()
            .stream()
            .map(x -> new AbstractMap.SimpleEntry<>(x.getKey(), x.getValue().jsonData(simplify)))
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    /**
     * Simplify all tensors
     */
    public TensorIO simplifyAll() {
        Map<String, Tensor> map = new HashMap<>();
        for (var entry : this.getTensors().entrySet()) {
            map.put(entry.getKey(), entry.getValue().simplifyShape());
        }
        return new TensorIO(map);
    }

}
