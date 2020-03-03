package com.ovh.mls.serving.runtime.core.tensor;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.google.common.base.Objects;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.stream.Collectors;

@JsonIgnoreProperties(ignoreUnknown = true)
public class TensorIndex extends Field {

    private Integer index;
    private Object key;

    public TensorIndex() {

    }

    public TensorIndex(String name, Integer index) {
        super(name);
        this.index = index;
    }

    public Integer getIndex() {
        return index;
    }

    public void setIndex(Integer index) {
        this.index = index;
    }

    public static void checkFields(List<? extends TensorIndex> fields) throws EvaluatorException {
        String emptyIndexes = fields.stream()
            .filter(field -> field.getIndex() == null)
            .map(Field::getName)
            .collect(Collectors.joining(","));

        if (!StringUtils.isEmpty(emptyIndexes)) {
            throw new EvaluatorException("Indexes are empty for: " + emptyIndexes);
        }
    }

    public Object getKey() {
        return key;
    }

    public void setKey(Object key) {
        this.key = key;
    }

    @Override
    public int hashCode() {
        return super.hashCode() + Objects.hashCode(index);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        TensorIndex field = (TensorIndex) o;
        return Objects.equal(index, field.index) &&
            Objects.equal(key, field.key) &&
            super.equals(field);
    }
}
