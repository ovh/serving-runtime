package com.ovh.mls.serving.runtime.core;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.google.common.base.Objects;

import java.util.ArrayList;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class Field implements Cloneable {

    private String name;
    private DataType type;

    // The possible values for a Field value when a Field is categorical
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private List<String> values = new ArrayList<>();

    // The optional interval value when a Field is continuous
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private List<Interval> continuousDomain;

    public Field() {

    }

    public Field(String name) {
        this.name = name;
    }

    public Field(String name, DataType type) {
        this.name = name;
        this.type = type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Field field = (Field) o;
        return Objects.equal(name, field.name) &&
            type == field.type &&
            Objects.equal(values, field.values);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(name, type, values);
    }

    public List<String> getValues() {
        return values;
    }

    public Field setValues(List<String> values) {
        this.values = values;
        return this;
    }

    public DataType getType() {
        return type;
    }

    public Field setType(DataType type) {
        this.type = type;
        return this;
    }

    public String getName() {
        return name;
    }

    public Field setName(String name) {
        this.name = name;
        return this;
    }

    public List<Interval> getContinuousDomain() {
        return continuousDomain;
    }

    public Field setContinuousDomain(List<Interval> continuousDomain) {
        this.continuousDomain = continuousDomain;
        return this;
    }

    public Field clone() {
        var cloned = new Field(this.name, this.type);
        cloned.setValues(this.values);
        cloned.setContinuousDomain(this.continuousDomain);
        return cloned;
    }
}
