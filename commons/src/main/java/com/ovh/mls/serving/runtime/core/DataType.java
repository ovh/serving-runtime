package com.ovh.mls.serving.runtime.core;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;

import java.util.EnumSet;

/**
 * Enumeration of possible DataTypes supported by evaluators
 *
 * @see Evaluator
 */
public enum DataType {

    STRING,
    INTEGER,
    LONG,
    FLOAT,
    DOUBLE,
    BOOLEAN,
    DATE;

    public static final EnumSet<DataType> NUMBER_TYPES = EnumSet.of(DOUBLE, INTEGER, FLOAT, LONG);

    @JsonCreator
    public static DataType fromString(String name) {
        return valueOf(name.toUpperCase());
    }

    public static boolean isNumberType(DataType type) {
        return NUMBER_TYPES.contains(type);
    }

    @JsonValue
    public String jsonValue() {
        return name().toLowerCase();
    }

    public String toString() {
        return jsonValue();
    }

    public static DataType fromClass(Class<?> clazz) {
        if (clazz == double.class || clazz == Double.class) {
            return DOUBLE;
        } else if (clazz == float.class || clazz == Float.class) {
            return FLOAT;
        } else if (clazz == int.class || clazz == Integer.class) {
            return INTEGER;
        } else if (clazz == long.class || clazz == Long.class) {
            return LONG;
        } else if (clazz == boolean.class || clazz == Boolean.class) {
            return BOOLEAN;
        } else if (clazz == String.class) {
            return STRING;
        } else {
            throw new EvaluationException(String.format("Not handle class type: %s", clazz.toString()));
        }
    }

    public Class<?> getNullableJavaClass() {
        switch (this) {
            case DOUBLE:
                return Double.class;
            case FLOAT:
                return Float.class;
            case INTEGER:
                return Integer.class;
            case LONG:
                return Long.class;
            case BOOLEAN:
                return Boolean.class;
            case STRING:
                return String.class;
            default:
                throw new EvaluationException(String.format("Not handle data type: %s", this));
        }
    }

    public Class<?> getJavaClass() {
        switch (this) {
            case DOUBLE:
                return double.class;
            case FLOAT:
                return float.class;
            case INTEGER:
                return int.class;
            case LONG:
                return long.class;
            case BOOLEAN:
                return boolean.class;
            case STRING:
                return String.class;
            default:
                throw new EvaluationException(String.format("Not handle data type: %s", this));
        }
    }

    public Object convert(Object object) {
        if (object == null) {
            return null;
        }
        switch (this) {
            case DOUBLE:
                return toNumberOrFail(object).doubleValue();
            case FLOAT:
                return toNumberOrFail(object).floatValue();
            case INTEGER:
                return toNumberOrFail(object).intValue();
            case LONG:
                return toNumberOrFail(object).longValue();
            case BOOLEAN:
                if (object instanceof Boolean) {
                    return object;
                } else {
                    throw new EvaluationException(
                        String.format(
                            "Impossible to convert %s of class (%s) into %s",
                            object.toString(),
                            object.getClass().toString(),
                            this.toString()
                        )
                    );
                }
            case STRING:
                return object.toString();
            default:
                throw new EvaluationException(String.format("Not handle data type: %s", this));
        }
    }

    private Number toNumberOrFail(Object number) {
        if (number == null) {
            return null;
        } else if (number instanceof Number) {
            return (Number) number;
        } else {
            throw new EvaluationException(
                String.format(
                    "Impossible to convert %s of class (%s) into %s",
                    number.toString(),
                    number.getClass().toString(),
                    this.toString()
                )
            );
        }
    }
}
