package com.ovh.mls.serving.runtime.timeseries;

import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeFormatterBuilder;
import java.time.temporal.ChronoField;
import java.util.function.Function;

/**
 * Specific TensorField supporting LocalDateTime interpretation and encoding for each tensor's value
 */
public class DateTensorField extends TensorField {

    /**
     * String representation the LocalDateTimeEncoder needed if this field as a numeric type, represente the string
     * formatter of the LocalDateTime otherwise
     */
    private String format;

    /**
     * Default constructor of DateTensorField
     */
    public DateTensorField() {

    }

    /**
     * Accessor of format string
     */
    public String getFormat() {
        return format;
    }

    /**
     * Setter of format string
     */
    public void setFormat(String format) {
        this.format = format;
    }

    /**
     * Create the decode function needed by this field to be interpreted as a LocalDateTime
     */
    public Function<Object, LocalDateTime> decodeDatetimeFunction() {
        switch (this.getType()) {
            case DOUBLE:
            case FLOAT:
            case INTEGER:
            case LONG:
                return x -> LocalDateTimeEncoder.fromString(this.getFormat()).decode(((Number) x).longValue());
            case STRING:
                DateTimeFormatter formatter = timeFormatter(this.getFormat());
                return x -> LocalDateTime.from(formatter.parse((String) x));
            case DATE:
                return x -> (LocalDateTime) x;
            default:
                throw new EvaluationException("Datetime evaluator only support numeric & string formats as inputs");
        }
    }

    /**
     * Create the encode function needed by this field so that the LocalDateTime can be correctly encoded
     */
    public Function<LocalDateTime, Object> encodeDatetimeFunction() {
        String format = this.getFormat();
        switch (this.getType()) {
            case DOUBLE:
                return x -> LocalDateTimeEncoder.fromString(format).encode(x).doubleValue();
            case FLOAT:
                return x -> LocalDateTimeEncoder.fromString(format).encode(x).floatValue();
            case INTEGER:
                return x -> LocalDateTimeEncoder.fromString(format).encode(x).intValue();
            case LONG:
                return x -> LocalDateTimeEncoder.fromString(format).encode(x);
            case STRING:
                DateTimeFormatter outputFormatter = timeFormatter(format);
                return outputFormatter::format;
            case DATE:
                return x -> x;
            default:
                throw new EvaluationException("Datetime evaluator only support numeric & string formats as inputs");
        }
    }

    /**
     * Create a default DateTimeFormatter for LocalDateTime string interpretations
     * @param format the string formatter to use in the DateTimeFormatter
     */
    private static DateTimeFormatter timeFormatter(String format) {
        return new DateTimeFormatterBuilder()
            .appendPattern(format)
            .parseDefaulting(ChronoField.MONTH_OF_YEAR, 1)
            .parseDefaulting(ChronoField.DAY_OF_MONTH, 1)
            .parseDefaulting(ChronoField.HOUR_OF_DAY, 0)
            .parseDefaulting(ChronoField.MINUTE_OF_HOUR, 0)
            .parseDefaulting(ChronoField.SECOND_OF_MINUTE, 0)
            .toFormatter();
    }

}
