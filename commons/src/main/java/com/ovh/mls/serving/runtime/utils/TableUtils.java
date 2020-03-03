package com.ovh.mls.serving.runtime.utils;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.deserialization.TableDeserializationException;
import com.ovh.mls.serving.runtime.exceptions.deserialization.UnexpectedValueForColumnException;
import org.apache.commons.lang3.StringUtils;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.ColumnType;
import tech.tablesaw.api.DateColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.LongColumn;
import tech.tablesaw.api.NumberColumn;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.AbstractColumnType;
import tech.tablesaw.columns.Column;
import tech.tablesaw.columns.numbers.DoubleColumnType;
import tech.tablesaw.columns.numbers.LongColumnType;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class TableUtils {

    private static final List<AbstractColumnType> numberColumnTypes = Arrays.asList(
        ColumnType.DOUBLE,
        ColumnType.FLOAT,
        ColumnType.INTEGER,
        ColumnType.SHORT,
        ColumnType.LONG
    );

    /**
     * Creates typed rows according to a list of fields and add it to the Table.
     *
     * @param table  Table in which to add new columns
     * @param fields field to add as columns
     */
    public static void createTableOutputColumns(Table table, List<? extends Field> fields) {
        int rowCount = table.rowCount();
        // Create output columns
        List<String> existingCols = table.columnNames();
        for (Field output : fields) {
            boolean exist = existingCols.contains(output.getName());
            switch (output.getType()) {
                case FLOAT:
                    if (exist) {
                        if (!(table.column(output.getName()) instanceof FloatColumn)) {
                            throw new EvaluationException("Column %s already exist with a different type");
                        }
                    } else {
                        table.addColumns(FloatColumn.create(output.getName(), rowCount));
                    }
                    break;
                case DOUBLE:
                    if (exist) {
                        if (!(table.column(output.getName()) instanceof DoubleColumn)) {
                            throw new EvaluationException("Column %s already exist with a different type");
                        }
                    } else {
                        table.addColumns(DoubleColumn.create(output.getName(), rowCount));
                    }
                    break;
                case INTEGER:
                    if (exist) {
                        if (!(table.column(output.getName()) instanceof IntColumn)) {
                            throw new EvaluationException("Column %s already exist with a different type");
                        }
                    } else {
                        table.addColumns(IntColumn.create(output.getName(), rowCount));
                    }
                    break;
                case LONG:
                    if (exist) {
                        if (!(table.column(output.getName()) instanceof LongColumn)) {
                            throw new EvaluationException("Column %s already exist with a different type");
                        }
                    } else {
                        table.addColumns(LongColumn.create(output.getName(), rowCount));
                    }
                    break;
                case DATE:
                    if (exist) {
                        if (!(table.column(output.getName()) instanceof DateColumn)) {
                            throw new EvaluationException("Column %s already exist with a different type");
                        }
                    } else {
                        table.addColumns(DateColumn.create(output.getName(), rowCount));
                    }
                    break;
                case BOOLEAN:
                    if (exist) {
                        if (!(table.column(output.getName()) instanceof BooleanColumn)) {
                            throw new EvaluationException("Column %s already exist with a different type");
                        }
                    } else {
                        table.addColumns(BooleanColumn.create(output.getName(), rowCount));
                    }
                    break;
                case STRING:
                    if (exist) {
                        if (!(table.column(output.getName()) instanceof StringColumn)) {
                            throw new EvaluationException("Column %s already exist with a different type");
                        }
                    } else {
                        table.addColumns(StringColumn.create(output.getName(), rowCount));
                    }
                    break;
            }
        }
    }

    /**
     * Get a number value from a row
     *
     * @param name the name of the column to get the number
     * @param row  Row in from which to get value
     * @return value as a Number
     */
    public static Number getNumberFromRow(String name, Row row) {
        Object rawValue = row.getObject(name);

        if (rawValue instanceof Number) {
            return (Number) rawValue;
        }
        return Double.NaN;
    }

    /**
     * Checks whether the column is missing at the specified Row
     *
     * @param row    Row in which we look for the column
     * @param column Column to check
     * @return boolean true if value is missinig
     */
    public static boolean isMissing(Row row, Column<?> column) {
        return column.isMissing(row.getRowNumber());
    }

    /**
     * Gets a column from a table as a NumberColumn.
     *
     * @param name  name of the column to get
     * @param table source table in which to get the column
     * @return an optional of the column as a NumberColumn if it is a NumberColumn or an empty optional if it is not
     * @throws IllegalStateException if the column is not in the table
     */
    public static Optional<NumberColumn> getNumberColumn(String name, Table table) {
        Column<?> column = table.column(name);
        if (column instanceof NumberColumn) {
            return Optional.of((NumberColumn) column);
        } else {
            return Optional.empty();
        }
    }

    /**
     * Checks that the list of specified Fields are all in the table
     *
     * @param table  source table
     * @param fields list of field to check
     * @throws EvaluationException if any Field is missing
     */
    public static void allFieldPresentAndCompatible(Table table, List<Field> fields) throws EvaluationException {
        List<String> columnNames = table.columnNames();
        String missingInputs = fields
            .stream()
            .filter(field -> {
                if (!columnNames.contains(field.getName())) {
                    return true;
                }
                return !isCompatibleTypeColumn(table.column(field.getName()).type(), field.getType());
            })
            .map(field -> String.format("(%s,%s)", field.getName(), field.getType()))
            .collect(Collectors.joining(","));

        if (!StringUtils.isEmpty(missingInputs)) {
            throw new EvaluationException(
                String.format("Required fields %s not found or not compatible in the Table", missingInputs)
            );
        }
    }

    /**
     * Checks that a DataType and a given ColumnType are compatible, meaning that a value associated to the DataType
     * can be set into the Column associated to this ColumnType
     *
     * @param columnType ColumnType to check
     * @param dataType   DataType to check
     * @return a boolean true if types are compatible and false if they are not
     */
    private static boolean isCompatibleTypeColumn(ColumnType columnType, DataType dataType) {
        if (DataType.isNumberType(dataType)) {
            // depending on the actual values of the field it might be stored in a tighter type (Int instead of Double)
            return numberColumnTypes.contains(columnType);
        } else {
            switch (dataType) {
                case DATE:
                    return columnType == ColumnType.LOCAL_DATE;
                case BOOLEAN:
                    return columnType == ColumnType.BOOLEAN;
                case STRING:
                    return columnType == ColumnType.STRING;
                default:
                    throw new IllegalStateException(String.format("Unhandled DataType %s", dataType));
            }
        }
    }

    /**
     * Convert a `List<Object>` object into `Column<?>` with a given expected field
     */
    public static Column<?> listToColumn(List<Object> list, Field expectedField) {
        switch (expectedField.getType()) {
            case FLOAT:
            case DOUBLE:
                return createDoubleColumn(list, expectedField);
            case INTEGER:
            case LONG:
                return createLongColumn(list, expectedField);
            case DATE:
                return createDateColumn(list, expectedField);
            case BOOLEAN:
                return createBooleanColumn(list, expectedField);
            case STRING:
                return createStringColumn(list, expectedField);
            default:
                throw new TableDeserializationException(
                    String.format("Unable to find column converter for class %s", expectedField.getType().toString())
                );
        }
    }

    /**
     * Create a Double column
     *
     * @param list          The object list representing the column
     * @param expectedField The expected field information
     * @return The Double column
     */
    private static Column<?> createDoubleColumn(List<Object> list, Field expectedField) {
        List<Number> doubleList = list
            .stream()
            .map(x -> Optional.ofNullable(x).orElse(DoubleColumnType.missingValueIndicator()))
            .map(x -> {
                if (x instanceof Number) {
                    return ((Number) x).doubleValue();
                } else {
                    throw new UnexpectedValueForColumnException(expectedField, x);
                }
            })
            .collect(Collectors.toList());
        return DoubleColumn.create(expectedField.getName(), doubleList);
    }

    /**
     * Create a Long column
     *
     * @param list          The object list representing the column
     * @param expectedField The expected field information
     * @return The Long column
     */
    private static Column<?> createLongColumn(List<Object> list, Field expectedField) {
        long[] integerList = list
            .stream()
            .map(x -> Optional.ofNullable(x).orElse(LongColumnType.missingValueIndicator()))
            .map(x -> {
                if (x instanceof Number) {
                    return ((Number) x).longValue();
                } else {
                    throw new UnexpectedValueForColumnException(expectedField, x);
                }
            })
            .mapToLong(x -> x)
            .toArray();
        return LongColumn.create(expectedField.getName(), integerList);
    }

    /**
     * Create a Date column
     *
     * @param list          The object list representing the column
     * @param expectedField The expected field information
     * @return The Date column
     */
    private static Column<?> createDateColumn(List<Object> list, Field expectedField) {
        List<LocalDate> dateList = new ArrayList<>();
        for (Object object : list) {
            if (object == null || object instanceof LocalDate) {
                dateList.add((LocalDate) object);
            } else {
                throw new UnexpectedValueForColumnException(expectedField, object);
            }
        }
        return DateColumn.create(expectedField.getName(), dateList);
    }

    /**
     * Create a String column
     *
     * @param list          The object list representing the column
     * @param expectedField The expected field information
     * @return The String column
     */
    private static Column<?> createStringColumn(List<Object> list, Field expectedField) {
        List<String> stringList = new ArrayList<>();
        for (Object object : list) {
            if (object == null || object instanceof String) {
                stringList.add((String) object);
            } else {
                throw new UnexpectedValueForColumnException(expectedField, object);
            }
        }
        return StringColumn.create(expectedField.getName(), stringList);
    }

    /**
     * Create a Boolean column
     *
     * @param list          The object list representing the column
     * @param expectedField The expected field information
     * @return The Boolean column
     */
    private static Column<?> createBooleanColumn(List<Object> list, Field expectedField) {
        List<Boolean> booleanList = new ArrayList<>();
        for (Object object : list) {
            if (object == null || object instanceof Boolean) {
                booleanList.add((Boolean) object);
            } else {
                throw new UnexpectedValueForColumnException(expectedField, object);
            }
        }
        return BooleanColumn.create(expectedField.getName(), booleanList);
    }

}
