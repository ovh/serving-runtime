package com.ovh.mls.serving.runtime.utils;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.ColumnType;
import tech.tablesaw.api.DateColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.time.LocalDate;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TableUtilsTest {

    @Test
    void createTableOutputColumns() {
        Table table = Table.create();
        List<Field> outputs = Arrays.asList(
            new Field("double", DataType.DOUBLE),
            new Field("float", DataType.FLOAT),
            new Field("integer", DataType.INTEGER),
            new Field("string", DataType.STRING),
            new Field("boolean", DataType.BOOLEAN),
            new Field("date", DataType.DATE)
        );
        TableUtils.createTableOutputColumns(table, outputs);

        List<String> columnNames = table.columnNames();
        assertEquals(6, columnNames.size());

        assertEquals(
            Arrays.asList("double",
                "float",
                "integer",
                "string",
                "boolean",
                "date"),
            columnNames);

        assertEquals(ColumnType.DOUBLE, table.column("double").type());
        assertEquals(ColumnType.FLOAT, table.column("float").type());
        assertEquals(ColumnType.INTEGER, table.column("integer").type());
        assertEquals(ColumnType.STRING, table.column("string").type());
        assertEquals(ColumnType.BOOLEAN, table.column("boolean").type());
        assertEquals(ColumnType.LOCAL_DATE, table.column("date").type());
    }

    @Test
    void getNumberFromRow() {
        Table table = Table.create(
            DoubleColumn.create("double", new double[] {1}),
            FloatColumn.create("float", new float[] {1}),
            IntColumn.create("integer", new int[] {1}),
            StringColumn.create("string", new String[] {"1"}),
            BooleanColumn.create("boolean", new boolean[] {true}),
            DateColumn.create("date", new LocalDate[] {LocalDate.MIN})
        );
        Row row = table.row(0);
        assertEquals(1.0, TableUtils.getNumberFromRow("double", row));
        assertEquals(1.0f, TableUtils.getNumberFromRow("float", row));
        assertEquals(1, TableUtils.getNumberFromRow("integer", row));
        assertEquals(Double.NaN, TableUtils.getNumberFromRow("string", row));
        assertEquals(Double.NaN, TableUtils.getNumberFromRow("boolean", row));
        assertEquals(Double.NaN, TableUtils.getNumberFromRow("date", row));
    }

    @Test
    void isMissing() {
        DoubleColumn column = DoubleColumn.create("double", new Double[] {1., 2.}).setMissing(1);
        Table table = Table.create(
            column
        );

        Assertions.assertFalse(TableUtils.isMissing(table.row(0), column));
        Assertions.assertTrue(TableUtils.isMissing(table.row(1), column));
    }

    @Test
    void getNumberColumn() {
        Table table = Table.create(
            DoubleColumn.create("double", new double[] {1}),
            FloatColumn.create("float", new float[] {1}),
            IntColumn.create("integer", new int[] {1}),
            StringColumn.create("string", new String[] {"1"}),
            BooleanColumn.create("boolean", new boolean[] {true}),
            DateColumn.create("date", new LocalDate[] {LocalDate.MIN})
        );

        Assertions.assertTrue(TableUtils.getNumberColumn("double", table).isPresent());
        Assertions.assertTrue(TableUtils.getNumberColumn("float", table).isPresent());
        Assertions.assertTrue(TableUtils.getNumberColumn("integer", table).isPresent());
        Assertions.assertTrue(TableUtils.getNumberColumn("string", table).isEmpty());
        Assertions.assertTrue(TableUtils.getNumberColumn("boolean", table).isEmpty());
        Assertions.assertTrue(TableUtils.getNumberColumn("date", table).isEmpty());

    }

    @Test
    void allFieldPresentAndCompatible() throws EvaluationException {
        Table table = Table.create(
            DoubleColumn.create("double", new double[] {1}),
            FloatColumn.create("float", new float[] {1}),
            IntColumn.create("integer", new int[] {1}),
            StringColumn.create("string", new String[] {"1"}),
            BooleanColumn.create("boolean", new boolean[] {true}),
            DateColumn.create("date", new LocalDate[] {LocalDate.MIN})
        );
        List<Field> outputs = Arrays.asList(
            new Field("double", DataType.DOUBLE),
            new Field("float", DataType.FLOAT),
            new Field("integer", DataType.INTEGER),
            new Field("string", DataType.STRING),
            new Field("boolean", DataType.BOOLEAN),
            new Field("date", DataType.DATE)
        );
        TableUtils.allFieldPresentAndCompatible(table, outputs);

        List<Field> outputs2 = Arrays.asList(
            new Field("double", DataType.STRING),
            new Field("float", DataType.FLOAT),
            new Field("integer", DataType.INTEGER),
            new Field("string", DataType.STRING),
            new Field("boolean", DataType.BOOLEAN),
            new Field("date", DataType.DATE)
        );
        assertThrows(
            EvaluationException.class,
            () -> TableUtils.allFieldPresentAndCompatible(table, outputs2)
        );
    }

}
