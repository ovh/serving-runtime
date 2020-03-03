package com.ovh.mls.serving.runtime.core.io;

import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import org.junit.jupiter.api.Test;
import tech.tablesaw.api.Table;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorIOTest {

    @Test
    public void testIntoTable() {
        Tensor tensor1 = Tensor.fromLongData(new long[]{1, 2, 3});
        Tensor tensor2 = Tensor.fromStringData(new String[]{"1", "2", "3"});

        Table table = new TensorIO(Map.of("col1", tensor1, "col2", tensor2)).intoTable();
        assertEquals(2, table.columnCount());
        assertEquals(3, table.rowCount());

        assertEquals(1L, table.column("col1").get(0));
        assertEquals(2L, table.column("col1").get(1));
        assertEquals(3L, table.column("col1").get(2));
        assertEquals("1", table.column("col2").get(0));
        assertEquals("2", table.column("col2").get(1));
        assertEquals("3", table.column("col2").get(2));
    }

}
