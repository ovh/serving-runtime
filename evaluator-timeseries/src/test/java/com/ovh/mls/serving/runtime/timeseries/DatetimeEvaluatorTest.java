package com.ovh.mls.serving.runtime.timeseries;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DatetimeEvaluatorTest {

    private static final ClassLoader LOADER = DatetimeEvaluatorTest.class.getClassLoader();

    DatetimeEvaluator createEvaluator(String path) throws IOException {
        ObjectMapper objectMapper = new ObjectMapper();
        DatetimeEvaluatorManifest predictionIntervalEvaluatorManifest = objectMapper.readValue(
            LOADER.getResourceAsStream(path),
            DatetimeEvaluatorManifest.class
        );
        return predictionIntervalEvaluatorManifest.create("");
    }

    @Test
    void evaluateTimestampManifest() throws IOException {
        DatetimeEvaluator datetimeStringEval = createEvaluator("timeseries/datetime-timestamp-manifest.json");
        TensorIO input = new TensorIO(
            Map.of("timestamp-seconds", Tensor.fromLongData(new long[]{1514764800L, 1577836799L})));
        TensorIO output = datetimeStringEval.evaluate(input, new EvaluationContext());
        assertEquals(2, output.getBatchSize());
        assertEquals(Set.of("date"), output.tensorsNames());
        Tensor dateTensor = output.getTensor("date");
        assertArrayEquals(new int[]{2, 1}, dateTensor.getShapeAsArray());
        assertEquals(DataType.STRING, dateTensor.getType());
        assertEquals("2018-01-01 00:00:00", ((String[][]) dateTensor.getData())[0][0]);
        assertEquals("2019-12-31 23:59:59", ((String[][]) dateTensor.getData())[1][0]);
    }

    @Test
    void evaluateDatetimeStringManifest() throws EvaluationException, IOException {
        DatetimeEvaluator datetimeStringEval = createEvaluator("timeseries/datetime-string-manifest.json");
        TensorIO input = new TensorIO(
            Map.of("date", Tensor.fromStringData(new String[]{"2018-01-01 00:00:00", "2019-12-31 23:59:59"})));
        TensorIO output = datetimeStringEval.evaluate(input, new EvaluationContext());

        assertEquals(2, output.getBatchSize());
        assertEquals(
            Set.of(
                "year",
                "month",
                "dayofmonth",
                "hourofday",
                "minuteofhour",
                "dayofweek",
                "year-string",
                "year-float",
                "timestamp-seconds",
                "timestamp-years"
            ),
            output.tensorsNames()
        );
        Tensor yearTensor = output.getTensor("year");
        Tensor monthTensor = output.getTensor("month");
        Tensor dayOfMonthTensor = output.getTensor("dayofmonth");
        Tensor hourofdayTensor = output.getTensor("hourofday");
        Tensor minuteofhourTensor = output.getTensor("minuteofhour");
        Tensor dayofweekTensor = output.getTensor("dayofweek");
        Tensor yearString = output.getTensor("year-string");
        Tensor yearFloat = output.getTensor("year-float");
        Tensor timestampSeconds = output.getTensor("timestamp-seconds");
        Tensor timestampYears = output.getTensor("timestamp-years");

        // Check shapes
        assertArrayEquals(new int[]{2, 1}, yearTensor.getShapeAsArray());
        assertArrayEquals(new int[]{2, 1}, monthTensor.getShapeAsArray());
        assertArrayEquals(new int[]{2, 1}, dayOfMonthTensor.getShapeAsArray());
        assertArrayEquals(new int[]{2, 1}, hourofdayTensor.getShapeAsArray());
        assertArrayEquals(new int[]{2, 1}, minuteofhourTensor.getShapeAsArray());
        assertArrayEquals(new int[]{2, 1}, dayofweekTensor.getShapeAsArray());
        assertArrayEquals(new int[]{2, 1}, yearString.getShapeAsArray());
        assertArrayEquals(new int[]{2, 1}, yearFloat.getShapeAsArray());
        assertArrayEquals(new int[]{2, 1}, timestampSeconds.getShapeAsArray());
        assertArrayEquals(new int[]{2, 1}, timestampYears.getShapeAsArray());

        // Check data types
        assertEquals(DataType.INTEGER, yearTensor.getType());
        assertEquals(DataType.INTEGER, monthTensor.getType());
        assertEquals(DataType.INTEGER, dayOfMonthTensor.getType());
        assertEquals(DataType.INTEGER, hourofdayTensor.getType());
        assertEquals(DataType.INTEGER, minuteofhourTensor.getType());
        assertEquals(DataType.INTEGER, dayofweekTensor.getType());
        assertEquals(DataType.STRING, yearString.getType());
        assertEquals(DataType.FLOAT, yearFloat.getType());
        assertEquals(DataType.LONG, timestampSeconds.getType());
        assertEquals(DataType.LONG, timestampYears.getType());

        assertEquals(2018, ((int[][]) yearTensor.getData())[0][0]);
        assertEquals(1, ((int[][]) monthTensor.getData())[0][0]);
        assertEquals(1, ((int[][]) dayOfMonthTensor.getData())[0][0]);
        assertEquals(0, ((int[][]) hourofdayTensor.getData())[0][0]);
        assertEquals(0, ((int[][]) minuteofhourTensor.getData())[0][0]);
        assertEquals(1, ((int[][]) dayofweekTensor.getData())[0][0]);
        assertEquals("2018", ((String[][]) yearString.getData())[0][0]);
        assertEquals(2018.0, ((float[][]) yearFloat.getData())[0][0]);
        assertEquals(1514764800L, ((long[][]) timestampSeconds.getData())[0][0]);
        assertEquals(48L, ((long[][]) timestampYears.getData())[0][0]);

        assertEquals(2019, ((int[][]) yearTensor.getData())[1][0]);
        assertEquals(12, ((int[][]) monthTensor.getData())[1][0]);
        assertEquals(31, ((int[][]) dayOfMonthTensor.getData())[1][0]);
        assertEquals(23, ((int[][]) hourofdayTensor.getData())[1][0]);
        assertEquals(59, ((int[][]) minuteofhourTensor.getData())[1][0]);
        assertEquals(2, ((int[][]) dayofweekTensor.getData())[1][0]);
        assertEquals(2019.0, ((float[][]) yearFloat.getData())[1][0]);
        assertEquals(1577836799L, ((long[][]) timestampSeconds.getData())[1][0]);
        assertEquals(49L, ((long[][]) timestampYears.getData())[1][0]);
    }

}
