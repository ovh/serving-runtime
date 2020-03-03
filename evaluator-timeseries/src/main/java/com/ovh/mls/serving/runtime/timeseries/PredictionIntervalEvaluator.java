package com.ovh.mls.serving.runtime.timeseries;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.TableUtils;
import com.ovh.mls.serving.runtime.validation.NumberOnly;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.tablesaw.api.Row;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

import java.util.AbstractMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * <p>
 * The PredictionIntervalEvaluator aims at estimating prediction intervals for a sequence forecast.
 * </p>
 * <p>
 * A Gaussian distribution is assumed for the forecast and its associated error (residual). At training time for each
 * horizon for which the forecast is learned the training residual is used as the standard deviation for the
 * distribution at this horizon. The actual forecast value is used as the mean of the same distribution. Both values
 * completely describe the Gaussian distribution allowing to compute quantiles according to some confidence level.
 * </p>
 * <p>
 * For sequences forecast can also be performed using a recursive approach, using the newly forecasted values as
 * input to forecast further values. In such cases the residual for each recursive call is unknown (only thew first
 * one is known during training) thus a correction factor is applied. The corrected standard deviation is residual
 * standard deviation multiplied by the square root of the number of recursive steps
 * <code>residual_std*Math.sqrt(n)<code/>
 * </p>
 */
@NumberOnly
public class PredictionIntervalEvaluator implements Evaluator {
    private static final Logger LOGGER = LoggerFactory.getLogger(PredictionIntervalEvaluator.class);

    private static final double DEFAULT_CONFIDENCE_LEVEL = 0.9;
    private static final String LOWER_BOUND_FORMAT = "%s_quantile_inf";
    private static final String UPPER_BOUND_FORMAT = "%s_quantile_sup";

    private final List<Field> inputs;
    private final List<Field> outputs;
    private final Map<String, Double> residualsStd;

    private final List<Field> generatedOutputs;
    private final List<String> desiredOutputNames;


    private PredictionIntervalEvaluator(
        Map<String, Double> residualsStd,
        List<Field> inputs,
        List<Field> outputs
    ) {
        this.residualsStd = residualsStd;
        this.inputs = inputs;
        this.outputs = outputs;
        this.desiredOutputNames = this.outputs.stream().map(Field::getName).collect(Collectors.toList());
        this.generatedOutputs = inputs.stream()
            .flatMap(input ->
                Stream.of(
                    new Field(String.format(LOWER_BOUND_FORMAT, input.getName()), DataType.DOUBLE),
                    new Field(String.format(UPPER_BOUND_FORMAT, input.getName()), DataType.DOUBLE)
                )
            ).collect(Collectors.toList());

    }

    public static PredictionIntervalEvaluator create(PredictionIntervalEvaluatorManifest manifest, String path) {
        return new PredictionIntervalEvaluator(manifest.getResidualsStd(), manifest.getInputs(), manifest.getOutputs());
    }

    @Override
    public TensorIO evaluate(TensorIO input, EvaluationContext context)
        throws EvaluationException {

        context.incEvaluation();
        return evaluate(input, DEFAULT_CONFIDENCE_LEVEL, 1);
    }

    /**
     * Computes prediction intervals for the set of inputs specified upon evaluator instantiation. Only prediction
     * forecast intervals specified in the outputs upon evaluator instantiation are effectively maintained in the Table.
     * On the event of a forecast made from another forecast (recursive forecast total) then a correction factor
     * is applied to the computed interval boundaries of square root of step.
     * <p>
     * Prediction intervals are computed assuming the forecast distribution is a Gaussian of mean forecasted value
     * and of standard deviation the training residuals mean squared error for each horizon.
     *
     * @param tensors         TensorIO over which predictions interval are computed
     * @param confidenceLevel confidence level desired for the prediction intervals (default: 0.9)
     * @param step            the number of recursive steps used to make the forecast
     * @return table with the prediction forecast
     */
    private TensorIO evaluate(TensorIO tensors, Double confidenceLevel, int step) throws EvaluationException {

        Table table = tensors.intoTable();

        TableUtils.allFieldPresentAndCompatible(table, inputs);

        TableUtils.createTableOutputColumns(table, generatedOutputs);

        Double lowerBoundProba = (1.0 - confidenceLevel) / 2.0;
        Double upperBoundProba = (1.0 + confidenceLevel) / 2.0;

        Map<String, Column<?>> columnMap = table
            .columns()
            .stream()
            .collect(Collectors.toMap(Column::name, Function.identity()));

        // apply evaluator, outputs will be written directly in the newly created table output columns
        table.rollingStream(1).forEach(
            rows -> {
                for (Row row : rows) {
                    evaluateRow(step, lowerBoundProba, upperBoundProba, row, columnMap);
                }
            }
        );

        Map<String, Tensor> outputTensors = this
            .generatedOutputs
            .stream()
            .filter(field -> desiredOutputNames.contains(field.getName()))
            .map(x -> {
                Column<?> column = table.column(x.getName());
                Tensor columnTensor = Tensor.fromData(x.getType(), column.asObjectArray());
                return new AbstractMap.SimpleEntry<>(x.getName(), columnTensor.simplifyShape());
            })
            .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        return new TensorIO(outputTensors);
    }

    private void evaluateRow(
        int step,
        Double lowerBoundProba,
        Double upperBoundProba,
        Row row,
        Map<String, Column<?>> columnMap
    ) {
        IntStream.range(0, getInputs().size()).boxed().forEach(
            index -> {
                Field input = getInputs().get(index);
                String inputName = input.getName();

                if (TableUtils.isMissing(row, columnMap.get(inputName))) {
                    return;
                }

                double mean = TableUtils.getNumberFromRow(inputName, row).doubleValue();
                NormalDistribution normalDistribution = new NormalDistribution(
                    mean,
                    residualsStd.get(inputName) * Math.sqrt(step) // Residual std with step adjustment
                );
                double lowerBound = normalDistribution.inverseCumulativeProbability(lowerBoundProba);
                double upperBound = normalDistribution.inverseCumulativeProbability(upperBoundProba);

                row.setDouble(this.generatedOutputs.get(2 * index).getName(), lowerBound);
                row.setDouble(this.generatedOutputs.get(2 * index + 1).getName(), upperBound);
            }
        );
    }

    @Override
    public List<Field> getInputs() {
        return inputs;
    }

    @Override
    public List<Field> getOutputs() {
        return outputs;
    }
}
