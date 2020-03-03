package com.ovh.mls.serving.runtime.processors;

import com.google.common.collect.Sets;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.ovh.mls.serving.runtime.utils.TableUtils;
import tech.tablesaw.api.NumberColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * <p>The StandardScaler applies a standardization by removing the mean and scaling to unit variance</p>
 * <p>The standard score is computed as z = (x - u) / s</p>
 */
public class StandardScaler implements Evaluator {

    private final List<Field> inputs;
    private final List<Field> outputs;
    private final Map<String, MeanStd> meanStdMap;

    public StandardScaler(List<Field> inputs, List<Field> outputs, Map<String, MeanStd> meanStdMap)
        throws EvaluatorException {
        this.inputs = inputs;
        this.outputs = outputs;
        this.meanStdMap = meanStdMap;

        Sets.SetView<String> difference = Sets.difference(
            inputs.stream().map(Field::getName).collect(Collectors.toSet()),
            meanStdMap.keySet()
        );
        if (!difference.isEmpty()) {
            throw new EvaluatorException(
                String.format(
                    "Mean and Standard Deviation are missing to scale %s",
                    String.join(",", difference)
                )
            );
        }
    }

    public static StandardScaler create(StandardScalerManifest manifest, String path) throws EvaluatorException {
        return new StandardScaler(manifest.getInputs(), manifest.getOutputs(), manifest.getMeanStdMap());
    }

    @Override
    public TensorIO evaluate(TensorIO io, EvaluationContext evaluationContext)
        throws EvaluationException {
        Table table = io.intoTable();

        TableUtils.allFieldPresentAndCompatible(table, inputs);
        Map<String, Tensor> output = new HashMap<>();

        IntStream.range(0, getInputs().size()).boxed().forEach(index -> {
            String inputName = inputs.get(index).getName();
            String outputName = outputs.get(index).getName();

            NumberColumn numberColumn = TableUtils.getNumberColumn(inputName, table).get();
            MeanStd meanStd = meanStdMap.get(inputName);

            Column<?> newColumn = numberColumn
                .subtract(meanStd.getMean())
                .divide(meanStd.getStd())
                .setName(outputName);

            Tensor columnTensor = Tensor.fromData(DataType.DOUBLE, newColumn.asObjectArray());
            output.put(newColumn.name(), columnTensor);

        });

        evaluationContext.incEvaluation();

        return new TensorIO(output);
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
