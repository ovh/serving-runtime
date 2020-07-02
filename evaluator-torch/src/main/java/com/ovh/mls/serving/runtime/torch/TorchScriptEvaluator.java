package com.ovh.mls.serving.runtime.torch;

import com.ovh.mls.serving.runtime.core.AbstractTensorEvaluator;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * Evaluate a TorchScript module
 */
public class TorchScriptEvaluator extends AbstractTensorEvaluator<TensorField> {

    // TorchScript module
    private final Module module;

    public TorchScriptEvaluator(Module module, List<TensorField> inputs, List<TensorField> outputs) {
        super(inputs, outputs, 0);
        this.module = module;
    }

    @Override
    protected TensorIO evaluateTensor(TensorIO io) throws EvaluationException {
        // Convert input for Torch
        IValue[] inputs = this.getInputTensorField().stream()
            .map(tensorField -> {
                String name = tensorField.getName();

                com.ovh.mls.serving.runtime.core.tensor.Tensor servingTensor = io.getTensor(name);

                return servingTensorToTorchTensor(servingTensor);
            })
            .map(IValue::from)
            .toArray(IValue[]::new);

        // Evaluate model
        IValue result = module.forward(inputs);
        IValue[] outputs;
        if (result.isTuple()) {
            outputs = result.toTuple();
        } else {
            outputs = new IValue[]{result};
        }

        // Convert output for Serving
        TensorIO output = new TensorIO();
        for (int i = 0; i < getOutputTensorField().size(); i++) {
            TensorField tensorField = getOutputTensorField().get(i);
            Tensor torchTensor = outputs[i].toTensor();

            com.ovh.mls.serving.runtime.core.tensor.Tensor servingTensor = torchTensorToServingTensor(torchTensor);

            output.getTensors().put(tensorField.getName(), servingTensor);
        }

        return output;
    }

    /**
     * Convert a Serving tensor into a Torch tensor
     */
    private Tensor servingTensorToTorchTensor(com.ovh.mls.serving.runtime.core.tensor.Tensor servingTensor) {
        Object data = servingTensor.toVector().getData();
        long[] shape = Arrays.stream(servingTensor.getShapeAsArray())
            .mapToLong(dim -> dim)
            .toArray();

        Tensor torchTensor;
        switch (servingTensor.getType()) {
            case INTEGER:
                torchTensor = Tensor.fromBlob((int[]) data, shape);
                break;
            case LONG:
                torchTensor = Tensor.fromBlob((long[]) data, shape);
                break;
            case FLOAT:
                torchTensor = Tensor.fromBlob((float[]) data, shape);
                break;
            case DOUBLE:
                torchTensor = Tensor.fromBlob(shape, (double[]) data);
                break;
            default:
                throw new EvaluatorException(String.format(
                    "Tensor of type '%s' cannot be converted to pyTorch tensor",
                    servingTensor.getType()
                ));
        }
        return torchTensor;
    }

    /**
     * Convert a Torch tensor into a Serving Tensor
     */
    private com.ovh.mls.serving.runtime.core.tensor.Tensor torchTensorToServingTensor(Tensor torchTensor) {
        int[] shape = new int[]{(int) torchTensor.numel()};

        com.ovh.mls.serving.runtime.core.tensor.Tensor servingTensor;
        switch (torchTensor.dtype()) {
            case INT32:
                servingTensor = new com.ovh.mls.serving.runtime.core.tensor.Tensor(
                    DataType.INTEGER,
                    shape,
                    torchTensor.getDataAsIntArray()
                );
                break;
            case INT64:
                servingTensor = new com.ovh.mls.serving.runtime.core.tensor.Tensor(
                    DataType.LONG,
                    shape,
                    torchTensor.getDataAsLongArray()
                );
                break;
            case FLOAT32:
                servingTensor = new com.ovh.mls.serving.runtime.core.tensor.Tensor(
                    DataType.FLOAT,
                    shape,
                    torchTensor.getDataAsFloatArray()
                );
                break;
            case FLOAT64:
                servingTensor = new com.ovh.mls.serving.runtime.core.tensor.Tensor(
                    DataType.DOUBLE,
                    shape,
                    torchTensor.getDataAsDoubleArray()
                );
                break;
            default:
                throw new EvaluatorException(String.format(
                    "Tensor of type '%s' cannot be converted from pyTorch tensor",
                    torchTensor.dtype()
                ));
        }
        return servingTensor;
    }
}
