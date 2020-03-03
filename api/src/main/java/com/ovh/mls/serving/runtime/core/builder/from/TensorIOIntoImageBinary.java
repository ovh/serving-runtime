package com.ovh.mls.serving.runtime.core.builder.from;

import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.builder.Builder;
import com.ovh.mls.serving.runtime.core.builder.TensorIntoImages;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.img.BinaryContent;
import com.ovh.mls.serving.runtime.utils.img.ImageDefaults;
import org.apache.http.entity.ContentType;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Convert a TensorIO into a Image Binary content
 */
public class TensorIOIntoImageBinary implements Builder<TensorIO, BinaryContent> {

    private final ContentType imageContentType;
    private final List<Field> outputFields;

    public TensorIOIntoImageBinary(List<Field> outputFields, ContentType imageContentType) {
        this.outputFields = outputFields;
        this.imageContentType = imageContentType;
    }

    @Override
    public BinaryContent build(TensorIO input) throws EvaluationException {
        if (input.getTensors().size() != 1) {
            throw new EvaluationException(
                    String.format(
                            "Unable to convert several tensors (%s) into a single image...",
                            Arrays.toString(input.tensorsNames().toArray())
                    )
            );
        }
        Map.Entry<String, Tensor> firstEntry = input.getTensors().entrySet().iterator().next();
        String tensorName = firstEntry.getKey();
        Tensor tensor = firstEntry.getValue();
        TensorIntoImages imageBuilder = ImageDefaults.getImageBuilderOrFail(tensorName, this.outputFields, tensor);
        List<BufferedImage> images = imageBuilder.build(tensor);
        if (images.size() != 1) {
            throw new EvaluationException(
                    String.format(
                            "Unable to convert a batch of %s images into a single image...",
                            images.size()
                    )
            );
        }
        return ImageDefaults.buildImage(images.get(0), this.imageContentType);
    }
}
