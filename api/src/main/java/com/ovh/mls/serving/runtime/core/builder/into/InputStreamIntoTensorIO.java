package com.ovh.mls.serving.runtime.core.builder.into;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.builder.Builder;
import com.ovh.mls.serving.runtime.core.builder.ImagesIntoTensor;
import com.ovh.mls.serving.runtime.core.builder.InputStreamJsonIntoTensorIO;
import com.ovh.mls.serving.runtime.core.builder.PartsIntoTensorIO;
import com.ovh.mls.serving.runtime.core.io.Part;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.MultipartUtils;
import com.ovh.mls.serving.runtime.utils.img.ImageDefaults;
import org.apache.http.entity.ContentType;

import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Builder of TensorIO from an InputStream
 */
public class InputStreamIntoTensorIO implements Builder<InputStream, TensorIO> {

    private final ContentType contentType;
    private final ObjectMapper mapper;
    private final List<Field> fields;

    public InputStreamIntoTensorIO(ObjectMapper mapper) {
        this(mapper, ContentType.APPLICATION_JSON, new ArrayList<>());
    }

    public InputStreamIntoTensorIO(ObjectMapper mapper, ContentType contentType, List<Field> fields) {
        this.contentType = contentType;
        this.mapper = mapper;
        this.fields = fields;
    }

    @Override
    public TensorIO build(InputStream inputStream) throws EvaluationException {
        String mimeType = this.contentType.getMimeType();
        if (ContentType.APPLICATION_JSON.getMimeType().equals(mimeType)) {
            return new InputStreamJsonIntoTensorIO(this.mapper).build(inputStream);
        } else if (ContentType.MULTIPART_FORM_DATA.getMimeType().equals(mimeType)) {
            final PartsIntoTensorIO builder = new PartsIntoTensorIO(this.mapper, this.fields);
            List<Part> parts = MultipartUtils.readParts(contentType, inputStream);
            return builder.build(parts);
        } else if (ImageDefaults.SUPPORTED_IMG_CONTENT_TYPE.contains(mimeType)) {
            if (this.fields.size() != 1) {
                throw new EvaluationException(
                        "Your model takes several tensors as parameters but you provided only one"
                );
            }
            Field field = this.fields.get(0);
            if (!(field instanceof TensorField)) {
                throw new EvaluationException("Unable to feed an image as tensor on a non-tensor model");
            }
            return buildSingleImage((TensorField) field, inputStream);
        } else {
            throw new EvaluationException(String.format("Unsupported Content-Type: %s", mimeType));
        }
    }

    private TensorIO buildSingleImage(TensorField field, InputStream inputStream) throws EvaluationException {
        ImagesIntoTensor builder = field.getImageTransformer().tensorBuilder();
        BufferedImage image = ImageDefaults.readImage(inputStream);
        Tensor tensor = builder.build(List.of(image));
        return new TensorIO(Map.of(field.getName(), tensor));
    }

}
