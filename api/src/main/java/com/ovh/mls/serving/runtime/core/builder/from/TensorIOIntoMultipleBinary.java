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
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;
import java.util.Map;

public abstract class TensorIOIntoMultipleBinary<O extends OutputStream>
        implements Builder<TensorIO, BinaryContent> {

    private static final String APPLICATION_JSON = "application/json";
    private static final String DEFAULT_MIME_TYPE = APPLICATION_JSON;

    protected final ContentType contentType;
    protected final List<Field> fields;
    protected final boolean shouldSimplify;

    public TensorIOIntoMultipleBinary(
            ContentType contentType,
            List<Field> fields,
            boolean shouldSimplify
    ) {
        this.contentType = contentType;
        this.fields = fields;
        this.shouldSimplify = shouldSimplify;
    }

    protected abstract O buildOutputStream() throws IOException;
    protected abstract void buildImagePart(String tensorName, BinaryContent image, O outputStream) throws IOException;
    protected abstract void buildDefaultPart(String tensorName, Tensor tensor, O outputStream) throws IOException;
    protected abstract BinaryContent buildBinaryContent(O outputStream);

    @Override
    public BinaryContent build(TensorIO input) throws EvaluationException {
        try (O outputStream = this.buildOutputStream()) {

            for (Map.Entry<String, Tensor> entry : input.getTensors().entrySet()) {
                String tensorName = entry.getKey();
                Tensor tensor = entry.getValue();

                String expectedMimeType = this.contentType.getParameter(tensorName);

                if (expectedMimeType == null) {
                    if (ImageDefaults.getMaybeImageBuilder(tensorName, this.fields, tensor).isPresent()) {
                        expectedMimeType = ImageDefaults.PNG_CONTENT_TYPE_STRING;
                    } else {
                        expectedMimeType = DEFAULT_MIME_TYPE;
                    }
                }

                switch (expectedMimeType) {
                    case ImageDefaults.IMG_CONTENT_TYPE_STRING:
                    case ImageDefaults.PNG_CONTENT_TYPE_STRING:
                    case ImageDefaults.JPG_CONTENT_TYPE_STRING:
                        TensorIntoImages imageBuilder = ImageDefaults.getImageBuilderOrFail(
                                tensorName, this.fields, tensor);
                        List<BufferedImage> images = imageBuilder.build(tensor);
                        for (BufferedImage bufferedImage : images) {
                            BinaryContent image = ImageDefaults.buildImage(
                                    bufferedImage,
                                    ContentType.parse(expectedMimeType)
                            );
                            this.buildImagePart(tensorName, image, outputStream);
                        }
                        break;
                    case APPLICATION_JSON:
                    default:
                        this.buildDefaultPart(tensorName, tensor, outputStream);
                        break;
                }
            }
            outputStream.close();
            return buildBinaryContent(outputStream);

        } catch (IOException e) {
            throw new EvaluationException("Unable to create multipart response...", e);
        }
    }
}
