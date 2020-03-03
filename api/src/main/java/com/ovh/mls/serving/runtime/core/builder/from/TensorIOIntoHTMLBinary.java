package com.ovh.mls.serving.runtime.core.builder.from;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.utils.img.BinaryContent;
import org.apache.http.entity.ContentType;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;

/**
 * Convert a TensorIO into an HTML binary
 *
 * Tensor's name are converted in 'h1' tag
 * Scalars are printed inside 'p' tag
 * Vectors are printed inside 'ul' and 'li' tag (listing)
 * Images are printed as png or jpeg inside 'img' tag
 * Other are printed inside 'code' tag
 */
public class TensorIOIntoHTMLBinary extends TensorIOIntoMultipleBinary<ByteArrayOutputStream> {

    private final ObjectMapper mapper;

    public TensorIOIntoHTMLBinary(
        ContentType contentType,
        ObjectMapper mapper,
        List<Field> fields,
        boolean shouldSimplify
    ) {
        super(contentType, fields, shouldSimplify);
        this.mapper = mapper;
    }

    @Override
    protected ByteArrayOutputStream buildOutputStream() throws IOException {
        return new ByteArrayOutputStream();
    }

    @Override
    protected void buildImagePart(
        String tensorName,
        BinaryContent image,
        ByteArrayOutputStream outputStream
    ) throws IOException {
        String imageAsBase64 = Base64.getEncoder().encodeToString(image.getBytes());
        String content = String.format(
            "<h1>%s</h1><img id=\"ItemPreview\" src=\"data:%s;base64,%s\">",
            tensorName,
            image.getContentType().getMimeType(),
            imageAsBase64
        );
        outputStream.write(content.getBytes());
    }

    @Override
    protected void buildDefaultPart(
        String tensorName,
        Tensor tensor,
        ByteArrayOutputStream outputStream
    ) throws IOException {
        String content = String.format("<h1>%s</h1>", tensorName);
        outputStream.write(content.getBytes());
        if (this.shouldSimplify) {
            tensor = tensor.simplifyShape();
        }
        if (tensor.isScalar()) {
            outputStream.write(String.format("<p>%s</p>", tensor.getData().toString()).getBytes());
        } else if (tensor.isVector()) {
            outputStream.write("<ul>".getBytes());
            for (int i = 0; i < tensor.getShapeAsArray()[0]; i++) {
                Object value = tensor.getCoord(i);
                outputStream.write(String.format("<li>%s</li>", value.toString()).getBytes());
            }
            outputStream.write("</ul>".getBytes());
        } else {
            String value = mapper.writeValueAsString(tensor.jsonData(this.shouldSimplify));
            outputStream.write(String.format("<code>%s</code>", value).getBytes());
        }
    }

    @Override
    protected BinaryContent buildBinaryContent(ByteArrayOutputStream outputStream) {
        return new BinaryContent(
            "html",
            ContentType.TEXT_HTML.withCharset(StandardCharsets.UTF_8),
            outputStream.toByteArray()
        );
    }
}
