package com.ovh.mls.serving.runtime.core.builder.from;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.utils.img.BinaryContent;
import org.apache.http.entity.ContentType;
import org.apache.http.message.BasicNameValuePair;
import org.eclipse.jetty.util.MultiPartOutputStream;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Convert a TensorIO into a Multipart Binary content
 */
public class TensorIOIntoMultipartBinary extends TensorIOIntoMultipleBinary<MultiPartOutputStream> {

    private static final String APPLICATION_JSON = "application/json";
    private final ByteArrayOutputStream os;
    private final ObjectMapper mapper;

    public TensorIOIntoMultipartBinary(
        ObjectMapper mapper,
        ContentType contentType,
        List<Field> fields,
        boolean shouldSimplify
    ) {
        super(contentType, fields, shouldSimplify);
        this.mapper = mapper;
        this.os = new ByteArrayOutputStream();
    }

    @Override
    protected MultiPartOutputStream buildOutputStream() throws IOException {
        return new MultiPartOutputStream(this.os);
    }

    @Override
    protected void buildImagePart(
        String tensorName,
        BinaryContent image,
        MultiPartOutputStream outputStream
    ) throws IOException {
        String imageFilename = String.format("%s.%s", tensorName, image.getFileExtension());
        outputStream.startPart(
            image.getContentType().getMimeType(),
            new String[]{
                buildContentDispositionHeader(tensorName, imageFilename)
            });
        outputStream.write(image.getBytes());
    }

    @Override
    protected void buildDefaultPart(
        String tensorName,
        Tensor tensor,
        MultiPartOutputStream outputStream
    ) throws IOException {
        String jsonFilename = String.format("%s.json", tensorName);
        outputStream.startPart(
            APPLICATION_JSON,
            new String[]{
                buildContentDispositionHeader(tensorName, jsonFilename)
            });
        outputStream.write(mapper.writeValueAsBytes(tensor.jsonData(this.shouldSimplify)));
    }

    @Override
    protected BinaryContent buildBinaryContent(MultiPartOutputStream outputStream) {
        BasicNameValuePair param = new BasicNameValuePair("boundary", outputStream.getBoundary());
        return new BinaryContent(
            null,
            ContentType.MULTIPART_FORM_DATA.withCharset(StandardCharsets.UTF_8).withParameters(param),
            this.os.toByteArray()
        );
    }

    protected String buildContentDispositionHeader(String name, String filename) {
        return String.format(
            "Content-Disposition: form-data; name=\"%s\"; filename=\"%s\"",
            name,
            filename
        );
    }

}
