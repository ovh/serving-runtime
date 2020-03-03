package com.ovh.mls.serving.runtime.core.builder.from;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.builder.Builder;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.img.BinaryContent;
import com.ovh.mls.serving.runtime.utils.img.ImageDefaults;
import org.apache.http.entity.ContentType;

import javax.ws.rs.core.Response;
import java.util.List;

public class TensorIOIntoResponse implements Builder<TensorIO, Response> {

    private static final String DEFAULT_ACCEPT_HEADER = "*/*";
    private static final String CONTENT_TYPE = "Content-Type";
    private static final String MULTIPART_MIME = "multipart/form-data";
    private static final String JSON_MIME = "application/json";
    private static final String HTML_MIME = "text/html";

    private final String acceptHeader;
    private final ContentType contentType;
    private final ObjectMapper mapper;

    // List of output fields
    private final List<Field> fields;
    // Should simplify the output (reshape the tensor)
    private final boolean shouldSimplify;

    public TensorIOIntoResponse(
        String acceptHeader,
        ObjectMapper mapper,
        List<Field> fields,
        boolean shouldSimplify
    ) {
        this.mapper = mapper;
        this.shouldSimplify = shouldSimplify;
        if (acceptHeader != null && !acceptHeader.isEmpty()) {
            this.acceptHeader = acceptHeader;
        } else {
            this.acceptHeader = DEFAULT_ACCEPT_HEADER;
        }

        this.contentType = ContentType.parse(this.acceptHeader);
        this.fields = fields;
    }

    @Override
    public Response build(TensorIO input) throws EvaluationException {
        Builder<TensorIO, BinaryContent> builder = getBuilder();
        BinaryContent content = builder.build(input);
        return Response
                .status(200)
                .header(CONTENT_TYPE, content.getContentType().toString())
                .entity(content.getBytes())
                .build();
    }

    private Builder<TensorIO, BinaryContent> getBuilder() {
        switch (this.contentType.getMimeType()) {
            case ImageDefaults.JPG_CONTENT_TYPE_STRING:
            case ImageDefaults.PNG_CONTENT_TYPE_STRING:
            case ImageDefaults.IMG_CONTENT_TYPE_STRING:
                return new TensorIOIntoImageBinary(this.fields, this.contentType);

            case MULTIPART_MIME:
                return new TensorIOIntoMultipartBinary(
                        this.mapper,
                        this.contentType,
                        this.fields,
                        this.shouldSimplify
                );

            case HTML_MIME:
                return new TensorIOIntoHTMLBinary(
                        this.contentType,
                        this.mapper,
                        this.fields,
                        this.shouldSimplify
                );

            case DEFAULT_ACCEPT_HEADER:
            case JSON_MIME:
                return new TensorIOIntoJsonBinary(mapper, shouldSimplify);
            default:
                throw new EvaluationException(String.format("Accept header '%s' not supported...", acceptHeader));
        }
    }
}
