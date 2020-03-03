package com.ovh.mls.serving.runtime;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.builder.from.TensorIOIntoResponse;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.exceptions.ErrorMessage;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.img.ImageDefaults;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeIn;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeType;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.security.SecurityScheme;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.apache.http.entity.ContentType;

import javax.inject.Inject;
import javax.ws.rs.Path;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Consumes;
import javax.ws.rs.Produces;
import javax.ws.rs.HeaderParam;
import javax.ws.rs.DefaultValue;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.InputStream;
import java.util.List;

import static javax.ws.rs.core.HttpHeaders.AUTHORIZATION;

@Path("/")
@Tag(name = "Model")
public class EvaluationResource {

    @Inject
    EvaluationService evaluationService;

    @Path("eval")
    @Operation(
        summary = "Evaluate your model"
    )
    @POST
    @Consumes({
        MediaType.APPLICATION_JSON,
        MediaType.MULTIPART_FORM_DATA,
        ImageDefaults.JPG_CONTENT_TYPE_STRING,
        ImageDefaults.PNG_CONTENT_TYPE_STRING
    })
    @Produces({
            MediaType.APPLICATION_JSON,
            MediaType.MULTIPART_FORM_DATA,
            MediaType.TEXT_HTML,
            ImageDefaults.JPG_CONTENT_TYPE_STRING,
            ImageDefaults.PNG_CONTENT_TYPE_STRING
    })
    @RequestBody(ref = "request")
    @ApiResponses({
        @ApiResponse(ref = "response", responseCode = "200"),
        @ApiResponse(
            responseCode = "400",
            content = @Content(
                schema = @Schema(implementation = ErrorMessage.class),
                examples = @ExampleObject(ref = "bad-input", name = "bad-input")
            )
        )
    })
    public Response evaluate(

        @HeaderParam("Step")
            String step,
        @DefaultValue(MediaType.APPLICATION_JSON)
        @HeaderParam("Content-Type")
            String contentTypeStr,
        @DefaultValue(MediaType.APPLICATION_JSON)
        @HeaderParam("Accept")
            String acceptHeader,
            InputStream inputStream
    )
        throws EvaluationException {

        ContentType contentType = ContentType.parse(contentTypeStr);
        ObjectMapper mapper = evaluationService.getEvaluatorUtil().getObjectMapper();
        List<Field> outputFields = evaluationService.getEvaluator().getOutputs();
        EvaluationContext context = new EvaluationContext(step);
        boolean shouldSimplify = context.shouldSimplify();
        TensorIOIntoResponse builder = new TensorIOIntoResponse(acceptHeader, mapper, outputFields, shouldSimplify);
        TensorIO output = evaluationService.evaluate(contentType, inputStream, context);
        return builder.build(output);
    }

    @GET
    @Path("describe")
    @Operation(
        summary = "Describe your model"
    )
    @Produces(MediaType.APPLICATION_JSON)
    public Evaluator describe() {
        return evaluationService.getEvaluator();
    }

    @SecurityScheme(
        name = AUTHORIZATION,
        type = SecuritySchemeType.HTTP,
        bearerFormat = "JWT",
        scheme = "bearer",
        in = SecuritySchemeIn.HEADER
    )
    public static class JWTAuth {
    }
}
