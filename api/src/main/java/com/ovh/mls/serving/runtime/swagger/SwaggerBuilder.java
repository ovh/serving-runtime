package com.ovh.mls.serving.runtime.swagger;


import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.ovh.mls.serving.runtime.EvaluationResource;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.Interval;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.core.tensor.TensorShape;
import com.ovh.mls.serving.runtime.exceptions.ErrorMessage;
import com.ovh.mls.serving.runtime.utils.img.ImageDefaults;
import com.typesafe.config.Config;
import io.swagger.v3.jaxrs2.integration.JaxrsOpenApiContextBuilder;
import io.swagger.v3.oas.integration.OpenApiConfigurationException;
import io.swagger.v3.oas.integration.SwaggerConfiguration;
import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.examples.Example;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.media.MediaType;
import io.swagger.v3.oas.models.media.Schema;
import io.swagger.v3.oas.models.media.Content;
import io.swagger.v3.oas.models.media.ArraySchema;
import io.swagger.v3.oas.models.media.ObjectSchema;
import io.swagger.v3.oas.models.media.StringSchema;
import io.swagger.v3.oas.models.media.NumberSchema;
import io.swagger.v3.oas.models.media.IntegerSchema;
import io.swagger.v3.oas.models.media.DateSchema;
import io.swagger.v3.oas.models.media.BooleanSchema;
import io.swagger.v3.oas.models.parameters.RequestBody;
import io.swagger.v3.oas.models.responses.ApiResponse;
import io.swagger.v3.oas.models.servers.Server;

import javax.ws.rs.core.Response;
import java.util.Set;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

public class SwaggerBuilder {
    private static final ThreadLocalRandom RANDOM = ThreadLocalRandom.current();
    private static final ObjectMapper MAPPER = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    private static final String INPUT_TENSORS = "Inputs";
    private static final String OUTPUT_TENSORS = "Outputs";

    private final Config config;
    private final Evaluator<?> evaluator;

    public SwaggerBuilder(Config config, Evaluator<?> evaluator) {
        this.config = config;
        this.evaluator = evaluator;
    }

    public void build() throws JsonProcessingException, OpenApiConfigurationException {
        OpenAPI openAPI = new OpenAPI();
        Components components = new Components();

        buildRequest(components);
        buildResponse(components);

        openAPI.components(components);

        Info info = new Info()
            .title(config.getString("title"))
            .description(config.getString("description"))
            .version(config.getString("version"));

        openAPI.addServersItem(new Server().description("Main Server").url(config.getString("path")));

        openAPI.info(info);

        Set<String> resources = new HashSet<>();

        resources.add(EvaluationResource.JWTAuth.class.getName());
        resources.add(EvaluationResource.class.getName());

        SwaggerConfiguration oasConfig = new SwaggerConfiguration()
            .openAPI(openAPI)
            .cacheTTL(Long.MAX_VALUE)
            .prettyPrint(true)
            .resourceClasses(resources);

        new JaxrsOpenApiContextBuilder()
            .openApiConfiguration(oasConfig)
            .buildContext(true);
    }

    private void buildResponse(Components components) throws JsonProcessingException {
        List<? extends Field> outputs = evaluator.getOutputs();
        MediaType jsonMediaType = new MediaType();

        // We build the schema with the output fields
        jsonMediaType.schema(buildSchemaFromFields(evaluator.getOutputs()));

        TensorIO randomTensor = generateRandomTensors(outputs);
        jsonMediaType.addExamples("response-exact-tensors",
            new Example()
                .description(getDescription(
                        outputs,
                        String.format("This model takes %s tensor(s) as outputs parameters :", outputs.size()),
                        "On the example, the runtime answer the following tensors :",
                        randomTensor
                ))
                .summary("Response with exact tensors")
                .value(MAPPER.writeValueAsString(randomTensor.intoMap()))
        );
        TensorIO simpleTensor = randomTensor.simplifyAll();
        jsonMediaType.addExamples("response-simplified-tensors",
            new Example()
                .description(getDescription(
                    outputs,
                    String.format("This model takes %s tensor(s) as outputs parameters :", outputs.size()),
                    "On the example, the runtime answer the following " +
                        "tensors whose shape have been simplify :",
                        simpleTensor))
                .summary("Response with simplified tensors")
                .value(MAPPER.writeValueAsString(simpleTensor.intoMap()))
        );

        Content content = new Content();

        // Add image media type if supported by the model
        if (ImageDefaults.supportSingleImageConversion(outputs)) {
            content
                .addMediaType(ImageDefaults.PNG_CONTENT_TYPE_STRING, new MediaType().schema(buildBinarySchema()))
                .addMediaType(ImageDefaults.JPG_CONTENT_TYPE_STRING, new MediaType().schema(buildBinarySchema()));
        }

        // Add default supported media type
        content
            .addMediaType(javax.ws.rs.core.MediaType.APPLICATION_JSON, jsonMediaType)
            .addMediaType(javax.ws.rs.core.MediaType.TEXT_HTML, new MediaType().schema(buildBinarySchema()))
            .addMediaType(javax.ws.rs.core.MediaType.MULTIPART_FORM_DATA,
                    new MediaType().schema(buildMultipartSchema(outputs)));

        components
            .addResponses("response", new ApiResponse().content(content))
            // We also add an example of bad input
            .addExamples("bad-input", new Example()
                .description("Error message")
                .summary("Error message")
                .value(MAPPER.writeValueAsString(
                    new ErrorMessage(
                        "Not enough data to evaluate. Required batch size is 3",
                        Response.Status.BAD_REQUEST.getStatusCode()
                    )
                ))
            );
    }

    private void buildRequest(Components components) throws JsonProcessingException {

        List<? extends Field> inputs = evaluator.getInputs();
        MediaType jsonMediaType = new MediaType();
        jsonMediaType.schema(new Schema().$ref(INPUT_TENSORS));

        TensorIO randomTensor = generateRandomTensors(inputs);
        jsonMediaType.addExamples("query-exact-tensors",
            new Example()
                .description(getDescription(
                    inputs,
                    String.format("This model takes %s tensor(s) as input parameters :", inputs.size()),
                    "On the example, we provide the following tensors matching expected shapes ",
                    randomTensor
                ))
                .summary("Query with exact tensors")
                .value(MAPPER.writeValueAsString(randomTensor.intoMap(false)))
        );
        TensorIO simpleTensor = randomTensor.simplifyAll();
        jsonMediaType.addExamples("query-simplified-tensors",
            new Example()
                .description(getDescription(
                    inputs,
                    String.format("This model takes %s tensor(s) as input parameters :", inputs.size()),
                    "On the example, we provide the following tensors that **will be reshaped** " +
                            "by the runtime to match the expected shape :",
                    simpleTensor
                ))
                .summary("Query with simplified tensors")
                .value(MAPPER.writeValueAsString(simpleTensor.intoMap()))
        );

        Schema inputTensorsSchema = new ObjectSchema();
        evaluator
            .getInputs()
            .forEach(input -> inputTensorsSchema.addProperties(input.getName(), buildSchemaFromSingleField(input)));

        Schema outputTensorsSchema = new ObjectSchema();
        evaluator
            .getOutputs()
            .forEach(input -> outputTensorsSchema.addProperties(input.getName(), buildSchemaFromSingleField(input)));

        Content content = new Content();

        // Add image media type if supported by the model
        if (ImageDefaults.supportSingleImageConversion(inputs)) {
            content
                .addMediaType(ImageDefaults.PNG_CONTENT_TYPE_STRING, new MediaType().schema(buildBinarySchema()))
                .addMediaType(ImageDefaults.JPG_CONTENT_TYPE_STRING, new MediaType().schema(buildBinarySchema()));
        }

        // Add default supported media type
        content
            .addMediaType(javax.ws.rs.core.MediaType.APPLICATION_JSON, jsonMediaType)
            .addMediaType(javax.ws.rs.core.MediaType.MULTIPART_FORM_DATA,
                    new MediaType().schema(buildMultipartSchema(inputs)));

        components
            .addRequestBodies("request", new RequestBody().content(content))
            // Add schema
            .addSchemas(INPUT_TENSORS, inputTensorsSchema)
            .addSchemas(OUTPUT_TENSORS, outputTensorsSchema);
    }

    private static Schema<?> buildMultipartSchema(List<? extends Field> fields) {
        Schema multipartFormat = new Schema().type("object");
        Map<String, Schema<?>> schemas = new HashMap<>();
        for (Field field : fields) {
            schemas.put(field.getName(), buildBinarySchema());
        }
        multipartFormat.properties(schemas);
        return multipartFormat;
    }

    private static Schema<?> buildArraySchema(Schema<?> itemSchema) {
        return new ArraySchema().items(itemSchema);
    }

    private static Schema<?> buildBinarySchema() {
        return new StringSchema().format("binary");
    }

    private static String getDescription(
            List<? extends Field> fields,
            String description1,
            String description2,
            TensorIO tensorIO) {

        StringBuilder description  = new StringBuilder();
        description.append(String.format("%s\n", description1));
        for (Field field : fields) {
            int[] shape = new int[]{-1};
            if (field instanceof TensorField) {
                TensorField tensorField = (TensorField) field;
                shape = tensorField.getShape();
            }
            description.append(
                    String.format("- `%s` of type `%s` with shape `%s`\n",
                            field.getName(), field.getType().toString(), Arrays.toString(shape))
            );
        }

        description.append("\n\n");
        description.append("A shape with some `-1` value(s) indicates a dimension of an **arbitrary length**. ");
        description.append("Usually it is used in the first dimension to indicates an **arbitrary batch size**. ");
        description.append("In some model that are working with images ");
        description.append("it can indicate an **arbitrary width and/or height**.");
        description.append("\n\n");
        description.append(String.format("%s\n", description2));
        for (Map.Entry<String, Tensor> entry : tensorIO.getTensors().entrySet()) {
            Tensor tensor = entry.getValue();
            description.append(
                String.format(
                    "- `%s` of type `%s` with shape `%s`\n",
                    entry.getKey(),
                    tensor.getType().toString(),
                    Arrays.toString(tensor.getShapeAsArray())
                )
            );
        }

        return description.toString();
    }

    private Schema buildSchemaFromFields(List<? extends Field> fields) {
        Schema schema = new ObjectSchema();

        fields.forEach(
            input -> schema.addProperties(
                input.getName(),
                buildSchemaFromSingleField(input)
            )
        );

        return schema;
    }

    private Schema buildSchemaFromSingleField(Field field) {
        // By default take the shape of a simple column (or vector)
        int[] shape = new int[]{-1};

        if (field instanceof TensorField) {
            TensorField tensorField = (TensorField) field;
            TensorShape tensorShape = tensorField.getTensorShape();
            if (tensorShape.isScalarShape()) {
                var schema = getScalarSchemaFromField(field);;
                schema.setName(field.getName());
                return schema;
            }
            shape = tensorShape.getArrayShape();
        }

        Schema<?> lastSchema = null;
        for (int i = shape.length - 1; i >= 0; i--) {
            int dimensionLength = shape[i];
            ArraySchema arraySchema = new ArraySchema();
            // If dimension length is not undefined, set the min and max values
            if (dimensionLength > 0) {
                arraySchema.minItems(dimensionLength);
                arraySchema.maxItems(dimensionLength);
            // If dimension length is undefined and it is the first dimension (batch dimension)
            // Take the rolling window size as the minimum length of array
            } else if (i == 0) {
                arraySchema.minItems(evaluator.getRollingWindowSize());
            }
            if (lastSchema != null) {
                arraySchema.items(lastSchema);
            } else {
                arraySchema.items(getScalarSchemaFromField(field));
            }
            lastSchema = arraySchema;
        }

        return lastSchema;
    }

    private Schema getScalarSchemaFromField(Field field) {
        switch (field.getType()) {
            case DOUBLE:
                return new NumberSchema().format("double").example(getRandomValue(field));
            case FLOAT:
                return new NumberSchema().format("float").example(getRandomValue(field));
            case STRING:
                final var stringSchema = new StringSchema();

                if (!field.getValues().isEmpty()) {
                    return stringSchema
                        ._enum(field.getValues())
                        .example(getRandomValue(field));
                }

                return stringSchema;
            case INTEGER:
            case LONG:
                return new IntegerSchema().example(getRandomValue(field));
            case BOOLEAN:
                return new BooleanSchema().example(getRandomValue(field));
            case DATE:
                return new DateSchema();
        }
        return new Schema();
    }

    private Object getRandomValue(Field field) {
        // For continuous values we might have continuous domain describing where the value should lay,
        // for query generation we get one of those intervals from which we will sample values
        Optional<Interval> maybeInterval = Optional
            .ofNullable(field.getContinuousDomain())
            .filter(x -> !x.isEmpty())
            .map(x -> x.get(0));

        switch (field.getType()) {
            case DOUBLE:
                return maybeInterval
                    .map(
                        interval ->
                            RANDOM.nextDouble(interval.getLowerBound(), interval.getUpperBound())
                    )
                    .orElse(RANDOM.nextDouble());
            case FLOAT:
                return maybeInterval
                    .map(
                        interval ->
                            Double.valueOf(
                                RANDOM.nextDouble(interval.getLowerBound(), interval.getUpperBound())
                            ).floatValue()
                    )
                    .orElse(RANDOM.nextFloat());
            case STRING:
                if (field.getValues().isEmpty()) {
                    return "";
                }
                return field.getValues().get(RANDOM.nextInt(0, field.getValues().size()));
            case INTEGER:
                return maybeInterval
                    .map(
                        interval ->
                            RANDOM.nextInt(interval.getLowerBound().intValue(), interval.getUpperBound().intValue())
                    )
                    .orElse(RANDOM.nextInt());
            case LONG:
                return maybeInterval
                    .map(
                        interval ->
                            RANDOM.nextLong(interval.getLowerBound().longValue(), interval.getUpperBound().longValue())
                    )
                    .orElse(RANDOM.nextLong());
            case BOOLEAN:
                return RANDOM.nextBoolean();
            case DATE:
                return RANDOM.nextLong(System.currentTimeMillis(), 10_000);
        }
        return new Schema();
    }

    private TensorIO generateRandomTensors(List<? extends Field> fields) {
        Map<String, Tensor> result = new HashMap<>();
        for (Field field: fields) {
            Tensor tensor = generateRandomTensor(field, 1);
            result.put(field.getName(), tensor);
        }
        return new TensorIO(result);
    }

    private Tensor generateRandomTensor(Field field, int undefinedValue) {
        int[] shape = new int[]{-1};

        if (field instanceof TensorField) {
            TensorField tensorField = (TensorField) field;
            TensorShape tensorShape = tensorField.getTensorShape();
            if (tensorShape.isScalarShape()) {
                var schema = getScalarSchemaFromField(field);;
                schema.setName(field.getName());
                Object scalarValue = getRandomValue(field);
                return new Tensor(field.getType(), new int[]{}, scalarValue);
            }
            shape = Arrays.copyOf(tensorShape.getArrayShape(), tensorShape.getArrayShape().length);
        }

        // For genrating example we replace all unknown dimensions with '2'
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] < 0) {
                shape[i] = undefinedValue;
            }
        }

        Tensor tensor = new Tensor(field.getType(), shape);
        var iterator = tensor.coordIterator();
        while (iterator.hasNext()) {
            int[] coord = iterator.next();
            Object randomValue = getRandomValue(field);
            tensor.setOnCoord(randomValue, coord);
        }

        return tensor;
    }
}
