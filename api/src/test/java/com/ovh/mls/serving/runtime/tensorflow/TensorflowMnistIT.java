package com.ovh.mls.serving.runtime.tensorflow;

import com.jayway.restassured.RestAssured;
import com.jayway.restassured.response.Header;
import com.jayway.restassured.response.Response;
import com.ovh.mls.serving.runtime.core.ApiServer;
import com.ovh.mls.serving.runtime.core.io.Part;
import com.ovh.mls.serving.runtime.onnx.OnnxIT;
import com.ovh.mls.serving.runtime.utils.MultipartUtils;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import io.swagger.v3.oas.integration.OpenApiContextLocator;
import org.apache.commons.codec.binary.Base64;
import org.apache.commons.io.IOUtils;
import org.apache.http.entity.ContentType;
import org.eclipse.jetty.util.MultiPartOutputStream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static com.jayway.restassured.RestAssured.given;
import static org.hamcrest.Matchers.equalTo;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorflowMnistIT {

    private static final ClassLoader LOADER = OnnxIT.class.getClassLoader();

    @BeforeAll
    public static void startServer() throws NoSuchFieldException, IllegalAccessException {
        // Reset swagger context
        Field field = OpenApiContextLocator.class.getDeclaredField("instance");
        field.setAccessible(true);
        field.set(null, null);

        Config config = ConfigFactory.load("tensorflow/mnist/api.conf");
        ApiServer apiServer = new ApiServer(config);
        apiServer.start();
        RestAssured.port = 8087;
    }

    @Test
    public void testEval0Json() throws IOException {
        final InputStream resourceAsStream = LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.json");
        given()
            .body(IOUtils.toByteArray(resourceAsStream))
            .header(new Header("Content-Type", "application/json"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .header("Content-Type", "application/json;charset=utf-8")
            .body(
                "classes", equalTo("0"),
                "class_ids", equalTo(0),
                "probabilities.get(0)", equalTo(1.0F),
                "probabilities.get(1)", equalTo(0.0F),
                "probabilities.get(2)", equalTo(0.0F),
                "probabilities.get(3)", equalTo(0.0F),
                "probabilities.get(4)", equalTo(0.0F),
                "probabilities.get(5)", equalTo(0.0F),
                "probabilities.get(6)", equalTo(0.0F),
                "probabilities.get(7)", equalTo(0.0F),
                "probabilities.get(8)", equalTo(0.0F),
                "probabilities.get(9)", equalTo(0.0F)
            );
    }

    @Test
    public void testEvalMultipart() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        MultiPartOutputStream os = new MultiPartOutputStream(baos);
        String boundary = os.getBoundary();
        writeImagePngPart(os, "image", "tensorflow/mnist/inputs/0.png");
        writeImagePngPart(os, "image", "tensorflow/mnist/inputs/1.png");
        writeImagePngPart(os, "image", "tensorflow/mnist/inputs/2.png");
        writeImagePngPart(os, "image", "tensorflow/mnist/inputs/3.png");
        writeImagePngPart(os, "image", "tensorflow/mnist/inputs/4.png");
        writeImagePngPart(os, "image", "tensorflow/mnist/inputs/5.png");
        os.close();

        given()
            .body(IOUtils.toByteArray(new ByteArrayInputStream(baos.toByteArray())))
            .header(new Header("Content-Type", String.format("multipart/form-data; boundary=%s", boundary)))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .header("Content-Type", "application/json;charset=utf-8")
            .body(
                "classes.get(0)", equalTo("0"),
                "classes.get(1)", equalTo("1"),
                "classes.get(2)", equalTo("2"),
                "classes.get(3)", equalTo("3"),
                "classes.get(4)", equalTo("4"),
                "classes.get(5)", equalTo("5"),
                "class_ids.get(0)", equalTo(0),
                "class_ids.get(1)", equalTo(1),
                "class_ids.get(2)", equalTo(2),
                "class_ids.get(3)", equalTo(3),
                "class_ids.get(4)", equalTo(4),
                "class_ids.get(5)", equalTo(5)
            );
    }

    @Test
    public void testEval7() throws IOException {
        final InputStream resourceAsStream = LOADER.getResourceAsStream("tensorflow/mnist/inputs/7.png");
        given()
            .body(IOUtils.toByteArray(resourceAsStream))
            .header(new Header("Content-Type", "image/png"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .contentType("application/json")
            .body(
                "classes", equalTo("7"),
                "class_ids", equalTo(7),
                "probabilities.get(0)", equalTo(0.0F),
                "probabilities.get(1)", equalTo(0.0F),
                "probabilities.get(2)", equalTo(0.0F),
                "probabilities.get(3)", equalTo(0.0F),
                "probabilities.get(4)", equalTo(0.0F),
                "probabilities.get(5)", equalTo(0.0F),
                "probabilities.get(6)", equalTo(0.0F),
                "probabilities.get(7)", equalTo(1.0F),
                "probabilities.get(8)", equalTo(0.0F),
                "probabilities.get(9)", equalTo(0.0F)
            );
    }

    @Test
    public void testEval8() throws IOException {
        final InputStream resourceAsStream = LOADER.getResourceAsStream("tensorflow/mnist/inputs/8.png");
        given()
            .body(IOUtils.toByteArray(resourceAsStream))
            .header(new Header("Content-Type", "image/png"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .header("Content-Type", "application/json;charset=utf-8")
            .body(
                "classes", equalTo("8"),
                "class_ids", equalTo(8),
                "probabilities.get(0)", equalTo(0.0F),
                "probabilities.get(1)", equalTo(0.0F),
                "probabilities.get(2)", equalTo(0.0F),
                "probabilities.get(3)", equalTo(0.0F),
                "probabilities.get(4)", equalTo(0.0F),
                "probabilities.get(5)", equalTo(0.0F),
                "probabilities.get(6)", equalTo(0.0F),
                "probabilities.get(7)", equalTo(0.0F),
                "probabilities.get(8)", equalTo(1.0F),
                "probabilities.get(9)", equalTo(0.0F)
            );
    }

    @Test
    public void testEval9() throws IOException {
        final InputStream resourceAsStream = LOADER.getResourceAsStream("tensorflow/mnist/inputs/9.png");
        given()
            .body(IOUtils.toByteArray(resourceAsStream))
            .header(new Header("Content-Type", "image/png"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .header("Content-Type", "application/json;charset=utf-8")
            .body(
                "classes", equalTo("9"),
                "class_ids", equalTo(9),
                "probabilities.get(0)", equalTo(0.0F),
                "probabilities.get(1)", equalTo(0.0F),
                "probabilities.get(2)", equalTo(0.0F),
                "probabilities.get(3)", equalTo(0.0F),
                "probabilities.get(4)", equalTo(0.0F),
                "probabilities.get(5)", equalTo(0.0F),
                "probabilities.get(6)", equalTo(0.0F),
                "probabilities.get(7)", equalTo(0.0F),
                "probabilities.get(8)", equalTo(0.0F),
                "probabilities.get(9)", equalTo(1.0F)
            );
    }

    @Test
    public void testEval0OnHtmlOutput() throws IOException {
        final InputStream resourceAsStream = LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.png");
        final InputStream expectedOutput = LOADER.getResourceAsStream("tensorflow/mnist/outputs/0.html");
        given()
            .body(IOUtils.toByteArray(resourceAsStream))
            .header(new Header("Content-Type", "image/png"))
            .header(new Header("Accept", "text/html"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .header("Content-Type", "text/html;charset=utf-8")
            .body(equalTo(new String(IOUtils.toByteArray(expectedOutput))));
    }

    @Test
    public void testEval0OnMultipartOutput() throws IOException {
        Response response = given()
                .body(IOUtils.toByteArray(LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.png")))
                .header(new Header("Content-Type", "image/png"))
                .header(new Header("Accept", "multipart/form-data"))
                .when()
                .post("/eval");

        assertEquals(200, response.statusCode());
        String contentTypeString = response.getContentType();
        List<Part> givenParts = MultipartUtils.readParts(
                ContentType.parse(contentTypeString),
                response.getBody().asInputStream()
        );
        List<Part> wantedParts = MultipartUtils.readParts(
                ContentType.parse("multipart/form-data; boundary=jetty1735298322k66b8jwv"),
                LOADER.getResourceAsStream("tensorflow/mnist/outputs/0.multipart")
        );
        assertEquals(wantedParts.size(), givenParts.size());
        for (int i = 0; i < wantedParts.size(); i++) {
            assertEquals(wantedParts.get(i).contentType.toString(), givenParts.get(i).contentType.toString());
            assertEquals(wantedParts.get(i).name, givenParts.get(i).name);
            assertEquals(
                new String(IOUtils.toByteArray(wantedParts.get(i).getContentAsInputStream())),
                new String(IOUtils.toByteArray(givenParts.get(i).getContentAsInputStream()))
            );
        }
    }

    @Test
    public void testEval0JsonOutput() throws IOException {
        Response response = given()
                .body(IOUtils.toByteArray(LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.png")))
                .header(new Header("Content-Type", "image/png"))
                .header(new Header("Accept", "application/json"))
                .when()
                .post("/eval");

        assertEquals(200, response.statusCode());
        assertEquals("application/json;charset=utf-8", response.getContentType());
        InputStream expectedJsonIS = LOADER.getResourceAsStream("tensorflow/mnist/outputs/0.json");
        assertEquals(new String(IOUtils.toByteArray(expectedJsonIS)), response.getBody().asString());
    }

    @Test
    public void testEval0Input0JsonOutput() throws IOException {
        Response response = given()
                .body(IOUtils.toByteArray(LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.png")))
                .header(new Header("Step", "input:0"))
                .header(new Header("Content-Type", "image/png"))
                .header(new Header("Accept", "application/json"))
                .when()
                .post("/eval");

        assertEquals(200, response.statusCode());
        assertEquals("application/json;charset=utf-8", response.getContentType());
        InputStream expectedJsonIS = LOADER.getResourceAsStream("tensorflow/mnist/outputs/0.input0.json");
        assertEquals(new String(IOUtils.toByteArray(expectedJsonIS)), response.getBody().asString());
    }

    @Test
    public void testEval0Input0HtmlOutput() throws IOException {
        Response response = given()
                .body(IOUtils.toByteArray(LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.png")))
                .header(new Header("Step", "input:0"))
                .header(new Header("Content-Type", "image/png"))
                .header(new Header("Accept", "text/html; image=application/json"))
                .when()
                .post("/eval");

        assertEquals(200, response.statusCode());
        assertEquals("text/html;charset=utf-8", response.getContentType());
        InputStream expectedHtmlIS = LOADER.getResourceAsStream("tensorflow/mnist/outputs/0.input0.html");
        String body = response.getBody().asString();
        String expectedBody = IOUtils.toString(expectedHtmlIS, StandardCharsets.UTF_8);
        assertEquals(expectedBody, body);
    }

    @Test
    public void testEval0Input0HtmlPngOutput() throws IOException {
        Response response = given()
                .body(IOUtils.toByteArray(LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.png")))
                .header(new Header("Step", "input:0"))
                .header(new Header("Content-Type", "image/png"))
                .header(new Header("Accept", "text/html"))
                .when()
                .post("/eval");

        assertEquals(200, response.statusCode());
        assertEquals("text/html;charset=utf-8", response.getContentType());
        InputStream expectedHtmlIS = LOADER.getResourceAsStream("tensorflow/mnist/outputs/0.input0.png.html");
        String body = response.getBody().asString();
        String expectedBody = IOUtils.toString(expectedHtmlIS, StandardCharsets.UTF_8);
        assertEquals(expectedBody, body);
    }

    @Test
    public void testEval0Input0PngOutput() throws IOException {
        Response response = given()
                .body(IOUtils.toByteArray(LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.png")))
                .header(new Header("Step", "input:0"))
                .header(new Header("Content-Type", "image/png"))
                .header(new Header("Accept", "image/png"))
                .when()
                .post("/eval");

        assertEquals(200, response.statusCode());
        assertEquals("image/png", response.getContentType());
        byte[] expectedImageBytes = IOUtils.toByteArray(LOADER.getResourceAsStream("tensorflow/mnist/outputs/0.png"));
        byte[] foundImageBytes = IOUtils.toByteArray(response.getBody().asInputStream());
        assertEquals(
            Base64.encodeBase64String(expectedImageBytes),
            Base64.encodeBase64String(foundImageBytes)
        );
    }

    @Test
    public void testEval0Input0MultipartOutput() throws IOException {
        Response response = given()
                .body(IOUtils.toByteArray(LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.png")))
                .header(new Header("Step", "input:0"))
                .header(new Header("Content-Type", "image/png"))
                .header(new Header("Accept", "multipart/form-data; image=image/png"))
                .when()
                .post("/eval");

        assertEquals(200, response.statusCode());
        String contentTypeString = response.getContentType();
        List<Part> givenParts = MultipartUtils.readParts(
                ContentType.parse(contentTypeString),
                response.getBody().asInputStream()
        );
        assertEquals(1, givenParts.size());
        assertEquals("image", givenParts.get(0).name);
        byte[] expectedImageBytes = IOUtils.toByteArray(
                LOADER.getResourceAsStream("tensorflow/mnist/outputs/0.png"));
        assertEquals(
                Base64.encodeBase64String(expectedImageBytes),
                Base64.encodeBase64String(IOUtils.toByteArray(givenParts.get(0).getContentAsInputStream()))
        );
    }

    @Test
    public void testErrorReadingJson() throws IOException {
        // Trying to send an image with a 'Content-Type: application/json'
        final InputStream resourceAsStream = LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.png");
        given()
            .body(IOUtils.toByteArray(resourceAsStream))
            .header(new Header("Content-Type", "application/json"))
            .when()
            .post("/eval")
            .then()
            .statusCode(400)
            .contentType("application/json")
            .body(
                "status", equalTo(400),
                "message", equalTo(
                    "Unable to parse the given bytes into a correct json map of (name -> tensor)")
            );
    }

    @Test
    public void testErrorReadingImage() throws IOException {
        // Trying to send a json with a 'Content-Type: image/png'
        final InputStream resourceAsStream = LOADER.getResourceAsStream("tensorflow/mnist/inputs/0.json");
        given()
            .body(IOUtils.toByteArray(resourceAsStream))
            .header(new Header("Content-Type", "image/png"))
            .when()
            .post("/eval")
            .then()
            .statusCode(400)
            .contentType("application/json")
            .body(
                "status", equalTo(400),
                "message", equalTo("Unable to load image from given bytes...")
            );
    }

    @Test
    public void testDescribe() throws IOException {
        given()
            .when()
            .get("/describe")
            .then()
            .statusCode(200)
            .contentType("application/json")
            .body(
                "rolling_windows_size", equalTo(1),
                    "inputs.get(0).name", equalTo("image"),
                    "inputs.get(0).type", equalTo("float"),
                    "inputs.get(0).shape.get(0)", equalTo(-1),
                    "inputs.get(0).shape.get(1)", equalTo(-1),
                    "inputs.get(0).shape.get(2)", equalTo(-1),
                    "inputs.get(0).shape.get(3)", equalTo(1),
                    "outputs.get(0).name", equalTo("class_ids"),
                    "outputs.get(0).type", equalTo("long"),
                    "outputs.get(0).shape.get(0)", equalTo(-1),
                    "outputs.get(0).shape.get(1)", equalTo(1),
                    "outputs.get(1).name", equalTo("classes"),
                    "outputs.get(1).type", equalTo("string"),
                    "outputs.get(1).shape.get(0)", equalTo(-1),
                    "outputs.get(1).shape.get(1)", equalTo(1),
                    "outputs.get(2).name", equalTo("logits"),
                    "outputs.get(2).type", equalTo("float"),
                    "outputs.get(2).shape.get(0)", equalTo(-1),
                    "outputs.get(2).shape.get(1)", equalTo(10),
                    "outputs.get(3).name", equalTo("probabilities"),
                    "outputs.get(3).type", equalTo("float"),
                    "outputs.get(3).shape.get(0)", equalTo(-1),
                    "outputs.get(3).shape.get(1)", equalTo(10)
            );
    }

    private void writeImagePngPart(MultiPartOutputStream os, String paramName, String fileName) throws IOException {
        os.startPart("image/png", new String[]{
            String.format("Content-Disposition: form-data; name=\"%s\"; filename=\"%s\"", paramName, fileName)
        });
        os.write(LOADER.getResourceAsStream(fileName).readAllBytes());
    }
}
