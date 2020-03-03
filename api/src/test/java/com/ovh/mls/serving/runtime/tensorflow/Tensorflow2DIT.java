package com.ovh.mls.serving.runtime.tensorflow;

import com.jayway.restassured.RestAssured;
import com.jayway.restassured.response.Header;
import com.ovh.mls.serving.runtime.IsCloseTo;
import com.ovh.mls.serving.runtime.core.ApiServer;
import com.ovh.mls.serving.runtime.onnx.OnnxIT;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import io.swagger.v3.oas.integration.OpenApiContextLocator;
import org.apache.commons.io.IOUtils;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;

import static com.jayway.restassured.RestAssured.given;
import static org.hamcrest.Matchers.equalTo;

public class Tensorflow2DIT {
    private static final ClassLoader LOADER = OnnxIT.class.getClassLoader();

    @BeforeAll
    public static void startServer() throws NoSuchFieldException, IllegalAccessException {
        // Reset swagger context
        Field field = OpenApiContextLocator.class.getDeclaredField("instance");
        field.setAccessible(true);
        field.set(null, null);

        Config config = ConfigFactory.load("tensorflow/2d_savedmodel/api.conf");
        ApiServer apiServer = new ApiServer(config);
        apiServer.start();
        RestAssured.port = 8085;
    }

    @Test
    public void testDescribe() {
        given()
            .when()
            .get("/describe")
            .then()
            .statusCode(200)
            .body(
                "inputs.get(0).name", equalTo("input"),
                "inputs.get(0).type", equalTo("float"),
                "outputs.get(0).name", equalTo("output"),
                "outputs.get(0).type", equalTo("float")
            );
    }

    @Test
    public void testPost() throws IOException {
        final InputStream resourceAsStream = LOADER.getResourceAsStream("tensorflow/2d_savedmodel/2d_input.json");

        given()
            .body(IOUtils.toByteArray(resourceAsStream))
            .header(new Header("Content-Type", "application/json"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .body(
                "scaled_imputed_label_predicted",
                IsCloseTo.closeTo(-0.1796778291463852F, 0.001F)
            );
    }

}
