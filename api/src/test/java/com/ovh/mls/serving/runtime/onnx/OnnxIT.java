package com.ovh.mls.serving.runtime.onnx;

import com.jayway.restassured.RestAssured;
import com.jayway.restassured.response.Header;
import com.ovh.mls.serving.runtime.IsCloseTo;
import com.ovh.mls.serving.runtime.core.ApiServer;
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

public class OnnxIT {
    private static final ClassLoader LOADER = OnnxIT.class.getClassLoader();

    @BeforeAll
    public static void startServer() throws NoSuchFieldException, IllegalAccessException {
        // Reset swagger context
        Field field = OpenApiContextLocator.class.getDeclaredField("instance");
        field.setAccessible(true);
        field.set(null, null);

        Config config = ConfigFactory.load("onnx/api.conf");
        ApiServer apiServer = new ApiServer(config);
        apiServer.start();
        RestAssured.port = 8083;
    }

    @Test
    public void testDescribe() {
        given()
            .when()
            .get("/describe")
            .then()
            .statusCode(200)
            .log().body()
            .body(
                "inputs.size()", equalTo(5),
                "outputs.size()", equalTo(2)
            );
    }

    @Test
    public void testPost() throws IOException {
        final InputStream resourceAsStream = LOADER.getResourceAsStream("onnx/batch_gen.json");
        byte[] bytes = IOUtils.toByteArray(resourceAsStream);

        given()
            .body(bytes)
            .header(new Header("Content-Type", "application/json"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .log().body()
            .body(
                "output_label.get(0)", equalTo(0),
                "output_label.get(1)", equalTo(1),
                "output_probability.get(0).get(0)", IsCloseTo.closeTo(0.80328506F, 0.001F),
                "output_probability.get(1).get(0)", IsCloseTo.closeTo(0.39885646F, 0.001F),
                "output_probability.get(0).get(1)", IsCloseTo.closeTo(0.19671494F, 0.001F),
                "output_probability.get(1).get(1)", IsCloseTo.closeTo(0.60114354F, 0.001F)
            );
    }
}
