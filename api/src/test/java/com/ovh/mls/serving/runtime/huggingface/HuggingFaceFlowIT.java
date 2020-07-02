package com.ovh.mls.serving.runtime.huggingface;

import com.jayway.restassured.RestAssured;
import com.jayway.restassured.response.Header;
import com.ovh.mls.serving.runtime.IsCloseTo;
import com.ovh.mls.serving.runtime.core.ApiServer;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import io.swagger.v3.oas.integration.OpenApiContextLocator;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.jayway.restassured.RestAssured.given;

public class HuggingFaceFlowIT {

    @BeforeAll
    public static void startServer() throws NoSuchFieldException, IllegalAccessException {
        // Reset swagger context
        Field field = OpenApiContextLocator.class.getDeclaredField("instance");
        field.setAccessible(true);
        field.set(null, null);

        Config config = ConfigFactory.load("huggingface/flow/api.conf");
        ApiServer apiServer = new ApiServer(config);
        apiServer.start();
        RestAssured.port = 8091;
    }

    @Test
    public void testEvalString() {
        Map<String, Object> body = new HashMap<>();
        body.put("input1", List.of("A test"));
        given()
            .body(body)
            .header(new Header("Content-Type", "application/json"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .body(
                "output.get(0)",
                IsCloseTo.closeTo(1.3058515f, 1e-6f)
            );
    }
}
