package com.ovh.mls.serving.runtime.huggingface;

import com.jayway.restassured.RestAssured;
import com.jayway.restassured.response.Header;
import com.ovh.mls.serving.runtime.core.ApiServer;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import io.swagger.v3.oas.integration.OpenApiContextLocator;
import org.hamcrest.core.IsEqual;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.jayway.restassured.RestAssured.given;
import static org.hamcrest.Matchers.equalTo;

public class HuggingFaceTokenizerTest {

    @BeforeAll
    public static void startServer() throws NoSuchFieldException, IllegalAccessException {
        // Reset swagger context
        Field field = OpenApiContextLocator.class.getDeclaredField("instance");
        field.setAccessible(true);
        field.set(null, null);

        Config config = ConfigFactory.load("huggingface/api.conf");
        ApiServer apiServer = new ApiServer(config);
        apiServer.start();
        RestAssured.port = 8089;
    }

    @Test
    public void testDescribe() {
        given()
            .when()
            .get("/describe")
            .then()
            .statusCode(200)
            .body(
                "inputs.get(0).name", equalTo("sequence"),
                "inputs.get(0).type", equalTo("string"),
                "outputs.get(0).name", equalTo("ids"),
                "outputs.get(0).type", equalTo("integer")
            );
    }

    @Test
    public void testEval() {
        Map<String, Object> body = new HashMap<>();
        body.put("sequence", List.of("This is a test"));
        given()
            .body(body)
            .header(new Header("Content-Type", "application/json"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .body(
                "ids",
                IsEqual.equalTo(List.of(1212, 271, 265, 395))
            );
    }
}
