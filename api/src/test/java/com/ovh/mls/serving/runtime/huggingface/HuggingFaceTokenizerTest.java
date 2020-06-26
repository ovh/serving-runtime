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
                "inputs.get(0).name", equalTo("input1"),
                "inputs.get(0).type", equalTo("string"),
                "inputs.get(1).name", equalTo("input2"),
                "inputs.get(1).type", equalTo("string"),
                "outputs.get(0).name", equalTo("tokens"),
                "outputs.get(0).type", equalTo("string"),
                "outputs.get(1).name", equalTo("ids"),
                "outputs.get(1).type", equalTo("integer"),
                "outputs.get(2).name", equalTo("typeIds"),
                "outputs.get(2).type", equalTo("integer"),
                "outputs.get(3).name", equalTo("specialTokensMask"),
                "outputs.get(3).type", equalTo("integer"),
                "outputs.get(4).name", equalTo("attentionMask"),
                "outputs.get(4).type", equalTo("integer")
            );
    }

    @Test
    public void testEvalString() {
        Map<String, Object> body = new HashMap<>();
        body.put("input1", List.of("This is a test"));
        given()
            .body(body)
            .header(new Header("Content-Type", "application/json"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .body(
                "ids",
                IsEqual.equalTo(List.of(83, 44, 58, 96, 83, 96, 93, 92, 55, 69, 70))
            );
    }

    @Test
    public void testEvalDualTokens() {
        Map<String, Object> body = new HashMap<>();
        body.put("input1", List.of("This is a test".split(" ")));
        body.put("input2", "And another sentence".split(" "));
        given()
            .body(body)
            .header(new Header("Content-Type", "application/json"))
            .when()
            .post("/eval")
            .then()
            .statusCode(200)
            .body(
                "ids",
                IsEqual.equalTo(List.of(
                    83, 44, 58, 96, 83, 96, 93, 92, 55, 69, 70, 83, 25, 64, 54, 93, 64, 65, 70, 91, 68, 83, 69, 55, 64,
                    70, 55, 64, 53, 55
                ))
            );
    }
}
