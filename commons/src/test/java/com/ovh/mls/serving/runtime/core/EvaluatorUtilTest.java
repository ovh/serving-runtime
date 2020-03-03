package com.ovh.mls.serving.runtime.core;

import com.typesafe.config.ConfigFactory;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SuppressWarnings("ALL")
public class EvaluatorUtilTest {
    private static final ClassLoader LOADER = EvaluatorUtilTest.class.getClassLoader();

    @Test
    public void reflectiveDeserializationManifest() throws IOException {
        EvaluatorUtil evaluatorUtil = new EvaluatorUtil(ConfigFactory.load());
        EvaluatorManifest evaluatorManifest = evaluatorUtil.getObjectMapper()
            .readValue(LOADER.getResourceAsStream("core/test-manifest.json"), EvaluatorManifest.class);
        assertTrue(evaluatorManifest instanceof TestEvaluatorManifest);

        TestEvaluatorManifest testEvaluatorManifest = (TestEvaluatorManifest) evaluatorManifest;

        assertEquals("someTestString", testEvaluatorManifest.getTestValue());

        List<Field> inputs = testEvaluatorManifest.getInputs();
        assertEquals(1, inputs.size());
        assertEquals("int", inputs.get(0).getName());
        assertEquals(DataType.INTEGER, inputs.get(0).getType());

        assertEquals(0, testEvaluatorManifest.getOutputs().size());
    }

    @IncludeAsEvaluatorManifest(type = "test")
    public static class TestEvaluatorManifest extends AbstractEvaluatorManifest {
        private String testValue;

        public String getTestValue() {
            return testValue;
        }

        public TestEvaluatorManifest setTestValue(String testValue) {
            this.testValue = testValue;
            return this;
        }

        @Override
        public Evaluator create(String path) {
            return null;
        }

        @Override
        public String getType() {
            return "test";
        }
    }
}
