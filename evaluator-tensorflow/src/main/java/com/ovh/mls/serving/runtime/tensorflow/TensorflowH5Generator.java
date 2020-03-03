package com.ovh.mls.serving.runtime.tensorflow;

import com.google.common.collect.Lists;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.EvaluatorGenerator;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorGenerator;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.typesafe.config.Config;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.RandomStringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.SavedModelBundle;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static com.ovh.mls.serving.runtime.tensorflow.TensorflowEvaluator.DEFAULT_TAG_TENSORFLOW;

@IncludeAsEvaluatorGenerator(extension = "h5")
public class TensorflowH5Generator implements EvaluatorGenerator {
    private static final Logger LOGGER = LoggerFactory.getLogger(TensorflowH5Generator.class);

    private static void convertH5(String binary, String input, String output)
        throws InterruptedException, IOException, EvaluatorException {

        List<String> commands = Lists.newArrayList(binary, input, output);

        ProcessBuilder processBuilder = new ProcessBuilder(commands);

        Process process = processBuilder.start();

        process.waitFor();

        if (process.exitValue() != 0) {
            LOGGER.error(IOUtils.toString(process.getErrorStream(), StandardCharsets.UTF_8));
            LOGGER.error(IOUtils.toString(process.getInputStream(), StandardCharsets.UTF_8));

            throw new EvaluatorException("Error during h5 conversion");
        }
    }

    @Override
    public Evaluator generate(File file, Config evaluatorConfig) throws EvaluatorException {
        var path = String.format("tmp/%s/", RandomStringUtils.randomAlphabetic(20));

        try {
            convertH5(evaluatorConfig.getString("tensorflow.h5_converter.path"), file.getAbsolutePath(), path);
            SavedModelBundle savedModel = SavedModelBundle.load(
                String.format("%s/savedmodel/", path),
                DEFAULT_TAG_TENSORFLOW
            );
            return TensorflowEvaluator.create(savedModel);

        } catch (IOException | InterruptedException e) {
            throw new EvaluatorException("Error during read manifest", e);
        } finally {
            try {
                FileUtils.deleteDirectory(new File(path));
            } catch (IOException e) {
                LOGGER.error("Error during delete", e);
            }
        }
    }
}
