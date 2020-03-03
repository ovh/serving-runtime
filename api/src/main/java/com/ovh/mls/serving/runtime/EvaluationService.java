package com.ovh.mls.serving.runtime;

import com.github.racc.tscg.TypesafeConfig;
import com.ovh.mls.serving.runtime.core.EvaluationContext;
import com.ovh.mls.serving.runtime.core.Evaluator;
import com.ovh.mls.serving.runtime.core.EvaluatorUtil;
import com.ovh.mls.serving.runtime.core.builder.Builder;
import com.ovh.mls.serving.runtime.core.builder.into.InputStreamIntoTensorIO;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.swagger.SwaggerBuilder;
import com.typesafe.config.Config;
import io.prometheus.client.Counter;
import io.swagger.v3.oas.integration.OpenApiConfigurationException;
import org.apache.commons.lang3.StringUtils;
import org.apache.http.entity.ContentType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import javax.inject.Singleton;
import java.io.IOException;
import java.io.InputStream;
import java.util.Optional;

@Singleton
public class EvaluationService {
    private static final Logger LOGGER = LoggerFactory.getLogger(EvaluationService.class);

    private static final Counter EVALUATOR_COUNTER = Counter.build()
        .name("evaluator_evaluation_count")
        .help("Evaluator Counter")
        .labelNames()
        .register();

    private final Evaluator evaluator;
    private final EvaluatorUtil evaluatorUtil;

    @Inject
    EvaluationService(
        @TypesafeConfig("files.path") String filePath,
        @TypesafeConfig("swagger") Config config,
        @TypesafeConfig("evaluator") Config evaluatorConfig
    ) {
        if (StringUtils.isEmpty(filePath)) {
            throw new RuntimeException("Missing Manifest Path");
        }

        this.evaluatorUtil = new EvaluatorUtil(evaluatorConfig);

        final Optional<Evaluator> optionalEvaluator = evaluatorUtil.findEvaluator(filePath);

        if (optionalEvaluator.isEmpty()) {
            throw new RuntimeException("No manifest or direct binary found or valid from files folder");
        }

        this.evaluator = optionalEvaluator.get();

        try {
            new SwaggerBuilder(config, this.evaluator).build();
        } catch (IOException | OpenApiConfigurationException e) {
            throw new RuntimeException(e);
        }
    }

    Evaluator getEvaluator() {
        return evaluator;
    }

    TensorIO evaluate(ContentType contentType, InputStream inputStream, EvaluationContext context)
        throws EvaluationException {

        // Create builder for input
        final Builder<InputStream, TensorIO> inputBuilder = new InputStreamIntoTensorIO(
            evaluatorUtil.getObjectMapper(),
            contentType,
            this.evaluator.getInputs()
        );
        // Convert an InputStream into TensorIO
        final TensorIO inputIO = inputBuilder.build(inputStream);
        return this.evaluate(inputIO, context);
    }

    private TensorIO evaluate(TensorIO tensorIO, EvaluationContext context) throws EvaluationException {
        // Get output Tensors from the model by feeding input Tensors
        final TensorIO outputIO = evaluator.evaluate(tensorIO, context);
        EVALUATOR_COUNTER.inc(context.totalEvaluation());
        return outputIO;
    }

    EvaluatorUtil getEvaluatorUtil() {
        return this.evaluatorUtil;
    }

}
