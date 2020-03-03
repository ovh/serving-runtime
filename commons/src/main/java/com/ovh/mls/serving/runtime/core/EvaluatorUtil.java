package com.ovh.mls.serving.runtime.core;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.typesafe.config.Config;
import io.github.classgraph.ClassGraph;
import io.github.classgraph.ClassInfo;
import io.github.classgraph.ScanResult;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class EvaluatorUtil {
    private static final Logger LOGGER = LoggerFactory.getLogger(EvaluatorUtil.class);
    private final ObjectMapper objectMapper;
    private final Map<String, Class<? extends EvaluatorGenerator>> generators;
    private final Config evaluatorConfig;

    public EvaluatorUtil(Config evaluatorConfig) {
        this.evaluatorConfig = evaluatorConfig;
        this.objectMapper = new ObjectMapper();

        ScanResult scanResult = new ClassGraph()
            .enableAllInfo() // Scan classes, methods, fields, annotations
            .whitelistPackages("com.ovh") // Scan com.ovh and subpackages (omit to scan all packages)
            .scan();

        for (ClassInfo classInfo : scanResult.getClassesWithAnnotation(IncludeAsEvaluatorManifest.class.getName())) {
            Class<?> clazz = classInfo.loadClass();
            LOGGER.info(String.format("Registering manifest class >> %s", clazz.getName()));
            objectMapper.registerSubtypes(
                new NamedType(clazz, clazz.getAnnotation(IncludeAsEvaluatorManifest.class).type())
            );
        }

        generators = new HashMap<>();

        for (ClassInfo classInfo : scanResult.getClassesWithAnnotation(IncludeAsEvaluatorGenerator.class.getName())) {
            var clazz = (Class<? extends EvaluatorGenerator>) classInfo.loadClass();
            final String extension = clazz.getAnnotation(IncludeAsEvaluatorGenerator.class).extension();
            generators.put(extension, clazz);
        }
    }

    public ObjectMapper getObjectMapper() {
        return objectMapper;
    }

    public EvaluatorManifest deserializeManifest(String manifest) throws IOException {
        return objectMapper.readValue(manifest, EvaluatorManifest.class);
    }

    public EvaluatorManifest deserializeManifestFromIS(InputStream manifestStream) throws IOException {
        return objectMapper.readValue(manifestStream, EvaluatorManifest.class);
    }

    public Optional<Evaluator> generate(String extension, File file) throws FileNotFoundException {
        if (generators.containsKey(extension)) {
            final Class<? extends EvaluatorGenerator> aClass = generators.get(extension);

            try {
                final Evaluator evaluator = aClass.getDeclaredConstructor()
                    .newInstance()
                    .generate(file, evaluatorConfig);
                return Optional.of(evaluator);
            } catch (InstantiationException
                | IllegalAccessException
                | InvocationTargetException
                | NoSuchMethodException e) {

                LOGGER.info("Invalid generator {} for {}", aClass.getName(), extension);
            } catch (EvaluatorException e) {
                LOGGER.error("Error during evaluator creation", e);
            }
        }

        return Optional.empty();
    }

    /**
     * Search the provided path for possible Evaluators. If none is found the returned Optional is empty.
     * <p>
     * First case: the path provided matches a File, we try an instantiate an Evaluator from this file
     * Second case: the path is directory, we try and instantiate an Evaluator from each file within the directory,
     * the method stops at the first found Evaluator
     *
     * @param path path to search
     * @return optional evaluator, empty if none were found
     */
    public Optional<Evaluator> findEvaluator(String path) {
        // Scan files in order to detect how to start.
        var actual = new File(path);

        if (!actual.exists()) {
            throw new RuntimeException("Folder not found");
        }

        if (actual.isFile()) {
            return getEvaluatorFromFile(actual, actual.getParent());
        }

        var files = actual.listFiles();
        if (files == null) {
            throw new RuntimeException("Error reading folder tree");
        }

        // Iterate over files to instantiate an evaluator
        // First we search for a manifest, JSON files
        for (File file : files) {
            if (FilenameUtils.getExtension(file.getName()).equals("json")) {
                var evaluator = getEvaluatorFromFile(file, path);
                if (evaluator.isPresent()) {
                    return evaluator;
                }
            }
        }

        // if we fail to find a manifest, we move on to direct files
        for (File file : files) {
            var evaluator = getEvaluatorFromFile(file, path);
            if (evaluator.isPresent()) {
                return evaluator;
            }
        }

        return Optional.empty();
    }

    /**
     * Instantiate an Evaluator from a file.
     * <p>
     * Starts first by trying to extract a manifest from JSON files. If unsuccessful try to instantiate an Evaluator
     * from the file if its extension is supported by one of the generators.
     *
     * @param file
     * @param fileParent
     * @return
     */
    private Optional<Evaluator> getEvaluatorFromFile(File file, String fileParent) {
        var extension = FilenameUtils.getExtension(file.getName());
        // First we look for a manifest
        if (extension.equals("json")) {
            var evaluatorManifest = extractManifest(file, fileParent);
            if (evaluatorManifest.isPresent()) {
                return evaluatorManifest;
            }
        } else {
            // If we do not find a manifest, we try to directly instantiate from files (no order, first to match)
            try {
                var optionalEvaluator = this.generate(extension, file);

                if (optionalEvaluator.isPresent()) {
                    return optionalEvaluator;
                }

            } catch (FileNotFoundException e) {
                LOGGER.error("Error while reading file: {}", file.getPath());
            }
        }
        return Optional.empty();
    }

    private Optional<Evaluator> extractManifest(File file, String fileParent) {
        try {
            var fileInputStream = new FileInputStream(file);
            final EvaluatorManifest evaluatorManifest = this.deserializeManifestFromIS(fileInputStream);
            LOGGER.info("Creating the evaluator with manifest file {}", file.getPath());
            return Optional.ofNullable(evaluatorManifest.create(fileParent));
        } catch (IOException e) {
            LOGGER.info(
                "Seems like this manifest json schema is not valid, {}, error: {}", file.getPath(), e.getMessage()
            );
            return Optional.empty();
        } catch (EvaluatorException e) {
            LOGGER.info("Seems like this manifest is not valid, {}, error: {}", file.getPath(), e.getMessage());
            return Optional.empty();
        }
    }
}
