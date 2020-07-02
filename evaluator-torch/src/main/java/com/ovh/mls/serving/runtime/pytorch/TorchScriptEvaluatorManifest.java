package com.ovh.mls.serving.runtime.pytorch;

import com.facebook.jni.CppException;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.ovh.mls.serving.runtime.core.EvaluatorManifest;
import com.ovh.mls.serving.runtime.core.IncludeAsEvaluatorManifest;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.ovh.mls.serving.runtime.utils.NativeUtils;
import org.pytorch.Module;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;

@IncludeAsEvaluatorManifest(type = TorchScriptEvaluatorManifest.TYPE)
public class TorchScriptEvaluatorManifest implements EvaluatorManifest {

    // Load pytorch_jni from JAR
    static {
        NativeLoader.init(libname -> {
            try {
                // Load library from filesystem (java.library.path)
                System.loadLibrary(libname);
                return true;
            } catch (UnsatisfiedLinkError e) {
                try {
                    // Load library from JAR
                    NativeUtils.loadLibraryFromJar("/" + System.mapLibraryName(libname));
                } catch (UnsatisfiedLinkError | IOException ignored) {
                    throw new EvaluatorException("Cannot load library " + libname, e);
                }
            }

            return true;
        });
        // Load OpenMP (platform dependant)
        // Linux
        try {
            NativeUtils.loadLibraryFromJar("/libgomp-7c85b1e2.so.1");
        } catch (Exception ignored) {
        }
        // Mac
        try {
            NativeLoader.loadLibrary("iomp5");
        } catch (Exception ignored) {
        }
        NativeLoader.loadLibrary("fbjni");
        NativeLoader.loadLibrary("c10");
        NativeLoader.loadLibrary("torch_cpu");
        NativeLoader.loadLibrary("torch");
        NativeLoader.loadLibrary("pytorch_jni");
    }

    private static final Logger LOGGER = LoggerFactory.getLogger(TorchScriptEvaluatorManifest.class);

    public static final String TYPE = "torch_script";

    @JsonProperty
    private String savedModelUri;

    @JsonProperty
    private List<TensorField> inputs;

    @JsonProperty
    private List<TensorField> outputs;

    @Override
    public TorchScriptEvaluator create(String path) throws EvaluatorException {
        Module module;
        try {
            module = Module.load(savedModelUri);
        } catch (CppException e1) {
            String localSavedModelUri = Paths.get(path, savedModelUri).toString();
            try {
                module = Module.load(localSavedModelUri);
            } catch (CppException e2) {
                LOGGER.error("Cannot load TorchScript {}", savedModelUri, e1);
                LOGGER.error("Cannot load TorchScript {}", localSavedModelUri, e2);
                throw new EvaluatorException(
                    String.format("Cannot load TorchScript %s or %s", savedModelUri, localSavedModelUri)
                );
            }
        }
        return new TorchScriptEvaluator(module, inputs, outputs);
    }

    @Override
    public String getType() {
        return TYPE;
    }
}
