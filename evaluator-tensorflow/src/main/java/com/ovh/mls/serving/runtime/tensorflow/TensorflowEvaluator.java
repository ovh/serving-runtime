package com.ovh.mls.serving.runtime.tensorflow;

import com.google.protobuf.InvalidProtocolBufferException;
import com.ovh.mls.serving.runtime.core.AbstractTensorEvaluator;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.exceptions.EvaluatorException;
import com.ovh.mls.serving.runtime.validation.NumberOnly;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.RandomStringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

@NumberOnly
public class TensorflowEvaluator extends AbstractTensorEvaluator<TensorField> {
    private static final Logger LOGGER = LoggerFactory.getLogger(TensorflowEvaluator.class);

    /**
     * Tensorflow saved models are organized between TAG (train, eval, serve)
     * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py
     */
    protected static final String DEFAULT_TAG_TENSORFLOW = "serve";

    /**
     * Each TAG contains a list of signature def that we can see as list of exported function
     * https://www.tensorflow.org/tfx/serving/signature_defs
     */
    protected static final String DEFAULT_SIGNATURE_DEF = "predict";


    private final SavedModelBundle savedModel;
    private Map<String, String> inputNameMapping;
    private Map<String, String> outputNameMapping;

    public static TensorflowEvaluator create(SavedModelBundle savedModel) {
        return create(savedModel, DEFAULT_SIGNATURE_DEF);
    }

    public static TensorflowEvaluator create(
        SavedModelBundle savedModel,
        String signatureDef
    ) {
        SignatureDef signature = getSignatureDef(signatureDef, savedModel);
        return new TensorflowEvaluator(
            savedModel,
            signature,
            getInputTensorFieldFromSignatureDef(signature),
            getOutputTensorFieldFromSignatureDef(signature),
            1
        );
    }

    protected TensorflowEvaluator(
        SavedModelBundle savedModel,
        String signatureName,
        List<TensorField> inputTensorFields,
        List<TensorField> outputTensorFields,
        int batchSize
    ) {
        this(savedModel, getSignatureDef(signatureName, savedModel), inputTensorFields, outputTensorFields, batchSize);
    }

    protected TensorflowEvaluator(
        SavedModelBundle savedModel,
        SignatureDef signature,
        List<TensorField> inputTensorFields,
        List<TensorField> outputTensorFields,
        int batchSize
    ) {
        super(inputTensorFields, outputTensorFields, batchSize);

        this.savedModel = savedModel;

        this.inputNameMapping = new HashMap<>();
        for (Map.Entry<String, TensorInfo> entry : signature.getInputsMap().entrySet()) {
            this.inputNameMapping.put(entry.getKey(), entry.getValue().getName());
        }

        this.outputNameMapping = new HashMap<>();
        for (Map.Entry<String, TensorInfo> entry : signature.getOutputsMap().entrySet()) {
            this.outputNameMapping.put(entry.getKey(), entry.getValue().getName());
        }
    }

    /**
     * Create a TensorflowEvaluator from a TensorflowEvaluatorManifest and a fileService
     *
     * @throws EvaluatorException if the manifest dont represent a valid model
     */
    public static TensorflowEvaluator create(TensorflowEvaluatorManifest manifest, String path)
        throws EvaluatorException, IOException {

        File file = new File(path, manifest.getSavedModelUri());
        SavedModelBundle savedModel;
        if (file.isDirectory()) {
            // If it's a directory
            String absolutePath = file.getAbsolutePath();
            savedModel = SavedModelBundle.load(absolutePath, DEFAULT_TAG_TENSORFLOW);
        } else {
            // If it's a file : Unzip it
            String tmpPath = String.format("tmp/%s/", RandomStringUtils.randomAlphabetic(20));
            ;
            var inputStream = FileUtils.openInputStream(file);
            try {
                unzipSavedModel(inputStream, tmpPath);
                savedModel = SavedModelBundle.load(tmpPath, DEFAULT_TAG_TENSORFLOW);
            } catch (IOException e) {
                throw new EvaluatorException("Error during saved model deserialization");
            } finally {
                try {
                    FileUtils.deleteDirectory(new File(tmpPath));
                } catch (IOException e) {
                    LOGGER.error("Error during delete", e);
                }
            }
        }

        return new TensorflowEvaluator(
            savedModel,
            DEFAULT_SIGNATURE_DEF,
            manifest.getInputs(),
            manifest.getOutputs(),
            manifest.getBatchSize()
        );

    }

    private static MetaGraphDef getGraphDefFromSavedModel(SavedModelBundle savedModel) throws EvaluatorException {
        try {
            return MetaGraphDef.newBuilder().mergeFrom(savedModel.metaGraphDef()).build();
        } catch (InvalidProtocolBufferException e) {
            throw new EvaluatorException("Error during saved model graph def deserialization");
        }
    }

    private static SignatureDef getSignatureDef(String name, SavedModelBundle savedModel) {
        MetaGraphDef graph = getGraphDefFromSavedModel(savedModel);
        SignatureDef signature = graph.getSignatureDefMap().get(name);
        if (signature == null) {
            signature = graph.getSignatureDefMap().entrySet().iterator().next().getValue();
        }
        return signature;
    }

    private static List<TensorField> getInputTensorFieldFromSignatureDef(SignatureDef signatureDef)
        throws EvaluatorException {

        return getTensorFieldFromMap(
            signatureDef.getInputsMap()
        );
    }

    private static List<TensorField> getOutputTensorFieldFromSignatureDef(SignatureDef signatureDef)
        throws EvaluatorException {

        return getTensorFieldFromMap(
            signatureDef.getOutputsMap()
        );
    }

    public static List<TensorField> getTensorFieldFromMap(Map<String, TensorInfo> map) throws EvaluatorException {
        return map
            .entrySet()
            .stream()
            .map(entry -> {
                String name = entry.getKey();
                TensorInfo info = entry.getValue();
                int[] shape = new int[info.getTensorShape().getDimCount()];
                for (int i = 0; i < shape.length; i++) {
                    shape[i] = Long.valueOf(info.getTensorShape().getDim(i).getSize()).intValue();
                }

                return new TensorField(
                    name,
                    getDatatype(info),
                    shape,
                    new ArrayList<>()
                );
            })
            .sorted(Comparator.comparing(TensorField::getName))
            .collect(Collectors.toList());
    }

    private static DataType getDatatype(TensorInfo info) throws EvaluatorException {
        switch (info.getDtype()) {
            case DT_INT8:
            case DT_UINT8:
            case DT_INT16:
            case DT_UINT16:
            case DT_INT32:
            case DT_UINT32:
                return DataType.INTEGER;
            case DT_INT64:
            case DT_UINT64:
                return DataType.LONG;
            case DT_STRING:
                return DataType.STRING;
            case DT_FLOAT:
                return DataType.FLOAT;
            case DT_DOUBLE:
                return DataType.DOUBLE;
            case DT_BOOL:
                return DataType.BOOLEAN;
            default:
                throw new EvaluatorException("Type not handle: " + info.getDtype().toString());
        }
    }

    private static void unzipSavedModel(InputStream savedModelZip, String path) throws IOException {
        // Load the zip file
        ZipInputStream zipFile = new ZipInputStream(savedModelZip);
        byte[] buffer = new byte[2048];
        ZipEntry entry;
        while ((entry = zipFile.getNextEntry()) != null) {
            String fileName = entry.getName();

            File newFile = new File(path, fileName);

            if (entry.isDirectory()) {
                boolean mkdirs = newFile.mkdirs();

                if (!mkdirs) {
                    throw new IOException(String.format("Cant create folder %s", newFile));
                }
                continue;
            }

            FileOutputStream fileOutputStream = new FileOutputStream(newFile);

            int len;

            while ((len = zipFile.read(buffer)) > 0) {
                fileOutputStream.write(buffer, 0, len);
            }

            fileOutputStream.close();
        }
    }

    @Override
    protected TensorIO evaluateTensor(TensorIO tensorIO) throws EvaluationException {
        var session = this.savedModel.session();
        var runner = session.runner();

        // Build input
        this.getInputTensorField().forEach(tensorField -> {
            String name = tensorField.getName();
            Object tensorData = tensorIO.getTensor(name).getData();
            String mappingName = this.inputNameMapping.get(name);
            Tensor tensor = Tensor.create(tensorData);
            runner.feed(mappingName, tensor);
        });

        // Declare outputs
        this.getOutputTensorField().forEach(tensorField -> {
            String name = tensorField.getName();
            String mappingName = this.outputNameMapping.get(name);
            runner.fetch(mappingName);
        });

        // Run
        List<Tensor<?>> results = runner.run();

        TensorIO output = new TensorIO();
        // Transfer output
        for (int i = 0; i < this.getOutputTensorField().size(); i++) {
            var outputBuilder = this.getOutputTensorField().get(i);
            Tensor<?> tensor = results.get(i);
            TensorIO outputTensor = getOutput(outputBuilder, tensor);
            output.merge(outputTensor);
        }

        // Free C memory
        results.forEach(Tensor::close);

        return output;
    }

    private static TensorIO getOutput(TensorField tensorField, Tensor<?> outputTensor) {
        long[] outputShape = outputTensor.shape();

        // converting shape long[] of tensorflow into int[]
        int[] outputShapeInt = new int[outputShape.length];
        for (int i = 0; i < outputShape.length; i++) {
            outputShapeInt[i] = (int) outputShape[i];
        }

        DataType dataType = tensorField.getType();
        Object tensorArray = Array.newInstance(dataType.getJavaClass(), outputShapeInt);
        if (dataType != DataType.STRING) {
            outputTensor.copyTo(tensorArray);
        } else {
            // String are retrieve from byte array => Need to add a dimension to the final tensor
            int[] byteArrayShape = Arrays.copyOf(outputShapeInt, outputShapeInt.length + 1);
            Object byteTensorArray = Array.newInstance(byte.class, byteArrayShape);
            outputTensor.copyTo(byteTensorArray);
            tensorArray = byteTensorArray;
        }

        Map<String, com.ovh.mls.serving.runtime.core.tensor.Tensor> tensorMap = new HashMap<>();
        tensorMap.put(
            tensorField.getName(),
            new com.ovh.mls.serving.runtime.core.tensor.Tensor(tensorField.getType(), outputShapeInt, tensorArray)
        );
        return new TensorIO(tensorMap);
    }

}
