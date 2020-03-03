package com.ovh.mls.serving.runtime.core.builder;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;

import java.io.InputStream;
import java.util.AbstractMap;
import java.util.Map;
import java.util.stream.Collectors;

public class InputStreamJsonIntoTensorIO implements Builder<InputStream, TensorIO> {

    private final ObjectMapper mapper;

    public InputStreamJsonIntoTensorIO(ObjectMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public TensorIO build(InputStream inputStream) throws EvaluationException {
        try {
            ObjectIntoTensor tensorBuilder = new ObjectIntoTensor();
            // The InputStream will be converted into Map<String, Object>
            TypeReference<Map<String, Object>> typeRef = new TypeReference<>() {};
            Map<String, Object> tensors = mapper.readValue(inputStream, typeRef);
            // Convert the map values (Object) into Tensors
            Map<String, Tensor> tensorsIO = tensors
                    .entrySet()
                    .stream()
                    .map(x -> new AbstractMap.SimpleEntry<>(x.getKey(), tensorBuilder.build(x.getValue())))
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

            return new TensorIO(tensorsIO);
        } catch (Exception e) {
            throw new EvaluationException(
                    "Unable to parse the given bytes into a correct json map of (name -> tensor)", e);
        }
    }
}
