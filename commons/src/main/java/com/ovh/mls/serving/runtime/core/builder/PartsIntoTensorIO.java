package com.ovh.mls.serving.runtime.core.builder;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.io.Part;
import com.ovh.mls.serving.runtime.core.io.TensorIO;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.img.ImageDefaults;
import org.apache.http.entity.ContentType;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class PartsIntoTensorIO implements Builder<List<Part>, TensorIO> {

    private ObjectMapper mapper;
    private List<Field> fields;

    public PartsIntoTensorIO(ObjectMapper mapper, List<Field> fields) {
        this.mapper = mapper;
        this.fields = fields;
    }

    @Override
    public TensorIO build(List<Part> input) throws EvaluationException {
        TensorIO tensorIO = buildImageParts(input);
        tensorIO.merge(buildJsonParts(input));
        return tensorIO;
    }

    private TensorIO buildJsonParts(List<Part> input) {
        try {
            List<Part> jsonParts = input
                .stream()
                .filter(x -> ContentType.APPLICATION_JSON.getMimeType().equals(x.contentType.getMimeType()))
                .collect(Collectors.toList());

            Set<String> distinctNamesGiven = jsonParts
                .stream()
                .map(x -> x.name)
                .collect(Collectors.toSet());

            ObjectIntoTensor tensorBuilder = new ObjectIntoTensor();
            TypeReference<Object> typeRef = new TypeReference<>() {};
            Map<String, Tensor> result = new HashMap<>();

            for (String name : distinctNamesGiven) {
                List<Part> singleJsonPart = jsonParts
                    .stream()
                    .filter(x -> x.name.equals(name))
                    .collect(Collectors.toList());

                List<Object> jsons = new ArrayList<>();
                for (Part part: singleJsonPart) {
                    jsons.add(mapper.readValue(part.content, typeRef));
                }
                Tensor tensor = tensorBuilder.build(jsons);
                result.put(name, tensor);
            }
            return new TensorIO(result);
        } catch (IOException e) {
            throw new EvaluationException("Unable to load image from bytes", e);
        }
    }

    private TensorIO buildImageParts(List<Part> input) {
        try {
            List<Part> imagesPart = input
                .stream()
                .filter(x -> ImageDefaults.SUPPORTED_IMG_CONTENT_TYPE.contains(x.contentType.getMimeType()))
                .collect(Collectors.toList());

            Set<String> distinctNamesGiven = imagesPart
                .stream()
                .map(x -> x.name)
                .collect(Collectors.toSet());

            Set<String> expectedNames = fields
                .stream()
                .map(Field::getName)
                .collect(Collectors.toSet());

            Map<String, Tensor> result = new HashMap<>();

            for (String name : distinctNamesGiven) {
                if (expectedNames.contains(name)) {

                    Field field = fields.stream().filter(x -> x.getName().equals(name)).findFirst().get();
                    if (!(field instanceof TensorField)) {
                        throw new EvaluationException("Unable to feed an image as tensor on a non-tensor model");
                    }
                    TensorField tensorField = (TensorField) field;
                    ImagesIntoTensor builder = tensorField.getImageTransformer().tensorBuilder();

                    List<Part> singleImagePart = imagesPart
                        .stream()
                        .filter(x -> x.name.equals(name))
                        .collect(Collectors.toList());

                    List<BufferedImage> images = new ArrayList<>();
                    for (Part part: singleImagePart) {
                        images.add(ImageIO.read(part.getContentAsInputStream()));
                    }

                    Tensor tensor = builder.build(images);
                    result.put(name, tensor);
                }
            }
            return new TensorIO(result);
        } catch (IOException e) {
            throw new EvaluationException("Unable to load image from bytes", e);
        }
    }
}
