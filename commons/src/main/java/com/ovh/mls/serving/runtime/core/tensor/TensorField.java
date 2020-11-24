package com.ovh.mls.serving.runtime.core.tensor;


import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.PropertyNamingStrategy;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.transformer.ImageTransformerInfo;

import java.util.List;
import java.util.Optional;

@JsonNaming(PropertyNamingStrategy.SnakeCaseStrategy.class)
public class TensorField extends Field {

    private int[] shape;

    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private List<TensorIndex> fields;

    @JsonInclude(JsonInclude.Include.NON_NULL)
    private ImageTransformerInfo imageTransformer;

    public TensorField() {
    }

    public TensorField(String name, DataType type, int[] shape, List<TensorIndex> tensorIndexList) {
        this(name, type, new TensorShape(shape), tensorIndexList);
    }

    public TensorField(String name, DataType type, TensorShape shape, List<TensorIndex> tensorIndexList) {
        super(name, type);
        this.shape = shape.getArrayShape();
        this.fields = tensorIndexList;
        this.imageTransformer = ImageTransformerInfo.fromShape(this.shape).orElse(null);
        this.initFieldsTypesIfNeeded();
    }


    public ImageTransformerInfo getImageTransformer() {
        return imageTransformer;
    }

    @JsonIgnore
    public Optional<ImageTransformerInfo> getMaybeImageTransformer() {
        return Optional.ofNullable(imageTransformer);
    }

    public int[] getShape() {
        return this.shape;
    }

    @JsonIgnore
    public TensorShape getTensorShape() {
        return new TensorShape(shape);
    }

    public void setShape(int[] shape) {
        this.shape = shape;
    }

    public List<TensorIndex> getFields() {
        return fields;
    }

    public void setFields(List<TensorIndex> fields) {
        this.fields = fields;
        this.initFieldsTypesIfNeeded();
    }

    /**
     * Indexes field should always be of the same data type than their parents
     */
    private void initFieldsTypesIfNeeded() {
        if (this.fields != null) {
            // Setting the same type than the current
            for (TensorIndex index: this.fields) {
                if (index.getType() == null) {
                    index.setType(this.getType());
                }
            }
        }
    }

}
