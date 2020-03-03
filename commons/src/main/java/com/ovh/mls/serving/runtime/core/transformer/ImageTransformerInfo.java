package com.ovh.mls.serving.runtime.core.transformer;

import com.fasterxml.jackson.databind.PropertyNamingStrategy;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.ovh.mls.serving.runtime.core.builder.ImagesIntoTensor;
import com.ovh.mls.serving.runtime.core.builder.TensorIntoImages;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.img.ImgChanelProperties;
import com.ovh.mls.serving.runtime.utils.img.ImgProperties;

import java.beans.Transient;
import java.util.List;
import java.util.Optional;

import static com.ovh.mls.serving.runtime.utils.img.ImageDefaults.getImgChanelProperties;
import static com.ovh.mls.serving.runtime.utils.img.ImageDefaults.getImgProperties;

@JsonNaming(PropertyNamingStrategy.SnakeCaseStrategy.class)
public class ImageTransformerInfo {

    private List<ImgProperties> imageShape;
    private List<ImgChanelProperties> propertyChanel;
    private int[] tensorShape;

    public ImageTransformerInfo() {

    }

    public ImageTransformerInfo(
            List<ImgChanelProperties> propertyChanel,
            List<ImgProperties> imageShape,
            int[] tensorShape
    ) {
        this.propertyChanel = propertyChanel;
        this.imageShape = imageShape;
        this.tensorShape = tensorShape;
    }

    public List<ImgProperties> getImageShape() {
        return imageShape;
    }

    public List<ImgChanelProperties> getPropertyChanel() {
        return propertyChanel;
    }

    @Transient
    public boolean getSupportBatch() {
        return getExpectedBatchSize() != 1;
    }

    @Transient
    public int getExpectedBatchSize() {
        if (!this.imageShape.contains(ImgProperties.BATCH_SIZE)) {
            return 1;
        }
        return this.tensorShape[imageShape.indexOf(ImgProperties.BATCH_SIZE)];
    }

    @Transient
    public int getExpectedWidth() {
        return this.tensorShape[imageShape.indexOf(ImgProperties.WIDTH)];
    }

    @Transient
    public int getExpectedHeigth() {
        return this.tensorShape[imageShape.indexOf(ImgProperties.HEIGHT)];
    }

    public static Optional<ImageTransformerInfo> fromShape(int[] expectedShape) {
        try {
            List<ImgProperties> shapeAttributes = getImgProperties(expectedShape);
            List<ImgChanelProperties> chanelProperties = getImgChanelProperties(expectedShape);
            if (
                shapeAttributes.contains(ImgProperties.HEIGHT) &&
                shapeAttributes.contains(ImgProperties.WIDTH) &&
                shapeAttributes.contains(ImgProperties.CHANEL)
            ) {
                return Optional.of(new ImageTransformerInfo(chanelProperties, shapeAttributes, expectedShape));
            } else {
                return Optional.empty();
            }
        } catch (EvaluationException e) {
            return Optional.empty();
        }
    }

    public ImagesIntoTensor tensorBuilder() {
        return new ImagesIntoTensor(this);
    }

    public TensorIntoImages imageBuilder() {
        return new TensorIntoImages(this);
    }
}
