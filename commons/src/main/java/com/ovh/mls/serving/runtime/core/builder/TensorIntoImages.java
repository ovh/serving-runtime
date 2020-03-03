package com.ovh.mls.serving.runtime.core.builder;

import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.transformer.ImageTransformerInfo;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.img.ImgChanelProperties;
import com.ovh.mls.serving.runtime.utils.img.ImgProperties;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TensorIntoImages implements Builder<Tensor, List<BufferedImage>> {

    private final List<ImgProperties> shapeAttributes;
    private final List<ImgChanelProperties> chanelProperties;

    public TensorIntoImages(ImageTransformerInfo transformerInfo) {
        this.chanelProperties = transformerInfo.getPropertyChanel();
        this.shapeAttributes = transformerInfo.getImageShape();
    }

    @Override
    public List<BufferedImage> build(Tensor input) throws EvaluationException {
        final int indexOfBatchSize = shapeAttributes.indexOf(ImgProperties.BATCH_SIZE);
        final int indexOfWidth = shapeAttributes.indexOf(ImgProperties.WIDTH);
        final int indexOfHeight = shapeAttributes.indexOf(ImgProperties.HEIGHT);
        final int indexOfChanel = shapeAttributes.indexOf(ImgProperties.CHANEL);

        // Check that there is no information missing
        if (indexOfWidth < 0 || indexOfHeight < 0 || indexOfChanel < 0) {
            throw new EvaluationException(
                String.format(
                    "Shape attributes for converting a tensor into an image should contains at least these " +
                        "3 elements : %s, but was given %s",
                    Arrays.toString(List.of(ImgProperties.WIDTH, ImgProperties.HEIGHT, ImgProperties.CHANEL).toArray()),
                    Arrays.toString(shapeAttributes.toArray())
                )
            );
        }

        final int[] shape = input.getShapeAsArray();

        // Check that the length of the tensor shape is the same than the expected shape length
        if (shape.length != this.shapeAttributes.size()) {
            throw new EvaluationException(
                String.format(
                    "The length of the tensor shape should be the same than the attribute shape " +
                        "of the image converter. Tensor shape is %s, image converter attribute shape is %s",
                    Arrays.toString(shape),
                    Arrays.toString(shapeAttributes.toArray())
                )
            );
        }

        int batchSize;
        // If BATCH_SIZE is not in shape description, we define it as 1 by default
        if (indexOfBatchSize < 0) {
            batchSize = 1;
        } else {
            batchSize = shape[indexOfBatchSize];
        }
        final int width = shape[indexOfWidth];
        final int height = shape[indexOfHeight];
        final int chanelSize = shape[indexOfChanel];

        // Check that the expected chanel size is the same than the chanel size in the tensor
        if (chanelSize != chanelProperties.size()) {
            throw new EvaluationException(
                String.format(
                    "Chanel is expected to be of size %s but found tensor with chanel size of %s",
                    chanelProperties.size(),
                    chanelSize
                )
            );
        }

        List<BufferedImage> images = new ArrayList<>();
        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            final int imageType = getImageType();
            final BufferedImage image = new BufferedImage(width, height, imageType);
            for (int widthIndex = 0; widthIndex < width; widthIndex++) {
                for (int heightIndex = 0; heightIndex < height; heightIndex++) {
                    int[] coordChanel = new int[chanelSize];
                    for (int chanelIndex = 0; chanelIndex < chanelSize; chanelIndex++) {
                        int[] coordTensor = new int[shape.length];
                        for (int i = 0; i < coordTensor.length; i++) {
                            switch (shapeAttributes.get(i)) {
                                case BATCH_SIZE:
                                    coordTensor[i] = batchIndex;
                                    break;
                                case WIDTH:
                                    coordTensor[i] = widthIndex;
                                    break;
                                case HEIGHT:
                                    coordTensor[i] = heightIndex;
                                    break;
                                case CHANEL:
                                    coordTensor[i] = chanelIndex;
                                    break;
                                default:
                                    throw new IllegalStateException("Impossible to get there");
                            }
                            int coordValue = ((Number) input.getCoord(coordTensor)).intValue();
                            switch (chanelProperties.get(chanelIndex)) {
                                case GRAY_SCALE:
                                case RED:
                                    coordChanel[0] = coordValue;
                                    break;
                                case GREEN:
                                    coordChanel[1] = coordValue;
                                    break;
                                case BLUE:
                                    coordChanel[2] = coordValue;
                                    break;
                                default:
                                    throw new IllegalStateException("Impossible to get there");
                            }
                        }
                    }
                    image.getRaster().setPixel(widthIndex, heightIndex, coordChanel);
                }
            }
            images.add(image);
        }
        return images;
    }

    private int getImageType() {
        if (chanelProperties.size() == 1) {
            return BufferedImage.TYPE_BYTE_GRAY;
        } else {
            return BufferedImage.TYPE_INT_RGB;
        }
    }

}
