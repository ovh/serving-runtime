package com.ovh.mls.serving.runtime.core.builder;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorShape;
import com.ovh.mls.serving.runtime.core.transformer.ImageTransformerInfo;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.img.ImgChanelProperties;
import com.ovh.mls.serving.runtime.utils.img.ImgProperties;

import java.awt.Image;
import java.awt.Graphics2D;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class ImagesIntoTensor implements Builder<List<BufferedImage>, Tensor> {

    private final List<ImgChanelProperties> chanelProperties;
    private final List<ImgProperties> shapeAttributes;
    private final int expectedWidth;
    private final int expectedHeigth;

    public ImagesIntoTensor(ImageTransformerInfo transformerInfo) {
        this.shapeAttributes = transformerInfo.getImageShape();
        this.chanelProperties = transformerInfo.getPropertyChanel();
        this.expectedHeigth = transformerInfo.getExpectedHeigth();
        this.expectedWidth = transformerInfo.getExpectedWidth();
    }

    @Override
    public Tensor build(List<BufferedImage> inputsImages) {
        final int batchSize = inputsImages.size();

        // Resize image to the expected size if needed
        List<BufferedImage> imagesList = inputsImages
            .stream()
            .map(x -> {
                int currentHeight = x.getHeight();
                int currentWidth = x.getWidth();
                boolean shouldResizeHeight = this.expectedHeigth != -1 && currentHeight != this.expectedHeigth;
                boolean shouldResizeWidth = this.expectedWidth != -1 && currentWidth != this.expectedWidth;
                if (shouldResizeHeight || shouldResizeWidth) {
                    // Resize the image
                    Image scaledImg = x.getScaledInstance(this.expectedWidth, this.expectedHeigth, Image.SCALE_SMOOTH);
                    // Convert Image into Buffered Image
                    BufferedImage output = new BufferedImage(this.expectedWidth, this.expectedHeigth, x.getType());
                    // Draw the image on to the buffered image
                    Graphics2D bGr = output.createGraphics();
                    bGr.drawImage(scaledImg, 0, 0, null);
                    bGr.dispose();
                    return output;
                } else {
                    // No need for resize
                    return x;
                }
            })
        .collect(Collectors.toList());

        // Check that all images are of the same width
        final Set<Integer> allWidth = imagesList
            .stream()
            .map(BufferedImage::getWidth)
            .collect(Collectors.toSet());

        if (allWidth.size() != 1) {
            throw new EvaluationException(
                String.format(
                    "To merge several images as a tensor, all images should have the same width. Found several : %s",
                    allWidth.toArray()
                )
            );
        }

        final int width = allWidth.iterator().next();

        // Check that all images are the same height
        final Set<Integer> allHeight = imagesList
            .stream()
            .map(BufferedImage::getHeight)
            .collect(Collectors.toSet());

        if (allHeight.size() != 1) {
            throw new EvaluationException(
                String.format(
                    "To merge several images as a tensor, all images should have the same height. Found several : %s",
                    allHeight.toArray()
                )
            );
        }
        final int height = allHeight.iterator().next();

        final int chanelSize = chanelProperties.size();

        final int[] tensorShapeArray = new int[shapeAttributes.size()];
        for (int i = 0; i < shapeAttributes.size(); i++) {
            final int dimensionSize;
            switch (shapeAttributes.get(i)) {
                case BATCH_SIZE:
                    dimensionSize = batchSize;
                    break;
                case HEIGHT:
                    dimensionSize = height;
                    break;
                case WIDTH:
                    dimensionSize = width;
                    break;
                case CHANEL:
                    dimensionSize = chanelSize;
                    break;
                default:
                    throw new IllegalStateException(
                        String.format("Unknown chanel property : %s", shapeAttributes.get(i).toString()));
            }
            tensorShapeArray[i] = dimensionSize;
        }

        final Tensor tensor = new Tensor(DataType.INTEGER, new TensorShape(tensorShapeArray));
        for (int imageIndex = 0; imageIndex < imagesList.size(); imageIndex++) {
            BufferedImage image = imagesList.get(imageIndex);
            for (int widthIndex = 0; widthIndex < image.getWidth(); widthIndex++) {
                for (int heightIndex = 0; heightIndex < image.getHeight(); heightIndex++) {
                    for (int chanelIndex = 0; chanelIndex < chanelProperties.size(); chanelIndex++) {

                        final ImgChanelProperties chanProp = chanelProperties.get(chanelIndex);
                        final int chanelValue = getChanelValue(image, widthIndex, heightIndex, chanProp);
                        final int[] tensorCoordinates = new int[shapeAttributes.size()];

                        for (int i = 0; i < shapeAttributes.size(); i++) {
                            final int coordinateValue;
                            switch (shapeAttributes.get(i)) {
                                case BATCH_SIZE:
                                    coordinateValue = imageIndex;
                                    break;
                                case HEIGHT:
                                    coordinateValue = heightIndex;
                                    break;
                                case WIDTH:
                                    coordinateValue = widthIndex;
                                    break;
                                case CHANEL:
                                    coordinateValue = chanelIndex;
                                    break;
                                default:
                                    throw new IllegalStateException("Unknown TensorImageProperties");
                            }
                            tensorCoordinates[i] = coordinateValue;
                        }
                        tensor.setOnCoord(chanelValue, tensorCoordinates);
                    }
                }
            }
        }

        return tensor;
    }

    private int getChanelValue(BufferedImage img, int widthIndex, int heightIndex, ImgChanelProperties property) {
        switch (property) {
            case RED:
                return new Color(img.getRGB(widthIndex, heightIndex)).getRed();
            case BLUE:
                return new Color(img.getRGB(widthIndex, heightIndex)).getBlue();
            case GREEN:
                return new Color(img.getRGB(widthIndex, heightIndex)).getGreen();
            case GRAY_SCALE:
                int[] buffer = new int[1];
                img.getData().getPixel(widthIndex, heightIndex, buffer);
                return buffer[0];
            default:
                throw new IllegalStateException(String.format("Unknown chanel property : %s", property.toString()));
        }
    }

}
