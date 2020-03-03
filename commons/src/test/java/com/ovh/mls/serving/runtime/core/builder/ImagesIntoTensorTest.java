package com.ovh.mls.serving.runtime.core.builder;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.transformer.ImageTransformerInfo;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import com.ovh.mls.serving.runtime.utils.img.ImageDefaults;
import com.ovh.mls.serving.runtime.utils.img.ImgChanelProperties;
import com.ovh.mls.serving.runtime.utils.img.ImgProperties;
import org.junit.jupiter.api.Test;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ImagesIntoTensorTest {

    @Test
    void testImagesWithSeveralHeight() {
        // simple image of 2x2 px
        BufferedImage image1 = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        BufferedImage image2 = new BufferedImage(2, 4, BufferedImage.TYPE_INT_RGB);
        ImagesIntoTensor builder = ImageTransformerInfo.fromShape(new int[]{-1, -1, -1, 3}).get().tensorBuilder();
        EvaluationException exp = assertThrows(
            EvaluationException.class,
            () -> builder.build(List.of(image1, image2))
        );
        assertEquals("To merge several images as a tensor," +
            " all images should have the same height. Found several : 2", exp.getMessage());
    }

    @Test
    void testImagesWithSeveralWidth() {
        // simple image of 2x2 px
        BufferedImage image1 = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        BufferedImage image2 = new BufferedImage(4, 2, BufferedImage.TYPE_INT_RGB);
        ImagesIntoTensor builder = ImageTransformerInfo.fromShape(new int[]{-1, -1, -1, 3}).get().tensorBuilder();
        EvaluationException exp = assertThrows(
            EvaluationException.class,
            () -> builder.build(List.of(image1, image2))
        );
        assertEquals("To merge several images as a tensor," +
            " all images should have the same width. Found several : 2", exp.getMessage());
    }

    @Test
    void testBuildDefaultImage() {
        // simple image of 2x2 px
        BufferedImage image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        image.setRGB(0, 0, new Color(0, 0, 255).getRGB());
        image.setRGB(0, 1, new Color(0, 255, 0).getRGB());
        image.setRGB(1, 0, new Color(255, 0, 0).getRGB());
        image.setRGB(1, 1, new Color(0, 0, 0).getRGB());

        ImagesIntoTensor builder = ImageTransformerInfo.fromShape(new int[]{-1, -1, -1, 3}).get().tensorBuilder();
        Tensor tensor = builder.build(List.of(image));

        assertEquals(DataType.INTEGER, tensor.getType());
        assertArrayEquals(new int[]{1, 2, 2, 3}, tensor.getShapeAsArray());

        int[][][][] data = ((int[][][][]) tensor.getData());
        assertEquals(1, data.length);

        int[][][] imageArray = data[0];
        assertArrayEquals(new int[]{0, 0, 255}, imageArray[0][0]);
        assertArrayEquals(new int[]{255, 0, 0}, imageArray[0][1]);
        assertArrayEquals(new int[]{0, 255, 0}, imageArray[1][0]);
        assertArrayEquals(new int[]{0, 0, 0}, imageArray[1][1]);
    }

    @Test
    void testBuilderCustomRGBImage() {
        // simple image of 2x2 px
        BufferedImage image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        image.setRGB(0, 0, new Color(0, 0, 255).getRGB());
        image.setRGB(0, 1, new Color(0, 255, 0).getRGB());
        image.setRGB(1, 0, new Color(255, 0, 0).getRGB());
        image.setRGB(1, 1, new Color(0, 0, 0).getRGB());
        ImagesIntoTensor builder = new ImageTransformerInfo(
            List.of(
                    ImgChanelProperties.BLUE,
                    ImgChanelProperties.RED
            ),
            List.of(
                    ImgProperties.BATCH_SIZE,
                    ImgProperties.HEIGHT,
                    ImgProperties.WIDTH,
                    ImgProperties.CHANEL
            ),
            new int[]{-1, 2, 2, 3}
        ).tensorBuilder();
        Tensor tensor = builder.build(List.of(image));

        assertEquals(DataType.INTEGER, tensor.getType());
        assertArrayEquals(new int[]{1, 2, 2, 2}, tensor.getShapeAsArray());

        int[][][][] data = ((int[][][][]) tensor.getData());
        assertEquals(1, data.length);

        int[][][] imageArray = data[0];
        assertArrayEquals(new int[]{255, 0}, imageArray[0][0]);
        assertArrayEquals(new int[]{0, 255}, imageArray[0][1]);
        assertArrayEquals(new int[]{0, 0}, imageArray[1][0]);
        assertArrayEquals(new int[]{0, 0}, imageArray[1][1]);
    }

    @Test
    void testBuilderCustomShapeImage() {
        // simple image of 2x2 px
        BufferedImage image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        image.setRGB(0, 0, new Color(0, 0, 255).getRGB());
        image.setRGB(0, 1, new Color(0, 255, 0).getRGB());
        image.setRGB(1, 0, new Color(255, 0, 0).getRGB());
        image.setRGB(1, 1, new Color(0, 0, 0).getRGB());

        ImagesIntoTensor builder = new ImageTransformerInfo(
                ImageDefaults.CHANEL_PROPERTIES_3D,
                List.of(
                        ImgProperties.HEIGHT,
                        ImgProperties.WIDTH,
                        ImgProperties.CHANEL
                ),
                new int[]{2, 2, 3}
        ).tensorBuilder();
        Tensor tensor = builder.build(List.of(image));

        assertEquals(DataType.INTEGER, tensor.getType());
        assertArrayEquals(new int[]{2, 2, 3}, tensor.getShapeAsArray());

        int[][][] imageArray = ((int[][][]) tensor.getData());

        assertArrayEquals(new int[]{0, 0, 255}, imageArray[0][0]);
        assertArrayEquals(new int[]{0, 255, 0}, imageArray[1][0]);
        assertArrayEquals(new int[]{255, 0, 0}, imageArray[0][1]);
        assertArrayEquals(new int[]{0, 0, 0}, imageArray[1][1]);
    }

}
