package com.ovh.mls.serving.runtime.utils.img;

import com.ovh.mls.serving.runtime.core.builder.Builder;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.transformer.ImageTransformerInfo;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ImagesTensorConversionTest {

    private static final ClassLoader LOADER = ImagesTensorConversionTest.class.getClassLoader();

    @Test
    void testImageConversionBackAndForward() throws IOException {
        BufferedImage input = ImageIO.read(LOADER.getResourceAsStream("./utils/img/amber.jpg"));
        int[] shape = new int[]{1, 3, 224, 224};
        Builder<List<BufferedImage>, Tensor> builder = ImageTransformerInfo.fromShape(shape).get().tensorBuilder();
        Builder<Tensor, List<BufferedImage>> builderRevert = ImageTransformerInfo.fromShape(shape).get().imageBuilder();
        Tensor tensor = builder.build(List.of(input));
        List<BufferedImage> output = builderRevert.build(tensor);
        assertTrue(compareImages(input, output.get(0)));
    }

    @Test
    void testImageConversionBackAndForwardWithResize() throws IOException {
        BufferedImage input = ImageIO.read(LOADER.getResourceAsStream("./utils/img/amber.jpg"));
        BufferedImage expectedOutput = ImageIO.read(
                LOADER.getResourceAsStream("./utils/img/amber_100px_100px.png"));
        int[] shape = new int[]{1, 3, 100, 100};
        Builder<List<BufferedImage>, Tensor> builder = ImageTransformerInfo.fromShape(shape).get().tensorBuilder();
        Builder<Tensor, List<BufferedImage>> builderRevert = ImageTransformerInfo.fromShape(shape).get().imageBuilder();
        Tensor tensor = builder.build(List.of(input));
        List<BufferedImage> output = builderRevert.build(tensor);
        assertTrue(compareImages(expectedOutput, output.get(0)));
    }

    /**
     * From https://stackoverflow.com/questions/11006394/is-there-a-simple-way-to-compare-bufferedimage-instances
     * Compares two images pixel by pixel.
     *
     * @param imgA the first image.
     * @param imgB the second image.
     * @return whether the images are both the same or not.
     */
    public static boolean compareImages(BufferedImage imgA, BufferedImage imgB) {
        // The images must be the same size.
        if (imgA.getWidth() != imgB.getWidth() || imgA.getHeight() != imgB.getHeight()) {
            return false;
        }

        int width  = imgA.getWidth();
        int height = imgA.getHeight();

        // Loop over every pixel.
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Compare the pixels for equality.
                if (imgA.getRGB(x, y) != imgB.getRGB(x, y)) {
                    return false;
                }
            }
        }

        return true;
    }



}
