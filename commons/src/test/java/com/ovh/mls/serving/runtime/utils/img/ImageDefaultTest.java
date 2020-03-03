package com.ovh.mls.serving.runtime.utils.img;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class ImageDefaultTest {

    ////////////////////////////
    ///// Grayscale Images /////
    ////////////////////////////

    @Test
    void testGetImgChanelProperties2DGrayscale() {
        // List of shapes of the same properties
        List<int[]> shapeList = List.of(
                new int[]{-1, -1},
                new int[]{28, -1},
                new int[]{-1, 28},
                new int[]{40, 50}
        );

        for (int[] shape : shapeList) {
            List<ImgChanelProperties> propC = ImageDefaults.getImgChanelProperties(shape);
            assertNull(ImageDefaults.guessChanelIndex(shape));
            assertTrue(ImageDefaults.guessIsGrayScale(shape));
            assertFalse(ImageDefaults.guessIsBatch(shape));
            assertEquals(
                    List.of(ImgChanelProperties.GRAY_SCALE),
                    propC
            );
            assertEquals(
                    List.of(ImgProperties.HEIGHT, ImgProperties.WIDTH),
                    ImageDefaults.getImgProperties(shape)
            );
        }
    }

    @Test
    void testGetImgChanelProperties4DGrayscale() {
        // List of shapes of the same properties
        List<int[]> shapeList = List.of(
                new int[]{-1, -1, -1, 1},
                new int[]{10, -1, -1, 1},
                new int[]{1, -1, -1, 1},
                new int[]{10, 25, -1, 1},
                new int[]{10, -1, 3, 1},
                new int[]{-1, 25, -1, 1},
                new int[]{-1, -1, 3, 1},
                new int[]{-1, 25, 32, 1}
        );

        for (int[] shape : shapeList) {
            System.out.println(String.format("Evaluating %s", Arrays.toString(shape)));
            List<ImgChanelProperties> propC = ImageDefaults.getImgChanelProperties(shape);
            assertEquals(3, ImageDefaults.guessChanelIndex(shape));
            assertTrue(ImageDefaults.guessIsGrayScale(shape));
            assertTrue(ImageDefaults.guessIsBatch(shape));
            assertEquals(
                    List.of(ImgChanelProperties.GRAY_SCALE),
                    propC
            );
            assertEquals(
                    List.of(ImgProperties.BATCH_SIZE, ImgProperties.HEIGHT, ImgProperties.WIDTH, ImgProperties.CHANEL),
                    ImageDefaults.getImgProperties(shape)
            );
        }
    }

    @Test
    void testGetImgChanelProperties3DGrayscale() {
        // List of shapes of the same properties
        List<int[]> shapeList = List.of(
                new int[]{-1, -1, 1},
                new int[]{1, -1, 1},
                new int[]{25, -1, 1},
                new int[]{-1, 50, 1},
                new int[]{25, -1, 1},
                new int[]{-1, 50, 1},
                new int[]{25, 32, 1}
        );

        for (int[] shape : shapeList) {
            List<ImgChanelProperties> propC = ImageDefaults.getImgChanelProperties(shape);
            assertEquals(2, ImageDefaults.guessChanelIndex(shape));
            assertTrue(ImageDefaults.guessIsGrayScale(shape));
            assertFalse(ImageDefaults.guessIsBatch(shape));
            assertEquals(
                    List.of(ImgChanelProperties.GRAY_SCALE),
                    propC
            );
            assertEquals(
                    List.of(ImgProperties.HEIGHT, ImgProperties.WIDTH, ImgProperties.CHANEL),
                    ImageDefaults.getImgProperties(shape)
            );
        }
    }


    //////////////////////
    ///// RGB Images /////
    //////////////////////

    @Test
    void testGetImgChanelProperties3DRgbChanelLast() {
        // List of shapes of the same properties
        List<int[]> shapeList = List.of(
                new int[]{-1, -1, 3},
                new int[]{-1, -1, 3},
                new int[]{-1, -1, 3},
                new int[]{25, -1, 3},
                new int[]{-1, 50, 3},
                new int[]{25, -1, 3},
                new int[]{-1, 50, 3},
                new int[]{25, 32, 3}
        );

        for (int[] shape : shapeList) {
            List<ImgChanelProperties> propC = ImageDefaults.getImgChanelProperties(shape);
            assertEquals(2, ImageDefaults.guessChanelIndex(shape));
            assertFalse(ImageDefaults.guessIsGrayScale(shape));
            assertFalse(ImageDefaults.guessIsBatch(shape));
            assertEquals(
                    List.of(ImgChanelProperties.RED, ImgChanelProperties.GREEN, ImgChanelProperties.BLUE),
                    propC
            );
            assertEquals(
                    List.of(ImgProperties.HEIGHT, ImgProperties.WIDTH, ImgProperties.CHANEL),
                    ImageDefaults.getImgProperties(shape)
            );
        }
    }

    @Test
    void testGetImgChanelProperties3DRgbChanelFirst() {
        // List of shapes of the same properties
        List<int[]> shapeList = List.of(
                new int[]{3, -1, -1},
                new int[]{3, -1, -1},
                new int[]{3, -1, -1},
                new int[]{3, 25, -1},
                new int[]{3, -1, 50},
                new int[]{3, 25, -1},
                new int[]{3, -1, 50},
                new int[]{3, 25, 32}
        );

        for (int[] shape : shapeList) {
            List<ImgChanelProperties> propC = ImageDefaults.getImgChanelProperties(shape);
            assertEquals(0, ImageDefaults.guessChanelIndex(shape));
            assertFalse(ImageDefaults.guessIsGrayScale(shape));
            assertFalse(ImageDefaults.guessIsBatch(shape));
            assertEquals(
                    List.of(ImgChanelProperties.RED, ImgChanelProperties.GREEN, ImgChanelProperties.BLUE),
                    propC
            );
            assertEquals(
                    List.of(ImgProperties.CHANEL, ImgProperties.HEIGHT, ImgProperties.WIDTH),
                    ImageDefaults.getImgProperties(shape)
            );
        }
    }

    @Test
    void testGetImgChanelProperties4DRgb() {
        // List of shapes of the same properties
        List<int[]> shapeList = List.of(
                new int[]{-1, -1, -1, 3},
                new int[]{10, -1, -1, 3},
                new int[]{10, -1, -1, 3},
                new int[]{10, 25, -1, 3},
                new int[]{10, -1, 3, 3},
                new int[]{-1, 25, -1, 3},
                new int[]{-1, -1, 3, 3},
                new int[]{-1, 25, 32, 3}
        );

        for (int[] shape : shapeList) {
            System.out.println(String.format("Evaluating %s", Arrays.toString(shape)));
            assertEquals(3, ImageDefaults.guessChanelIndex(shape));
            assertFalse(ImageDefaults.guessIsGrayScale(shape));
            assertTrue(ImageDefaults.guessIsBatch(shape));
            List<ImgChanelProperties> propC = ImageDefaults.getImgChanelProperties(shape);
            assertEquals(
                    List.of(ImgChanelProperties.RED, ImgChanelProperties.GREEN, ImgChanelProperties.BLUE),
                    propC
            );
            assertEquals(
                    List.of(ImgProperties.BATCH_SIZE, ImgProperties.HEIGHT, ImgProperties.WIDTH, ImgProperties.CHANEL),
                    ImageDefaults.getImgProperties(shape)
            );
        }
    }

}
