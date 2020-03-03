package com.ovh.mls.serving.runtime.core.builder;

import com.ovh.mls.serving.runtime.core.DataType;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.transformer.ImageTransformerInfo;
import org.junit.jupiter.api.Test;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class TensorIntoImagesTest {

    @Test
    void testBuildDefaultImage() {
        Tensor tensor = new Tensor(
            DataType.INTEGER,
            new int[]{1, 2, 2, 3},
            new int[][][][]{
                new int[][][]{
                    new int[][]{
                        new int[]{0, 0, 255},
                        new int[]{0, 255, 0}
                    },
                    new int[][]{
                        new int[]{255, 0, 0},
                        new int[]{0, 0, 0}
                    }
                }
            }
        );

        List<BufferedImage> images = ImageTransformerInfo.fromShape(tensor.getShapeAsArray())
                .get()
                .imageBuilder()
                .build(tensor);

        assertEquals(1, images.size());

        BufferedImage image = images.get(0);

        assertEquals(2, image.getWidth());
        assertEquals(2, image.getHeight());
        assertEquals(new Color(0, 0, 255), new Color(image.getRGB(0, 0)));
        assertEquals(new Color(255, 0, 0), new Color(image.getRGB(0, 1)));
        assertEquals(new Color(0, 255, 0), new Color(image.getRGB(1, 0)));
        assertEquals(new Color(0, 0, 0), new Color(image.getRGB(1, 1)));
    }

    @Test
    void testBuildDefaultImage2() {
        Tensor tensor = new Tensor(
            DataType.INTEGER,
            new int[]{1, 3, 2, 2},
            new int[][][][]{
                new int[][][]{
                    new int[][]{
                        new int[]{200, 201},
                        new int[]{202, 203}
                    },
                    new int[][]{
                        new int[]{0, 1},
                        new int[]{2, 3}
                    },
                    new int[][]{
                        new int[]{100, 101},
                        new int[]{102, 103}
                    }
                }
            }
        );

        List<BufferedImage> images = ImageTransformerInfo.fromShape(tensor.getShapeAsArray())
                .get()
                .imageBuilder()
                .build(tensor);
        assertEquals(1, images.size());

        BufferedImage image = images.get(0);

        assertEquals(2, image.getWidth());
        assertEquals(2, image.getHeight());
        assertEquals(new Color(200, 0, 100), new Color(image.getRGB(0, 0)));
        assertEquals(new Color(201, 1, 101), new Color(image.getRGB(1, 0)));
        assertEquals(new Color(202, 2, 102), new Color(image.getRGB(0, 1)));
        assertEquals(new Color(203, 3, 103), new Color(image.getRGB(1, 1)));
    }

}
