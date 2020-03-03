package com.ovh.mls.serving.runtime.utils.img;

import com.ovh.mls.serving.runtime.core.Field;
import com.ovh.mls.serving.runtime.core.builder.TensorIntoImages;
import com.ovh.mls.serving.runtime.core.tensor.Tensor;
import com.ovh.mls.serving.runtime.core.tensor.TensorField;
import com.ovh.mls.serving.runtime.core.transformer.ImageTransformerInfo;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import org.apache.http.entity.ContentType;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

public class ImageDefaults {

    public static final String IMG_CONTENT_TYPE_STRING = "image/*";
    public static final String PNG_CONTENT_TYPE_STRING = "image/png";
    public static final String JPG_CONTENT_TYPE_STRING = "image/jpeg";
    public static final ContentType PNG_CONTENT_TYPE_CONTENT_TYPE = ContentType.parse(PNG_CONTENT_TYPE_STRING);
    public static final ContentType JPG_CONTENT_TYPE_CONTENT_TYPE = ContentType.parse(JPG_CONTENT_TYPE_STRING);

    public static final List<String> SUPPORTED_IMG_CONTENT_TYPE = List.of(
        PNG_CONTENT_TYPE_STRING,
        JPG_CONTENT_TYPE_STRING
    );

    public static final List<ImgChanelProperties> CHANEL_PROPERTIES_1D = List.of(ImgChanelProperties.GRAY_SCALE);
    public static final List<ImgChanelProperties> CHANEL_PROPERTIES_3D = List.of(
        ImgChanelProperties.RED,
        ImgChanelProperties.GREEN,
        ImgChanelProperties.BLUE
    );

    public static final List<ImgProperties> PROPERTIES_4D = List.of(
        ImgProperties.BATCH_SIZE,
        ImgProperties.HEIGHT,
        ImgProperties.WIDTH,
        ImgProperties.CHANEL
    );

    public static BufferedImage readImage(InputStream inputStream) throws EvaluationException{
        try {
            BufferedImage image = ImageIO.read(inputStream);
            if (image == null) {
                throw new EvaluationException("Image was 'null' after reading it...");
            }
            return image;
        } catch (IOException | EvaluationException e) {
            throw new EvaluationException("Unable to load image from given bytes...", e);
        }
    }

    /**
     * Infer the index of the CHANEL property in the given shape
     * It can be null in case there is no CHANEL property (ex. grayscale image)
     */
    public static Integer guessChanelIndex(int[] shape) {
        Integer index = null;
        for (int i = 0; i < shape.length; i++) {
            int value = shape[i];
            if (value == 3 || value == 1) {
                index = i;
            }
        }
        return index;
    }

    /**
     * Infer if the given shape is a Grayscale image
     * True if it is grayscale, false otherwise
     */
    public static boolean guessIsGrayScale(int[] shape) {
        Integer chanelIndex = guessChanelIndex(shape);
        if (chanelIndex != null) {
            return shape[chanelIndex] == 1;
        } else if (shape.length <= 3) {
            return true;
        } else {
            throw new EvaluationException(
                String.format(
                    "Impossible to guess the needed kind of image from the shape %s",
                    Arrays.toString(shape)
                )
            );
        }
    }

    /**
     * Infer if the given shape contains a BATCH property
     */
    public static boolean guessIsBatch(int[] shape) {
        boolean isGrayscale = guessIsGrayScale(shape);
        Integer chanelIndex = guessChanelIndex(shape);
        //int[] simplifiedShape = new TensorShape(shape).simplifyShape().getArrayShape();
        if (shape.length == 2) {
            return false;
        } else if (shape.length == 3 && isGrayscale && chanelIndex == null) {
            return true;
        } else if (shape.length == 3 && isGrayscale) {
            return false;
        } else if (shape.length == 3) {
            return false;
        } else if (shape.length == 4) {
            return true;
        } else {
            throw new EvaluationException(
                String.format("Unable to get chanel properties of image from a tensor of shape %s",
                    Arrays.toString(shape)));
        }
    }

    public static List<ImgChanelProperties> getImgChanelProperties(int[] shape) {
        if (guessIsGrayScale(shape)) {
            return ImageDefaults.CHANEL_PROPERTIES_1D;
        } else {
            return ImageDefaults.CHANEL_PROPERTIES_3D;
        }
    }

    /**
     * Infer the List of image properties from a given tensor shape
     * @param expectedShape The tensor shape from witch to infer
     */
    public static List<ImgProperties> getImgProperties(int[] expectedShape) {
        //boolean isGrayScale = guessIsGrayScale(expectedShape);
        Integer chanelIndex = guessChanelIndex(expectedShape);
        ArrayList<ImgProperties> expectedOrder = new ArrayList<>();
        if (guessIsBatch(expectedShape)) {
            expectedOrder.add(ImgProperties.BATCH_SIZE);
        }
        expectedOrder.add(ImgProperties.HEIGHT);
        expectedOrder.add(ImgProperties.WIDTH);
        expectedOrder.add(ImgProperties.CHANEL);

        List<ImgProperties> result = new ArrayList<>();
        for (int i = 0; i < expectedShape.length; i++) {
            if (chanelIndex != null && chanelIndex == i) {
                result.add(ImgProperties.CHANEL);
            } else {
                result.add(expectedOrder.remove(0));
            }
        }

        return result;
    }

    public static BinaryContent buildImage(BufferedImage image, ContentType wantedContentType) {
        try (ByteArrayOutputStream os = new ByteArrayOutputStream()) {
            String extension;
            ContentType contentType;
            if (wantedContentType.getMimeType().equals(ImageDefaults.JPG_CONTENT_TYPE_STRING)) {
                extension = "jpg";
                contentType = ImageDefaults.JPG_CONTENT_TYPE_CONTENT_TYPE;
            } else {
                extension = "png";
                contentType = ImageDefaults.PNG_CONTENT_TYPE_CONTENT_TYPE;
            }
            ImageIO.write(image, extension, os);
            return new BinaryContent(extension, contentType, os.toByteArray());

        } catch (IOException e) {
            throw new EvaluationException("Unable to convert image into bytes array...");
        }
    }

    public static Optional<TensorIntoImages> getMaybeImageBuilder(
        String tensorName,
        List<Field> fields,
        Tensor tensor
    ) {
        Optional<TensorIntoImages> maybeBuilder = fields
            .stream()
            .filter(x -> x.getName().equals(tensorName))
            .filter(x -> x instanceof TensorField)
            .findFirst()
            .flatMap(x -> ((TensorField) x).getMaybeImageTransformer())
            .map(ImageTransformerInfo::imageBuilder);

        // In case the tensor is returning early (ex: the header parameter is filled), there is no field description
        // We need to infer the image shape directly from the tensor shape
        if (maybeBuilder.isEmpty()) {
            maybeBuilder = ImageTransformerInfo
                    .fromShape(tensor.getShapeAsArray())
                    .map(ImageTransformerInfo::imageBuilder);
        }

        return maybeBuilder;
    }

    public static TensorIntoImages getImageBuilderOrFail(String tensorName, List<Field> fields, Tensor tensor) {
        Optional<TensorIntoImages> maybeBuilder = getMaybeImageBuilder(tensorName, fields, tensor);
        if (maybeBuilder.isEmpty()) {
            throw new EvaluationException(
                    String.format("Impossible to convert tensor '%s' into image", tensorName));
        }
        return maybeBuilder.get();
    }

    public static boolean supportSingleImageConversion(List<Field> fields) {
        if (fields.size() == 1) {
            Field field = fields.get(0);
            if (field instanceof TensorField) {
                return ((TensorField) field).getMaybeImageTransformer().isPresent();
            }
        }
        return false;
    }

}
