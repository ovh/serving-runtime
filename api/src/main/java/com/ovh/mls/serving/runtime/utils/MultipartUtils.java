package com.ovh.mls.serving.runtime.utils;

import com.google.common.io.Files;
import com.ovh.mls.serving.runtime.core.io.Part;
import com.ovh.mls.serving.runtime.exceptions.EvaluationException;
import org.apache.http.entity.ContentType;
import org.eclipse.jetty.http.MultiPartFormInputStream;

import javax.servlet.MultipartConfigElement;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MultipartUtils {

    /**
     * Read a multipart body and create parts from it
     * @param contentType The global Content-Type of the payload
     * @param inputStream The global input stream
     * @return The List of Parts
     * @throws EvaluationException
     */
    public static List<Part> readParts(ContentType contentType, InputStream inputStream)
            throws EvaluationException {

        try {

            MultiPartFormInputStream multipartIS = new MultiPartFormInputStream(
                    inputStream,
                    contentType.toString(),
                    new MultipartConfigElement(""),
                    Files.createTempDir()
            );

            List<Part> result = new ArrayList<>();
            for (var part : multipartIS.getParts()) {
                try (InputStream partIs = part.getInputStream()) {
                    String name = part.getName();
                    String contentTypeStr = part.getContentType();
                    String filename = part.getSubmittedFileName();
                    result.add(Part.from(name, filename, contentTypeStr, partIs.readAllBytes()));
                } catch (IOException e) {
                    throw new EvaluationException("Error while reading multipart body", e);
                }
            }

            return result;

        } catch (IOException e) {
            throw new EvaluationException("Error while reading multipart body", e);
        }
    }

}
