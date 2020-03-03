package com.ovh.mls.serving.runtime.core.io;

import org.apache.http.entity.ContentType;
import org.apache.tika.Tika;

import java.io.ByteArrayInputStream;
import java.io.InputStream;

public class Part {
    public String name;
    public String filename;
    public ContentType contentType;
    public byte[] content;

    private Part(String name, String filename, ContentType contentType, byte[] content) {
        this.name = name;
        this.contentType = contentType;
        this.content = content;
    }

    public InputStream getContentAsInputStream() {
        return new ByteArrayInputStream(this.content);
    }

    public static Part from(String name, String filename, String contentTypeString, byte[] content) {
        ContentType contentType = null;
        if (contentTypeString != null) {
            contentType = ContentType.parse(contentTypeString);
        }
        return from(name, filename, contentType, content);
    }

    public static Part from(String name, String filename, ContentType contentType, byte[] content) {
        // If content type null or application/octet-stream (default), detect it
        if (
            contentType == null ||
            ContentType.APPLICATION_OCTET_STREAM.getMimeType().equals(contentType.getMimeType())
        ) {
            return from(name, filename, content);
        }
        return new Part(name, filename, contentType, content);
    }

    /**
     * Create a Part by detecting the content-type
     */
    public static Part from(String name, String filename, byte[] content) {
        String mimetype = new Tika().detect(content, filename);
        ContentType detectedContentType = ContentType.parse(mimetype);
        return new Part(name, filename, detectedContentType, content);
    }
}
