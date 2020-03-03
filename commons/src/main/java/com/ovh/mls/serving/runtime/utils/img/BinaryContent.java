package com.ovh.mls.serving.runtime.utils.img;

import org.apache.http.entity.ContentType;

public class BinaryContent {
    private String fileExtension;
    private ContentType contentType;
    private byte[] bytes;

    public BinaryContent(String extension, ContentType contentType, byte[] bytes) {
        this.fileExtension = extension;
        this.contentType = contentType;
        this.bytes = bytes;
    }

    public String getFileExtension() {
        return fileExtension;
    }

    public ContentType getContentType() {
        return contentType;
    }

    public byte[] getBytes() {
        return bytes;
    }

}
