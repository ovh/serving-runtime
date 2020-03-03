package com.ovh.mls.serving.runtime.exceptions;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.WebApplicationException;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.ext.ExceptionMapper;
import javax.ws.rs.ext.Provider;


/**
 * Wrap all resteasy exception with clear error messages and status
 */
@Provider
public class WebApplicationExceptionMapper implements ExceptionMapper<WebApplicationException> {
    private static final Logger LOGGER = LoggerFactory.getLogger(WebApplicationExceptionMapper.class);

    @Override
    public Response toResponse(WebApplicationException e) {
        LOGGER.error("Error {}", e.getMessage());

        return Response
            .status(e.getResponse().getStatus())
            .entity(new ErrorMessage(e.getMessage(), e.getResponse().getStatus()))
            .type(MediaType.APPLICATION_JSON)
            .build();
    }
}
