package com.ovh.mls.serving.runtime.exceptions;

import com.google.inject.Singleton;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.ext.ExceptionMapper;
import javax.ws.rs.ext.Provider;

import static org.eclipse.jetty.http.HttpStatus.INTERNAL_SERVER_ERROR_500;

/**
 * Return a json error message in case of any exception not handle by the default Exception mapper
 */
@Provider
@Singleton
public class RestExceptionMapper implements ExceptionMapper<Exception> {

    private static final Logger LOGGER = LoggerFactory.getLogger(RestExceptionMapper.class);

    @Context
    private HttpServletRequest request;

    /**
     * Create a Json response from an exception
     */
    @Override
    public Response toResponse(Exception exception) {

        LOGGER.error("Error during Request {}, Exception {}", exception.getMessage());
        LOGGER.error("Error", exception);

        return Response
            // Return 500 status and Json message
            .status(INTERNAL_SERVER_ERROR_500)
            .entity(new ErrorMessage("Internal error"))
            .type(MediaType.APPLICATION_JSON)
            .build();
    }
}
