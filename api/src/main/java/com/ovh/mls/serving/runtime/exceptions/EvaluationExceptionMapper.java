package com.ovh.mls.serving.runtime.exceptions;

import com.google.inject.Singleton;

import javax.servlet.http.HttpServletRequest;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.ext.ExceptionMapper;
import javax.ws.rs.ext.Provider;


/**
 * Return a json error message in case of any exception not handle by the default Exception mapper
 */
@Provider
@Singleton
public class EvaluationExceptionMapper implements ExceptionMapper<EvaluationException> {

    @Context
    private HttpServletRequest request;

    /**
     * Create a Json response from an exception
     */
    @Override
    public Response toResponse(EvaluationException exception) {
        return Response
            // Return 500 status and Json message
            .status(Response.Status.BAD_REQUEST)
            .entity(new ErrorMessage(exception.getMessage(), Response.Status.BAD_REQUEST.getStatusCode()))
            .type(MediaType.APPLICATION_JSON)
            .build();
    }
}
