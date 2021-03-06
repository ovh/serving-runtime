package com.ovh.mls.serving.runtime.exceptions;


import com.fasterxml.jackson.databind.JsonMappingException;
import com.google.inject.Singleton;

import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.ext.ExceptionMapper;
import javax.ws.rs.ext.Provider;

import static org.eclipse.jetty.http.HttpStatus.BAD_REQUEST_400;

@Provider
@Singleton
public class JsonMappingExceptionMapper implements ExceptionMapper<JsonMappingException> {

    @Override
    public Response toResponse(JsonMappingException exception) {

        return Response
            .status(BAD_REQUEST_400)
            .entity(new ErrorMessage("Bad json provided", BAD_REQUEST_400))
            .type(MediaType.APPLICATION_JSON)
            .build();
    }
}
