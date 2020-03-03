package com.ovh.mls.serving.runtime.swagger;


import io.swagger.v3.oas.annotations.Hidden;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.PathParam;
import javax.ws.rs.Produces;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import javax.ws.rs.core.UriInfo;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;

/**
 * Swagger UI
 */
@Path("/")
@Hidden
@Produces(MediaType.TEXT_HTML)
public class SwaggerHomeResource {
    private static final ClassLoader contextClassLoader = Thread.currentThread().getContextClassLoader();

    @Context
    private UriInfo uriInfo;

    @GET
    public Response viewHome() throws URISyntaxException {
        if (!uriInfo.getAbsolutePath().toString().endsWith("/")) {
            return Response.temporaryRedirect(new URI(uriInfo.getAbsolutePath().toString() + "/")).build();
        }

        return Response
            .ok(contextClassLoader.getResourceAsStream("swagger/swagger.html"))
            .build();
    }

    @GET
    @Path("{file}{ext:(.js|.css)}")
    @Produces("text/css")
    public InputStream renderFiler(@PathParam("file") String file, @PathParam("ext") String ext) {
        return contextClassLoader.getResourceAsStream("swagger/" + file + ext);
    }
}
