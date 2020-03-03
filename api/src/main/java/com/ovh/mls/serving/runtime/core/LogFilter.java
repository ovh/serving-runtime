package com.ovh.mls.serving.runtime.core;

import io.prometheus.client.Summary;

import javax.annotation.Priority;
import javax.servlet.http.HttpServletRequest;
import javax.ws.rs.container.ContainerRequestContext;
import javax.ws.rs.container.ContainerRequestFilter;
import javax.ws.rs.container.ContainerResponseContext;
import javax.ws.rs.container.ContainerResponseFilter;
import javax.ws.rs.container.ResourceInfo;
import javax.ws.rs.core.Context;
import javax.ws.rs.ext.Provider;


@Provider
@Priority(1)
public class LogFilter implements ContainerRequestFilter, ContainerResponseFilter {
    private static final ThreadLocal<Long> threadLocal = new ThreadLocal<>();

    private static final Summary LATENCY = Summary.build()
        .name("evaluator_api_request_latency_ms")
        .help("Request latency in ms.")
        .labelNames("class", "method", "status")
        .quantile(0.5, 0.05)
        .quantile(0.9, 0.01)
        .quantile(0.99, 0.001)
        .register();

    @Context
    private ResourceInfo resourceInfo;

    @Context
    private HttpServletRequest request;

    @Override
    public void filter(ContainerRequestContext requestContext) {
        LogFilter.threadLocal.set(System.currentTimeMillis());
    }

    @Override
    public void filter(ContainerRequestContext requestContext, ContainerResponseContext responseContext) {
        Long start = LogFilter.threadLocal.get();

        if (start == null) {
            // In case of error, we don't enter on the first filter
            start = System.currentTimeMillis();
        }
        long time = System.currentTimeMillis() - start;

        if (resourceInfo.getResourceClass() != null) {
            LATENCY.labels(
                resourceInfo.getResourceClass().getSimpleName(),
                resourceInfo.getResourceMethod().getName(),
                String.valueOf(responseContext.getStatus())
            ).observe(time);
        }
    }
}
