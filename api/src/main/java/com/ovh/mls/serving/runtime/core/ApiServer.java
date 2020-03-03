package com.ovh.mls.serving.runtime.core;

import com.github.racc.tscg.TypesafeConfigModule;
import com.google.inject.Guice;
import com.google.inject.Injector;
import com.google.inject.Stage;
import com.google.inject.servlet.GuiceFilter;
import com.ovh.mls.serving.runtime.EvaluationResource;
import com.ovh.mls.serving.runtime.exceptions.EvaluationExceptionMapper;
import com.ovh.mls.serving.runtime.exceptions.JsonMappingExceptionMapper;
import com.ovh.mls.serving.runtime.exceptions.JsonParseExceptionMapper;
import com.ovh.mls.serving.runtime.exceptions.RestExceptionMapper;
import com.ovh.mls.serving.runtime.exceptions.WebApplicationExceptionMapper;
import com.ovh.mls.serving.runtime.swagger.SwaggerHomeResource;
import com.typesafe.config.Config;
import io.prometheus.client.exporter.MetricsServlet;
import io.prometheus.client.hotspot.DefaultExports;
import io.swagger.v3.jaxrs2.integration.resources.OpenApiResource;
import org.eclipse.jetty.server.Connector;
import org.eclipse.jetty.server.HttpConfiguration;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.servlet.FilterHolder;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.jboss.resteasy.plugins.guice.GuiceResteasyBootstrapServletContextListener;
import org.jboss.resteasy.plugins.guice.ext.RequestScopeModule;
import org.jboss.resteasy.plugins.server.servlet.HttpServletDispatcher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.ServletContextListener;
import java.net.InetSocketAddress;

import static io.swagger.v3.oas.integration.api.OpenApiContext.OPENAPI_CONTEXT_ID_DEFAULT;
import static io.swagger.v3.oas.integration.api.OpenApiContext.OPENAPI_CONTEXT_ID_KEY;

public class ApiServer {
    private static final Logger LOGGER = LoggerFactory.getLogger(ApiServer.class);

    private final Config config;
    private final Server jettyServer;
    private final ServletContextHandler servletHandler;
    private Class<? extends ServletContextListener> contextListenerClass;

    public ApiServer(Config config) {
        this.config = config;
        LOGGER.debug(config.root().render());

        // Create jetty server
        InetSocketAddress addr = new InetSocketAddress(
            config.getString("server.bind"), config.getInt("server.port"));

        jettyServer = new Server();

        // Remove jetty header
        HttpConfiguration httpConfig = new HttpConfiguration();
        httpConfig.setSendServerVersion(false);
        HttpConnectionFactory httpFactory = new HttpConnectionFactory(httpConfig);
        ServerConnector httpConnector = new ServerConnector(jettyServer, httpFactory);

        httpConnector.setPort(addr.getPort());
        httpConnector.setHost(addr.getHostName());
        jettyServer.setConnectors(new Connector[]{httpConnector});

        // Init the Context Handler
        servletHandler = new ServletContextHandler();

        // By default, We register the GuiceResteasy Context Listener
        contextListenerClass = GuiceResteasyBootstrapServletContextListener.class;
    }

    /**
     * Create the Guice Injector
     */
    private Injector createGuiceInjector(Config config) {
        return Guice.createInjector(
            Stage.PRODUCTION,
            new AppModule(config),
            TypesafeConfigModule.fromConfigWithPackage(config, "com.ovh.mls")
        );
    }

    /**
     * Start the API
     */
    public ApiServer start() {
        // Create the Guice injector
        Injector injector = createGuiceInjector(config);

        configureServlet(injector);

        startMetrics();

        try {
            jettyServer.start();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }

        return this;
    }

    public void join() {
        try {
            this.jettyServer.join();
        } catch (InterruptedException e) {
            throw new IllegalStateException(e);
        }
    }

    public void startMetrics() {
        // Init the metrics server on different port
        Server metricsServer = new Server();
        ServerConnector metricsConnector = new ServerConnector(metricsServer);
        metricsConnector.setPort(config.getInt("server.metrics.port"));
        metricsServer.setConnectors(new Connector[]{metricsConnector});
        ServletContextHandler metricsServletHandler = new ServletContextHandler();
        metricsServletHandler.addServlet(new ServletHolder(new MetricsServlet()), "/metrics");
        metricsServer.setHandler(metricsServletHandler);
        // Init default JVM Metrics
        DefaultExports.initialize();
        try {
            metricsServer.start();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private void configureServlet(Injector injector) {
        servletHandler.addEventListener(injector.getInstance(this.contextListenerClass));
        // Add Guice Filter, will bind Request and Response to Guice
        servletHandler.addFilter(new FilterHolder(GuiceFilter.class), "/*", null);
        // Add Resteasy servlet
        ServletHolder servlet = new ServletHolder(HttpServletDispatcher.class);

        // Fix swagger servlet cache
        servlet.setInitParameter(OPENAPI_CONTEXT_ID_KEY, OPENAPI_CONTEXT_ID_DEFAULT);
        servletHandler.addServlet(servlet, "/*");
        jettyServer.setHandler(servletHandler);
    }


    private static class AppModule extends RequestScopeModule {
        private final Config config;

        AppModule(
            Config config
        ) {
            this.config = config;
        }

        /**
         * Load RestEasy Providers + Resources from the package
         */
        @Override
        protected void configure() {
            super.configure();

            // We provide reflections results for others components
            bind(Config.class).toInstance(config);

            // Bind RestExceptionMapper
            bind(JsonMappingExceptionMapper.class);
            bind(JsonParseExceptionMapper.class);
            bind(WebApplicationExceptionMapper.class);
            bind(RestExceptionMapper.class);
            bind(EvaluationExceptionMapper.class);

            bind(OpenApiResource.class);
            bind(SwaggerHomeResource.class);

            bind(EvaluationResource.class);
            bind(LogFilter.class);
        }
    }

}
