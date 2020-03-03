package com.ovh.mls.serving.runtime;

import com.ovh.mls.serving.runtime.core.ApiServer;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;

public class Main {
    public static void main(String[] args) {
        Config config = ConfigFactory.load();

        new ApiServer(config)
            .start()
            .join();
    }
}
